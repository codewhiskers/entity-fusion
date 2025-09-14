# Entity Fusion — Polars Rewrite
# -------------------------------------------------------------
# Notes
# - Uses Polars for I/O, filtering, and blocking/grouping.
# - Similarity math (TF‑IDF + sparse top‑N cosine) stays in NumPy/SciPy
#   via scikit‑learn + sparse_dot_topn for performance.
# - Requires a unique identifier column `id_col` (string). If not provided,
#   we create one named "_id" from row numbers.
# - `conditions` nested dict follows the same AND/OR/leaf schema as your
#   original implementation.
# - Outputs are Polars DataFrames.
# -------------------------------------------------------------

from __future__ import annotations

import re
from collections import defaultdict, deque
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import polars as pl
from scipy.sparse import coo_matrix, lil_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sparse_dot_topn import awesome_cossim_topn, sp_matmul_topn
from tqdm import tqdm

from collections import defaultdict, deque
from itertools import combinations
from typing import Dict, List, Optional, Set, Tuple, Union

import polars as pl
from tqdm import tqdm


# -----------------------------
# Utilities
# -----------------------------


def _canonical_pair(a, b):
    """Return a stable (min, max) pair using string comparison to avoid type issues."""
    return (a, b) if str(a) <= str(b) else (b, a)


def _ensure_id_col(df: pl.DataFrame, id_col: Optional[str]) -> Tuple[pl.DataFrame, str]:
    if id_col is None:
        if "_id" in df.columns:
            raise ValueError(
                "id_col not provided and '_id' already exists in DataFrame."
            )
        df = df.with_row_count("_id")
        return df, "_id"
    if id_col not in df.columns: 
        raise ValueError(f"id_col '{id_col}' not found in DataFrame.")
    return df, id_col


# -----------------------------
# Similarity Calculator
# -----------------------------


class SimilarityCalculator:
    def initialize_vectorizer(self, similarity_method: str, data: Sequence[str]):
        if similarity_method == "numeric":
            vectorizer = TfidfVectorizer(
                tokenizer=lambda x: re.findall(r"\d+", x),
                preprocessor=None,
                lowercase=False,
            )
            return vectorizer.fit_transform(list(map(str, data)))
        elif similarity_method == "tfidf":
            vectorizer = TfidfVectorizer(
                analyzer="char_wb",
                preprocessor=None,
                lowercase=True,
                ngram_range=(2, 3),
                norm="l2",
                smooth_idf=True,
                use_idf=True,
                stop_words="english",
            )
            return vectorizer.fit_transform(list(map(str, data)))
        elif similarity_method == "exact":
            # For exact, we just return the raw values
            return list(map(str, data))
        else:
            raise ValueError(f"Unsupported similarity method: {similarity_method}")

    def create_similarity_matrix(
        self,
        group_tfidf,
        group_ids: Sequence[Union[str, int]],
        column_name: str,
        threshold: float,
        similarity_method: str,
    ) -> np.ndarray:
        if similarity_method == "exact":
            return self._create_exact_match_matrix(group_tfidf, group_ids)
        else:
            return self._create_cosine_similarity_matrix(
                group_tfidf, group_ids, column_name, threshold
            )

    def _create_exact_match_matrix(
        self, values: Sequence[str], group_ids: Sequence[Union[str, int]]
    ) -> np.ndarray:
        value_to_ids: Dict[str, List[Union[str, int]]] = defaultdict(list)
        for idx, val in zip(group_ids, values):
            value_to_ids[str(val)].append(idx)

        pairs: List[List[object]] = []
        for _, ids in value_to_ids.items():
            n = len(ids)
            if n > 1:
                for i in range(n):
                    for j in range(i + 1, n):
                        pairs.append([ids[i], ids[j], 1.0])

        if not pairs:
            return np.zeros((0, 3), dtype=object)
        return np.array(pairs, dtype=object)

    def _create_cosine_similarity_matrix(
        self,
        group_tfidf,
        group_ids: Sequence[Union[str, int]],
        column_name: str,
        threshold: float,
    ) -> np.ndarray:
        n_samples = group_tfidf.shape[0]
        large_group_threshold = 500
        chunk_size = 2000

        if n_samples > large_group_threshold:
            # Use top‑N sparse with threshold for A x A^T
            # awesome_cossim_topn expects CSR; sp_matmul_topn works with CSR/CSC
            cos_sim_sparse = sp_matmul_topn(
                group_tfidf, group_tfidf, top_n=100, threshold=threshold, n_threads=-1
            )
        else:
            cos_sim_sparse = lil_matrix((n_samples, n_samples), dtype=np.float32)
            for start in range(0, n_samples, chunk_size):
                end = min(start + chunk_size, n_samples)
                chunk = cosine_similarity(group_tfidf[start:end], group_tfidf)
                mask = chunk >= threshold
                chunk = np.where(mask, chunk, 0)
                cos_sim_sparse[start:end, :] = chunk

        return self._finalize_similarity_matrix(cos_sim_sparse, list(group_ids))

    def _finalize_similarity_matrix(
        self, cos_sim_sparse, group_ids: List[Union[str, int]]
    ) -> np.ndarray:
        coo_ = coo_matrix(cos_sim_sparse)
        rows, cols, vals = coo_.row, coo_.col, coo_.data
        mask = rows != cols
        rows, cols, vals = rows[mask], cols[mask], vals[mask]

        triples: List[List[object]] = []
        for i, j, v in zip(rows, cols, vals):
            triples.append([group_ids[i], group_ids[j], v])
        if not triples:
            return np.zeros((0, 3), dtype=object)
        return np.array(triples, dtype=object)


# -----------------------------
# Data Grouper (Polars)
# -----------------------------


class DataGrouperPolars:
    def __init__(self, df: pl.DataFrame, id_col: str):
        self.df = df
        self.id_col = id_col

    def group_dataframe(self, params: Dict, column: str) -> List[pl.DataFrame]:
        """
        Return a list of Polars sub-DataFrames according to blocking criteria.
        Filters out null/empty/placeholder values for the target column.
        """
        df_filtered = self._preprocess_column(column)
        blocking_criteria: Optional[List[str]] = params.get("blocking_criteria")

        if not blocking_criteria:
            # No blocking => single group (if at least 2 rows)
            return [df_filtered] if df_filtered.height > 1 else []

        groups: List[Tuple[object, pl.DataFrame]] = [(None, df_filtered)]
        for criterion in blocking_criteria:
            next_groups: List[Tuple[object, pl.DataFrame]] = []
            for _, gdf in groups:
                if gdf.height <= 1:
                    continue
                next_groups.extend(
                    self._apply_blocking_criterion(gdf, criterion, params, column)
                )
            groups = next_groups

        # keep only DataFrames with >1 rows
        return [gdf for _, gdf in groups if gdf.height > 1]

    def _preprocess_column(self, column: str) -> pl.DataFrame:
        # Keep rows where column is meaningful (non-null, non-empty, not in placeholders)
        placeholders = ["unknown", "nan", "none"]  # lowercased
        return self.df.filter(
            pl.col(column).is_not_null()
            & (pl.col(column).cast(pl.Utf8).str.strip_chars() != "")
            & (~pl.col(column).cast(pl.Utf8).str.to_lowercase().is_in(placeholders))
        )

    def _apply_blocking_criterion(
        self, df: pl.DataFrame, criterion: str, params: Dict, column: str
    ) -> List[Tuple[object, pl.DataFrame]]:
        if criterion == "first_letter":
            tmp = df.with_columns(
                pl.col(column).cast(pl.Utf8).str.slice(0, 1).alias("__fl")
            )
            out = []
            for key, sub in tmp.group_by("__fl", maintain_order=True):
                out.append((key[0], sub.drop("__fl")))
            return out

        elif criterion == "blocking_column":
            blocking_cols = params.get("blocking_column")
            if not blocking_cols:
                return [(None, df)]
            if isinstance(blocking_cols, str):
                blocking_cols = [blocking_cols]
            # ensure string grouping keys without nulls (use "NA")
            tmp = df
            for bc in blocking_cols:
                tmp = tmp.with_columns(
                    pl.when(pl.col(bc).is_null())
                    .then(pl.lit("NA"))
                    .otherwise(pl.col(bc).cast(pl.Utf8))
                    .alias(bc)
                )
            out = []
            for key, sub in tmp.group_by(blocking_cols, maintain_order=True):
                # key is a Row with values for each bc
                out.append((tuple(key), sub))
            return out

        else:
            raise ValueError(f"Unsupported blocking criterion: {criterion}")


# -----------------------------
# Similarity Matrix + Clustering (Polars version)
# -----------------------------





# If you already have this elsewhere, feel free to remove this helper.
def _canonical_pair(a, b):
    """Order-agnostic pair for set membership."""
    return (a, b) if str(a) <= str(b) else (b, a)



# ---------- tiny helper ----------
def _canonical_pair(a, b):
    """Order-agnostic pair for set membership."""
    return (a, b) if str(a) <= str(b) else (b, a)


class SimilarityMatrixGeneratorPolars:
    """
    Build pairwise similarity edges (by column with AND/OR logic), cluster via
    connected components, and support incremental updates against an existing
    preclustered snapshot passed in as a DataFrame.
    """

    def __init__(
        self,
        df: pl.DataFrame,
        conditions: Dict,
        id_col: Optional[str] = None,
        must_links: Optional[Set[Tuple[object, object]]] = None,
        cannot_links: Optional[Set[Tuple[object, object]]] = None,
        *,
        unclustered_sentinels: Set[object] = frozenset({"-1", -1, "unclustered", None, ""}),
        carry_forward_singletons: bool = True,
        include_old_labeled_in_run: bool = False,
        always_string_labels: bool = True,  # keep labels Utf8 across runs (recommended)
    ):
        self.df, self.id_col = _ensure_id_col(df, id_col)
        self.conditions = conditions

        self.must_links: Set[Tuple[object, object]] = set(must_links or set())
        self.cannot_links: Set[Tuple[object, object]] = set(cannot_links or set())

        self.graph: Dict[object, Set[object]] = defaultdict(set)
        self.clusters: Dict[object, Union[int, str]] = {}
        self.sim_calc = SimilarityCalculator()
        self.grouper = DataGrouperPolars(self.df, self.id_col)
        self.similarity_results: List[Tuple[str, List[List[object]]]] = []

        self.unclustered_sentinels = set(unclustered_sentinels)
        self.carry_forward_singletons = carry_forward_singletons
        self.include_old_labeled_in_run = include_old_labeled_in_run
        self.always_string_labels = always_string_labels

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------

    def create_similarity_matrices(self) -> None:
        """Compute per-column similarity results and store them in self.similarity_results."""
        column_params = self._extract_column_params(self.conditions)
        if not column_params:
            raise ValueError("No column parameters found in conditions.")

        results: List[Tuple[str, List[List[object]]]] = []

        for column, params in column_params.items():
            if column not in self.df.columns:
                raise ValueError(f"Column '{column}' not found in the dataframe.")

            column_results: List[List[object]] = []
            groups = self.grouper.group_dataframe(params, column)

            for group in tqdm(groups, desc=f"Processing blocks for column '{column}'", leave=False):
                ids = group[self.id_col].to_list()
                texts = group[column].cast(pl.Utf8).to_list()

                group_data = self.sim_calc.initialize_vectorizer(
                    params.get("similarity_method", "tfidf"), texts
                )
                threshold = 1.0 if params.get("similarity_method") == "exact" else params.get("threshold", 0.8)
                arr = self.sim_calc.create_similarity_matrix(
                    group_data, ids, column, threshold, params.get("similarity_method", "tfidf")
                )
                if arr.shape[0] > 0:
                    column_results.extend(arr.tolist())

            if column_results:
                results.append((column, column_results))

        self.similarity_results = results

    def cluster_data(self, old_label_map: Optional[Dict[object, str]] = None) -> pl.DataFrame:
        """
        Build edges per self.conditions, apply must/cannot links, find components, and label.
        Ensures consistent cluster label types and always returns a frame.
        """
        # reset per-run state
        self.graph.clear()
        self.clusters.clear()

        # Start from static/user-provided must_links; don't mutate self.must_links across runs
        must_links = set(self.must_links)

        # Add must-links for nodes that share a non-sentinel old label
        if old_label_map:
            filtered = {n: lbl for n, lbl in old_label_map.items()
                        if lbl not in self.unclustered_sentinels}
            by_label: Dict[str, List[object]] = defaultdict(list)
            for n, lbl in filtered.items():
                by_label[lbl].append(n)
            for _, nodes in by_label.items():
                for a, b in combinations(nodes, 2):
                    must_links.add(_canonical_pair(a, b))

        # Edges from condition tree
        final_edges = self._compute_edges_for_condition(self.conditions)
        # Apply must/cannot
        final_edges |= { _canonical_pair(a, b) for a, b in must_links }
        final_edges -= { _canonical_pair(a, b) for a, b in self.cannot_links }

        # Materialize graph; include isolated nodes from current df
        for node in self.df.get_column(self.id_col).to_list():
            self.graph.setdefault(node, set())
        for a, b in final_edges:
            self.graph[a].add(b)
            self.graph[b].add(a)

        # Find connected components
        self._find_clusters_from_graph()

        # Build label frame with consistent types
        id_dtype = self.df.schema[self.id_col]
        use_string_labels = self.always_string_labels or (old_label_map is not None)
        cl_dtype = pl.Utf8 if use_string_labels else pl.Int64

        if old_label_map:
            node_to_label = self._derive_final_labels_with_old(self.clusters, old_label_map)
            label_df = (
                pl.DataFrame({
                    self.id_col: pl.Series(self.id_col, list(node_to_label.keys()), dtype=id_dtype),
                    "cluster_label": pl.Series("cluster_label", list(node_to_label.values()), dtype=pl.Utf8),
                })
                if node_to_label else
                pl.DataFrame(schema={self.id_col: id_dtype, "cluster_label": pl.Utf8})
            )
        else:
            if use_string_labels and self.clusters:
                cluster_values = [str(v) for v in self.clusters.values()]
            else:
                cluster_values = list(self.clusters.values())

            label_df = (
                pl.DataFrame({
                    self.id_col: pl.Series(self.id_col, list(self.clusters.keys()), dtype=id_dtype),
                    "cluster_label": pl.Series("cluster_label", cluster_values, dtype=cl_dtype),
                })
                if self.clusters else
                pl.DataFrame(schema={self.id_col: id_dtype, "cluster_label": cl_dtype})
            )

        # Join labels; prefer newly computed one and keep exactly one 'cluster_label'
        out = self.df.join(label_df, on=self.id_col, how="left", suffix="_new")
        if "cluster_label_new" in out.columns:
            out = out.with_columns(
                pl.coalesce([pl.col("cluster_label_new"), pl.col("cluster_label")]).alias("cluster_label")
            ).drop("cluster_label_new")

        # Fill unlabeled deterministically
        null_total = out.select(pl.col("cluster_label").is_null().sum().alias("n")).item()
        if null_total > 0:
            # Determine start for "new_{k}" or integer labels
            if self.clusters:
                if use_string_labels:
                    max_val = -1
                    for v in self.clusters.values():
                        if isinstance(v, str):
                            if v.startswith("new_"):
                                try:
                                    max_val = max(max_val, int(v.split("_")[1]))
                                except (IndexError, ValueError):
                                    pass
                            else:
                                try:
                                    max_val = max(max_val, int(v))
                                except ValueError:
                                    pass
                        elif isinstance(v, (int, float)):
                            max_val = max(max_val, int(v))
                    start = max_val + 1
                else:
                    start = int(max([v for v in self.clusters.values()], default=-1)) + 1
            else:
                start = 0

            out = out.with_row_index("__rc__")
            null_rc = (
                out.filter(pl.col("cluster_label").is_null())
                .select("__rc__").to_series().to_list()
            )

            if cl_dtype == pl.Utf8:
                fill_vals = [f"new_{start + i}" for i in range(null_total)]
            else:
                fill_vals = [start + i for i in range(null_total)]

            fill_df = pl.DataFrame({
                "__rc__": pl.Series("__rc__", null_rc, dtype=pl.UInt32),   # match with_row_index dtype
                "__fill__": pl.Series("__fill__", fill_vals, dtype=cl_dtype),
            })

            out = (
                out.join(fill_df, on="__rc__", how="left")
                .with_columns(pl.coalesce([pl.col("cluster_label"), pl.col("__fill__")]).alias("cluster_label"))
                .drop(["__fill__", "__rc__"])
            )

        # Enforce final dtype and return
        if use_string_labels:
            out = out.with_columns(pl.col("cluster_label").cast(pl.Utf8))
        else:
            out = out.with_columns(pl.col("cluster_label").cast(pl.Int64))

        return out


    def _align_for_concat(self, dfs: list[pl.DataFrame]) -> list[pl.DataFrame]:
        if not dfs:
            return dfs
        # union of all column names
        all_cols = []
        seen = set()
        for df in dfs:
            for c in df.columns:
                if c not in seen:
                    seen.add(c)
                    all_cols.append(c)
        # add missing columns as Null and select same order
        out = []
        for df in dfs:
            missing = [c for c in all_cols if c not in df.columns]
            if missing:
                df = df.with_columns([pl.lit(None).alias(c) for c in missing])
            out.append(df.select(all_cols))
        return out


    def incremental_cluster(self, preclustered_df: Optional[pl.DataFrame]) -> Tuple[pl.DataFrame, pl.DataFrame]:
        """
        Incrementally update an existing preclustered snapshot with the current batch (self.df).
        Returns tuple of (combined_snapshot, batch_results).
        """

        
        # no prior snapshot: cluster new and return it as the snapshot
        if preclustered_df is None or preclustered_df.height == 0:
            clustered = self.cluster_data(old_label_map=None)
            if self.always_string_labels:
                clustered = clustered.with_columns(pl.col("cluster_label").cast(pl.Utf8))
            return clustered, clustered

        # harmonize dtypes with current batch
        id_dtype = self.df.schema[self.id_col]
        old_all = preclustered_df.with_columns(pl.col(self.id_col).cast(id_dtype))
        
        if "cluster_label" not in old_all.columns:
            raise ValueError("preclustered_df must contain a 'cluster_label' column.")
        
        if self.always_string_labels:
            old_all = old_all.with_columns(pl.col("cluster_label").cast(pl.Utf8))

        # Create sentinel mask - handle dtype issues properly
        cluster_dtype = old_all.schema["cluster_label"]
        if cluster_dtype == pl.Utf8:
            # For string columns, convert sentinels to strings and handle nulls
            sentinel_strs = [str(s) for s in self.unclustered_sentinels if s is not None]
            is_sentinel = pl.col("cluster_label").is_null() | pl.col("cluster_label").is_in(sentinel_strs)
        else:
            # For numeric columns, keep only numeric sentinels
            sentinel_nums = [s for s in self.unclustered_sentinels if isinstance(s, (int, float)) and s is not None]
            if sentinel_nums:
                is_sentinel = pl.col("cluster_label").is_null() | pl.col("cluster_label").is_in(sentinel_nums)
            else:
                is_sentinel = pl.col("cluster_label").is_null()

        old_un_sentinel = old_all.filter(is_sentinel)

        if self.carry_forward_singletons:
            counts = old_all.group_by("cluster_label").agg(pl.len().alias("n"))
            old_all_with_n = old_all.join(counts, on="cluster_label", how="left")
            old_singletons = old_all_with_n.filter(pl.col("n") == 1).drop("n")
            carry_parts = [old_un_sentinel, old_singletons]
        else:
            carry_parts = [old_un_sentinel]

        old_carry = (
            pl.concat(carry_parts, how="vertical_relaxed").unique(subset=[self.id_col])
            if carry_parts else pl.DataFrame(schema=old_all.schema)
        )

        # build old label map (non-sentinels only)
        old_labeled = old_all.filter(~is_sentinel)
        old_label_map = dict(
            zip(
                old_labeled.get_column(self.id_col).to_list(),
                old_labeled.get_column("cluster_label").to_list(),
            )
        )

        # *** FIX: Ensure schema compatibility before concatenation ***
        parts = []
        
        # Add old carry data (already has correct schema)
        if old_carry.height > 0:
            parts.append(old_carry)
        
        # Add new data - ensure it has cluster_label column (null values)
        new_data_with_schema = self.df.with_columns(pl.lit(None, dtype=cluster_dtype).alias("cluster_label"))
        parts.append(new_data_with_schema)
        
        # Add old labeled data if requested
        if self.include_old_labeled_in_run and old_labeled.height > 0:
            parts.insert(-1, old_labeled)  # Insert before new data

        # Now concatenate with compatible schemas
        parts = [p.with_columns(pl.col("cluster_label").cast(pl.Utf8)) for p in parts]
        parts = self._align_for_concat(parts)                     # ← NEW
        df_run = pl.concat(parts, how="vertical_relaxed").unique(subset=[self.id_col])

        # run clustering on the composed frame
        prev_df, prev_grouper = self.df, self.grouper
        try:
            self.df = df_run
            self.grouper = DataGrouperPolars(self.df, self.id_col)
            clustered_run = self.cluster_data(old_label_map=old_label_map)
            if self.always_string_labels:
                clustered_run = clustered_run.with_columns(pl.col("cluster_label").cast(pl.Utf8))
        finally:
            self.df, self.grouper = prev_df, prev_grouper

        # merge clustered run back into untouched old rows
        run_ids = set(clustered_run.get_column(self.id_col).to_list())
        untouched_old = old_all.filter(~pl.col(self.id_col).is_in(list(run_ids)))  # <-- add this
        # make sure we only have ONE cluster_label and it’s Utf8
        clustered_run = clustered_run.with_columns(pl.col("cluster_label").cast(pl.Utf8))
        untouched_old  = untouched_old.with_columns(pl.col("cluster_label").cast(pl.Utf8))

        to_combine = self._align_for_concat([untouched_old, clustered_run])   # ← NEW
        combined = (
            pl.concat(to_combine, how="vertical_relaxed")
            .unique(subset=[self.id_col], keep="last")
        )

        return combined, clustered_run

    # convenience alias if you prefer the original name
    def fit_incremental_from_df(self, preclustered_df: Optional[pl.DataFrame]) -> pl.DataFrame:
        return self.incremental_cluster(preclustered_df)

    # ---------------------------------------------------------------------
    # Internals
    # ---------------------------------------------------------------------

    def _extract_column_params(self, condition: Dict) -> Dict[str, Dict]:
        """Collect leaf-level column param dicts from an AND/OR tree."""
        params: Dict[str, Dict] = {}
        if "and" in condition:
            for sub in condition["and"]:
                params.update(self._extract_column_params(sub))
        elif "or" in condition:
            for sub in condition["or"]:
                params.update(self._extract_column_params(sub))
        else:
            for col, p in condition.items():
                params[col] = p
        return params

    def _compute_edges_for_condition(self, condition: Dict) -> Set[Tuple[object, object]]:
        """Compute edge set according to AND/OR composition of columns."""
        if "and" in condition:
            parts = [self._compute_edges_for_condition(c) for c in condition["and"]]
            return set.intersection(*parts) if parts else set()
        if "or" in condition:
            parts = [self._compute_edges_for_condition(c) for c in condition["or"]]
            return set.union(*parts) if parts else set()

        edges_per_col = [
            self._compute_edges_for_single_column(col, params)
            for col, params in condition.items()
        ]
        return set.intersection(*edges_per_col) if edges_per_col else set()

    def _compute_edges_for_single_column(self, column: str, params: Dict) -> Set[Tuple[object, object]]:
        threshold = 1.0 if params.get("similarity_method") == "exact" else params.get("threshold", 0.8)
        method = params.get("similarity_method", "tfidf")

        groups = self.grouper.group_dataframe(params, column)
        all_pairs: Set[Tuple[object, object]] = set()

        for group in tqdm(groups, desc=f"Matching within blocks for '{column}'", leave=False):
            ids = group[self.id_col].to_list()
            texts = group[column].cast(pl.Utf8).to_list()
            data = self.sim_calc.initialize_vectorizer(method, texts)
            arr = self.sim_calc.create_similarity_matrix(data, ids, column, threshold, method)
            for a, b, _ in arr:
                all_pairs.add(_canonical_pair(a, b))
        return all_pairs

    def _find_clusters_from_graph(self) -> None:
        """Simple BFS over the undirected graph to assign component IDs."""
        visited: Set[object] = set()
        comp_id = 0
        for node in self.graph.keys():
            if node in visited:
                continue
            q = deque([node])
            while q:
                u = q.popleft()
                if u in visited:
                    continue
                visited.add(u)
                self.clusters[u] = comp_id
                for v in self.graph[u]:
                    if v not in visited:
                        q.append(v)
            comp_id += 1

    def _derive_final_labels_with_old(
        self, clusters: Dict[object, int], old_map: Dict[object, str]
    ) -> Dict[object, str]:
        """
        If a component intersects exactly one non-sentinel old label => inherit that label.
        If it intersects multiple labels => join them (sorted) with '_'.
        Else => assign "new_{cid}".
        """
        comp_to_nodes: Dict[int, List[object]] = defaultdict(list)
        for node, cid in clusters.items():
            comp_to_nodes[cid].append(node)

        comp_label: Dict[int, str] = {}
        for cid, nodes in comp_to_nodes.items():
            old_labels = {
                old_map[n]
                for n in nodes
                if n in old_map and old_map[n] not in self.unclustered_sentinels
            }
            if len(old_labels) == 1:
                comp_label[cid] = next(iter(old_labels))
            elif len(old_labels) > 1:
                comp_label[cid] = "_".join(sorted(old_labels))
            else:
                comp_label[cid] = f"new_{cid}"

        return {node: comp_label[cid] for node, cid in clusters.items()}

    def _sentinel_mask_expr_safe(self, df: pl.DataFrame, col: str) -> pl.Expr:
        """
        Safe sentinel detection that handles dtype mismatches.
        """
        # Get the actual column dtype
        col_dtype = df.schema[col]
        
        # Convert sentinels to match the column type
        if col_dtype == pl.Utf8:
            # For string columns, convert all sentinels to strings (except None)
            sentinel_values = [str(s) for s in self.unclustered_sentinels if s is not None]
            # Handle None separately
            return pl.col(col).is_null() | pl.col(col).is_in(sentinel_values)
        elif col_dtype in [pl.Int64, pl.Int32, pl.UInt64, pl.UInt32]:
            # For integer columns, keep only numeric sentinels
            sentinel_values = [s for s in self.unclustered_sentinels if isinstance(s, (int, float)) and s is not None]
            if sentinel_values:
                return pl.col(col).is_null() | pl.col(col).is_in(sentinel_values)
            else:
                return pl.col(col).is_null()
        else:
            # For other types, fall back to string comparison
            sentinel_values = [str(s) for s in self.unclustered_sentinels if s is not None]
            return pl.col(col).is_null() | pl.col(col).cast(pl.Utf8, strict=False).is_in(sentinel_values)





# -----------------------------
# Two‑DataFrame Matcher (Polars)
# -----------------------------


class TwoDFMatcherPolars:
    """
    Match rows between two Polars DataFrames under nested AND/OR conditions.
    Returns a Polars DataFrame with df1_index, df2_index, and per‑column sim scores.
    """

    def __init__(
        self, id_col_left: Optional[str] = None, id_col_right: Optional[str] = None
    ):
        self.id_col_left = id_col_left
        self.id_col_right = id_col_right
        self.sim_calc = SimilarityCalculator()

    def match_two_dataframes_blocking_conditions(
        self,
        df1: pl.DataFrame,
        df2: pl.DataFrame,
        conditions: Dict,
        top_n: int = 1,
        global_threshold: float = 0.0,
    ) -> pl.DataFrame:
        df1, id_left = _ensure_id_col(df1, self.id_col_left)
        df2, id_right = _ensure_id_col(df2, self.id_col_right)

        match_map: Dict[Tuple[object, object], Dict[str, float]] = (
            self._compute_matches_for_condition(
                conditions, df1, df2, id_left, id_right, top_n
            )
        )

        if not match_map:
            return pl.DataFrame({"df1_index": [], "df2_index": []})

        # Convert to rows
        rows: List[Dict[str, object]] = []
        for (a, b), col_scores in match_map.items():
            row = {"df1_index": a, "df2_index": b}
            row.update(
                {
                    f"{col}_sim": s
                    for col, s in col_scores.items()
                    if s >= global_threshold
                }
            )
            rows.append(row)
        return pl.DataFrame(rows)

    # ---- internals ----

    def _compute_matches_for_condition(
        self,
        condition: Dict,
        df1: pl.DataFrame,
        df2: pl.DataFrame,
        id_left: str,
        id_right: str,
        top_n: int,
    ) -> Dict[Tuple[object, object], Dict[str, float]]:
        if "and" in condition:
            dicts = [
                self._compute_matches_for_condition(
                    c, df1, df2, id_left, id_right, top_n
                )
                for c in condition["and"]
            ]
            if not dicts:
                return {}
            return self._intersect_match_dicts(dicts)
        elif "or" in condition:
            dicts = [
                self._compute_matches_for_condition(
                    c, df1, df2, id_left, id_right, top_n
                )
                for c in condition["or"]
            ]
            if not dicts:
                return {}
            return self._union_match_dicts(dicts)
        else:
            return self._compute_matches_for_leaf(
                condition, df1, df2, id_left, id_right, top_n
            )

    def _compute_matches_for_leaf(
        self,
        leaf: Dict[str, Dict],
        df1: pl.DataFrame,
        df2: pl.DataFrame,
        id_left: str,
        id_right: str,
        top_n: int,
    ) -> Dict[Tuple[object, object], Dict[str, float]]:
        dicts: List[Dict[Tuple[object, object], Dict[str, float]]] = []
        for col_name, params in leaf.items():
            dicts.append(
                self._compute_matches_for_single_column(
                    df1, df2, id_left, id_right, col_name, params, top_n
                )
            )
        if not dicts:
            return {}
        return self._intersect_match_dicts(dicts)

    def _compute_matches_for_single_column(
        self,
        df1: pl.DataFrame,
        df2: pl.DataFrame,
        id_left: str,
        id_right: str,
        col_name: str,
        params: Dict,
        top_n: int,
    ) -> Dict[Tuple[object, object], Dict[str, float]]:
        threshold = params.get("threshold", 0.8)
        method = params.get("similarity_method", "tfidf")
        blocking_criteria = params.get("blocking_criteria")

        # Discover matched blocks in both frames
        for sub1, sub2 in self._generate_block_pairs(
            df1, df2, col_name, blocking_criteria, params
        ):
            # Filter invalid rows for the column
            v1 = sub1.filter(
                pl.col(col_name).is_not_null()
                & (
                    pl.col(col_name)
                    .cast(pl.Utf8)
                    .str.to_lowercase()
                    .is_in(["none", "nan"])
                    .not_()
                )
            )
            v2 = sub2.filter(
                pl.col(col_name).is_not_null()
                & (
                    pl.col(col_name)
                    .cast(pl.Utf8)
                    .str.to_lowercase()
                    .is_in(["none", "nan"])
                    .not_()
                )
            )
            if v1.height == 0 or v2.height == 0:
                continue

            ids_left = v1[id_left].to_list()
            ids_right = v2[id_right].to_list()
            texts_left = v1[col_name].cast(pl.Utf8).to_list()
            texts_right = v2[col_name].cast(pl.Utf8).to_list()

            if method == "exact":
                # Expand exact joins
                lmap: Dict[str, List[object]] = defaultdict(list)
                for i, s in zip(ids_left, map(str, texts_left)):
                    lmap[s].append(i)
                rmap: Dict[str, List[object]] = defaultdict(list)
                for j, s in zip(ids_right, map(str, texts_right)):
                    rmap[s].append(j)
                for val in set(lmap.keys()) & set(rmap.keys()):
                    for i in lmap[val]:
                        for j in rmap[val]:
                            yield_map = {(i, j): {col_name: 1.0}}
                            return yield_map  # fast path for exact only block
                continue

            # TF‑IDF for cross sim
            vectorizer = TfidfVectorizer(
                analyzer="char_wb",
                preprocessor=None,
                lowercase=True,
                ngram_range=(2, 3),
                norm="l2",
                smooth_idf=True,
                use_idf=True,
            )
            all_fit = list(map(str, texts_left + texts_right))
            vectorizer.fit(all_fit)
            A = vectorizer.transform(list(map(str, texts_left)))
            B = vectorizer.transform(list(map(str, texts_right)))
            # top‑N
            S = sp_matmul_topn(A, B.T, top_n=top_n, threshold=threshold, n_threads=-1)
            coo_ = S.tocoo()
            out: Dict[Tuple[object, object], Dict[str, float]] = {}
            for r, c, v in zip(coo_.row, coo_.col, coo_.data):
                out[(ids_left[r], ids_right[c])] = {col_name: float(v)}
            if out:
                return out  # yield block results

        return {}

    def _generate_block_pairs(
        self,
        df1: pl.DataFrame,
        df2: pl.DataFrame,
        col_name: str,
        blocking_criteria: Optional[List[str]],
        params: Dict,
    ) -> Iterator[Tuple[pl.DataFrame, pl.DataFrame]]:
        if not blocking_criteria:
            yield (df1, df2)
            return

        def group(
            df: pl.DataFrame, criterion: str
        ) -> List[Tuple[object, pl.DataFrame]]:
            if criterion == "first_letter":
                t = df.with_columns(
                    pl.col(col_name).cast(pl.Utf8).str.slice(0, 1).alias("__fl")
                )
                return [
                    (k[0], g.drop("__fl"))
                    for k, g in t.group_by("__fl", maintain_order=True)
                ]
            elif criterion == "blocking_column":
                bcols = params.get("blocking_column")
                if isinstance(bcols, str):
                    bcols = [bcols]
                t = df
                for bc in bcols:
                    t = t.with_columns(
                        pl.when(pl.col(bc).is_null())
                        .then(pl.lit("NA"))
                        .otherwise(pl.col(bc).cast(pl.Utf8))
                        .alias(bc)
                    )
                return [
                    (tuple(k), g) for k, g in t.group_by(bcols, maintain_order=True)
                ]
            else:
                raise ValueError(f"Unsupported blocking criterion: {criterion}")

        # apply criteria sequentially
        g1: List[Tuple[object, pl.DataFrame]] = [(None, df1)]
        g2: List[Tuple[object, pl.DataFrame]] = [(None, df2)]
        for crit in blocking_criteria:
            g1 = [x for key, sub in g1 for x in group(sub, crit)]
            g2 = [x for key, sub in g2 for x in group(sub, crit)]

        # index by key
        d1: Dict[object, pl.DataFrame] = {k: g for k, g in g1}
        d2: Dict[object, pl.DataFrame] = {k: g for k, g in g2}
        for key in set(d1.keys()) & set(d2.keys()):
            yield (d1[key], d2[key])

    def _intersect_match_dicts(
        self, dicts: List[Dict[Tuple[object, object], Dict[str, float]]]
    ) -> Dict[Tuple[object, object], Dict[str, float]]:
        common = set(dicts[0].keys())
        for d in dicts[1:]:
            common &= set(d.keys())
        out: Dict[Tuple[object, object], Dict[str, float]] = {}
        for k in common:
            merged: Dict[str, float] = {}
            for d in dicts:
                merged.update(d[k])
            out[k] = merged
        return out

    def _union_match_dicts(
        self, dicts: List[Dict[Tuple[object, object], Dict[str, float]]]
    ) -> Dict[Tuple[object, object], Dict[str, float]]:
        out: Dict[Tuple[object, object], Dict[str, float]] = {}
        for d in dicts:
            for k, v in d.items():
                if k not in out:
                    out[k] = dict(v)
                else:
                    out[k].update(v)
        return out


if __name__ == "__main__":
# ===============================
# WORKING DEMO: Based on your actual code structure
# ===============================

    def working_incremental_demo():
        """
        A working demo that handles all the schema and dtype issues properly
        """
        
        # Simple conditions to avoid complexity
        conditions = {
            "BorrowerName": {
                "threshold": 0.8,
                "similarity_method": "tfidf"
            }
        }
        
        print("=== WORKING INCREMENTAL DEMO ===\n")
        
        # ===============================
        # BATCH 1: Initial clustering
        # ===============================
        
        batch1 = pl.DataFrame({
            "entity_id": [1001, 1002, 1003],
            "BorrowerName": ["JOHN SMITH", "JANE DOE", "BOB WILSON"],
            "BorrowerAddress": ["123 MAIN ST", "456 OAK AVE", "789 PINE RD"],
            "BorrowerCity": ["NYC", "CHI", "LA"]
        })
        
        print("BATCH 1 - Initial data:")
        print(batch1.select(["entity_id", "BorrowerName"]))
        print()
        
        generator1 = SimilarityMatrixGeneratorPolars(
            df=batch1,
            conditions=conditions,
            id_col="entity_id",
            always_string_labels=True,  # This ensures string labels from the start
            carry_forward_singletons=True
        )
        
        # Initial clustering
        combined1, batch1_results = generator1.incremental_cluster(preclustered_df=None)
        
        print("After BATCH 1:")
        print(combined1.select(["entity_id", "BorrowerName", "cluster_label"]))
        print(f"Clusters: {combined1['cluster_label'].n_unique()}")
        print()
        
        # ===============================
        # BATCH 2: Add similar records
        # ===============================
        
        batch2 = pl.DataFrame({
            "entity_id": [2001, 2002],
            "BorrowerName": ["JON SMITH", "J. DOE"],  # Similar to existing
            "BorrowerAddress": ["123 MAIN STREET", "456 OAK AVENUE"],
            "BorrowerCity": ["NYC", "CHI"]
        })
        
        print("BATCH 2 - Adding similar records:")
        print(batch2.select(["entity_id", "BorrowerName"]))
        print()
        
        generator2 = SimilarityMatrixGeneratorPolars(
            df=batch2,
            conditions=conditions,
            id_col="entity_id",
            always_string_labels=True,
            carry_forward_singletons=True
        )
        
        # Incremental clustering
        combined2, batch2_results = generator2.incremental_cluster(preclustered_df=combined1)
        
        print("After BATCH 2:")
        print(combined2.select(["entity_id", "BorrowerName", "cluster_label"]).sort("entity_id"))
        print(f"Clusters: {combined2['cluster_label'].n_unique()}")
        print()
        
        print("BATCH 2 results only:")
        print(batch2_results.select(["entity_id", "BorrowerName", "cluster_label"]))
        print()
        
        # ===============================
        # BATCH 3: Add diverse records
        # ===============================
        
        batch3 = pl.DataFrame({
            "entity_id": [3001, 3002, 3003],
            "BorrowerName": ["ALICE BROWN", "CHARLIE DAVIS", "JOHN SMITH"],  # Mix new + exact match
            "BorrowerAddress": ["321 ELM ST", "999 NEW AVE", "123 MAIN ST"],
            "BorrowerCity": ["BOSTON", "SF", "NYC"]
        })
        
        print("BATCH 3 - Adding diverse records:")
        print(batch3.select(["entity_id", "BorrowerName"]))
        print()
        
        generator3 = SimilarityMatrixGeneratorPolars(
            df=batch3,
            conditions=conditions,
            id_col="entity_id",
            always_string_labels=True,
            carry_forward_singletons=True
        )
        
        # Incremental clustering
        combined3, batch3_results = generator3.incremental_cluster(preclustered_df=combined2)
        
        print("FINAL RESULTS:")
        print(combined3.select(["entity_id", "BorrowerName", "cluster_label"]).sort("entity_id"))
        print(f"Total entities: {combined3.height}")
        print(f"Unique clusters: {combined3['cluster_label'].n_unique()}")
        print()
        
        # Show cluster membership
        print("=== CLUSTER MEMBERSHIP ===")
        cluster_analysis = (
            combined3
            .group_by("cluster_label")
            .agg([
                pl.count().alias("size"),
                pl.col("BorrowerName").alias("members")
            ])
            .sort("size", descending=True)
        )
        
        for row in cluster_analysis.iter_rows(named=True):
            print(f"Cluster {row['cluster_label']}: {row['size']} members")
            print(f"  Names: {', '.join(row['members'])}")
            print()
        
        return combined3

    def test_with_existing_labels():
        """
        Test incremental clustering with pre-existing cluster labels
        """
        print("=== TEST WITH EXISTING LABELS ===\n")
        
        # Simulate existing clustered data
        existing_snapshot = pl.DataFrame({
            "entity_id": [100, 101, 102, 103],
            "BorrowerName": ["ACME CORP", "ACME CORPORATION", "BETA LLC", "GAMMA INC"],
            "BorrowerAddress": ["100 BUSINESS ST", "100 BUSINESS STREET", "200 TRADE AVE", "300 COMMERCE RD"],
            "BorrowerCity": ["DALLAS", "DALLAS", "HOUSTON", "AUSTIN"],
            "cluster_label": ["company_1", "company_1", "company_2", "-1"]  # Note: -1 is unclustered
        })
        
        print("Existing snapshot:")
        print(existing_snapshot.select(["entity_id", "BorrowerName", "cluster_label"]))
        print()
        
        # New batch that might match existing
        new_batch = pl.DataFrame({
            "entity_id": [200, 201, 202],
            "BorrowerName": ["ACME CORP", "BETA LIMITED", "DELTA SYSTEMS"],
            "BorrowerAddress": ["100 BUSINESS ST", "200 TRADE AVENUE", "400 TECH BLVD"],
            "BorrowerCity": ["DALLAS", "HOUSTON", "AUSTIN"]
        })
        
        print("New batch:")
        print(new_batch.select(["entity_id", "BorrowerName"]))
        print()
        
        conditions = {
            "BorrowerName": {
                "threshold": 0.7,
                "similarity_method": "tfidf"
            }
        }
        
        generator = SimilarityMatrixGeneratorPolars(
            df=new_batch,
            conditions=conditions,
            id_col="entity_id",
            always_string_labels=True,
            unclustered_sentinels={"-1", "unclustered", None}
        )
        
        # Incremental clustering
        final_snapshot, new_results = generator.incremental_cluster(preclustered_df=existing_snapshot)
        
        print("Final results:")
        print(final_snapshot.select(["entity_id", "BorrowerName", "cluster_label"]).sort("entity_id"))
        print()
        
        print("New batch results only:")
        print(new_results.select(["entity_id", "BorrowerName", "cluster_label"]))
        print()
        
        # Analysis
        print("=== CLUSTER ANALYSIS ===")
        non_unclustered = final_snapshot.filter(pl.col("cluster_label") != "-1")
        cluster_summary = (
            non_unclustered
            .group_by("cluster_label")
            .agg([
                pl.count().alias("size"),
                pl.col("BorrowerName").alias("members")
            ])
            .sort("cluster_label")
        )
        
        for row in cluster_summary.iter_rows(named=True):
            print(f"{row['cluster_label']}: {', '.join(row['members'])}")
        
        return final_snapshot

    def minimal_test():
        """
        Absolute minimal test to verify the fix works
        """
        print("=== MINIMAL TEST ===\n")
        
        # Simplest possible case
        batch1 = pl.DataFrame({
            "name": ["JOHN", "JANE"],
            "id": [1, 2]
        })
        
        conditions = {"name": {"similarity_method": "exact"}}
        
        gen1 = SimilarityMatrixGeneratorPolars(
            df=batch1,
            conditions=conditions,
            id_col="id",
            always_string_labels=True
        )
        
        result1, _ = gen1.incremental_cluster(None)
        print("Batch 1 result:")
        print(result1)
        print()
        
        # Second batch
        batch2 = pl.DataFrame({
            "name": ["JOHN"],  # Exact match
            "id": [3]
        })
        
        gen2 = SimilarityMatrixGeneratorPolars(
            df=batch2,
            conditions=conditions,
            id_col="id",
            always_string_labels=True
        )
        
        result2, batch2_only = gen2.incremental_cluster(result1)
        print("Final result:")
        print(result2)
        print()
        print("Batch 2 only:")
        print(batch2_only)
        
        return result2
    print("Running minimal test...\n")
    minimal_test()
    print("\n" + "="*60 + "\n")
    
    print("Running working demo...\n")
    working_incremental_demo()
    print("\n" + "="*60 + "\n")
    
    print("Running test with existing labels...\n")
    test_with_existing_labels()
    
    print("\n✓ All tests completed successfully!")