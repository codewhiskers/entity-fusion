import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import normalize
from sparse_dot_topn import awesome_cossim_topn
from sparse_dot_topn import sp_matmul_topn
import hashlib
import pdb
import re
import pickle
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from itertools import combinations
import warnings
from scipy.sparse import vstack, hstack
from datasketch import MinHash, MinHashLSH
from simhash import Simhash
from tqdm import tqdm
from itertools import combinations
from sklearn.feature_extraction.text import CountVectorizer
from concurrent.futures import ThreadPoolExecutor

warnings.filterwarnings("ignore", category=DeprecationWarning)


class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        self.parent[self.find(x)] = self.find(y)

    def components(self):
        groups = defaultdict(set)
        for node in self.parent:
            root = self.find(node)
            groups[root].add(node)
        return list(groups.values())


class EntityClustering:
    def __init__(
        self,
        df,
        fields_config,
        combine_method="OR",
        index=None,
        deduped_clustered_path=None,
        group_nulls=True,
        use_hash_cluster_id=False,
        verbose=True,
    ):
        self.df = df.reset_index(drop=True)
        self.df = self.df.replace(["None", "NaN", "Unknown", ""], None)
        self.fields_config = fields_config
        self.combine_method = combine_method
        self.uf = UnionFind()
        self.topn = 5  # can be overridden per field
        self.group_nulls = group_nulls
        self.verbose = verbose
        self.use_hash_cluster_id = use_hash_cluster_id
        self.deduped_clustered_path = deduped_clustered_path

        self.custom_stopwords = set(
            [
                "inc",
                "ltd",
                "llc",
                "co",
                "corp",
                "corporation",
                "association",
                "hospital",
                "church",
                "limited",
                "partners",
                "services",
                "supply",
                "house",
                "group",
                "foundation",
                "club",
                "the",
                "international",
                "investment",
                "investments",
                "na",
            ]
        )

        original_columns = self.df.columns.tolist()
        columns_used_to_cluster = []
        for col, col_config in self.fields_config.items():
            if "block_on" in col_config:
                blocked_column = col_config["block_on"]
                if blocked_column not in columns_used_to_cluster:
                    columns_used_to_cluster.append(blocked_column)
            if col not in columns_used_to_cluster:
                columns_used_to_cluster.append(col)

        # Add row_hash if index not provided
        if index is None:
            print(
                "No index provided. Generating row_hash based on clustering columns..."
            )
            self.df["row_hash"] = self.df.apply(
                lambda row: self._compute_row_hash(row, columns_used_to_cluster), axis=1
            )
            self.index = "row_hash"
        else:
            self.index = index

        self.df_original = self.df[[self.index] + original_columns].copy()
        self.df = self.df[[self.index] + columns_used_to_cluster]
        self.df.drop_duplicates(subset=self.index, inplace=True)

    def _compute_row_hash(self, row, cols):
        val = "|".join(
            str(row[col]).strip().lower() if pd.notnull(row[col]) else ""
            for col in cols
        )
        return hashlib.md5(val.encode()).hexdigest()

    def _vectorize_column(self, series, vectorizer_type="count"):
        def string_grouper_ngram_analyzer(string):
            # Normalize, lowercase, remove punctuation/whitespace
            string = normalize("NFKD", string).encode("ASCII", "ignore").decode()
            string = re.sub(r"[,-./]|\s", "", string.lower())

            # Generate trigrams
            return ["".join(t) for t in zip(*[string[i:] for i in range(3)])]

        series = series.fillna("").astype(str)

        if vectorizer_type == "count":
            vectorizer = CountVectorizer(analyzer="char", ngram_range=(1, 4))
        else:
            # vectorizer = TfidfVectorizer(
            #     analyzer=string_grouper_ngram_analyzer,
            #     dtype=np.float32,
            # )

            vectorizer = TfidfVectorizer(
                analyzer="char",
                ngram_range=(3, 3),
                norm="l2",
                sublinear_tf=False,
                dtype=np.float32,
            )

        X = vectorizer.fit_transform(series)

        # Normalize if using CountVectorizer
        if vectorizer_type == "count":
            X = normalize(X, norm="l2", axis=1)

        return X

    def _get_exact_match_pairs_fast(self, series):
        duplicates = series.reset_index().groupby(series).filter(lambda x: len(x) > 1)
        grouped = duplicates.groupby(series)
        pairs = []
        for _, group in grouped:
            indices = group["index"].tolist()
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    pairs.append((indices[i], indices[j], 1.0))
        return pairs

    def get_minhash(self, text, num_perm=64):
        m = MinHash(num_perm=num_perm)
        for ngram in self._char_ngrams(text):
            m.update(ngram.encode("utf8"))
        return m

    def _char_ngrams(self, text, n=3):
        text = text.strip().lower()
        if len(text) < n:
            return {text}

        # Use list comprehension instead of set comprehension for better performance
        # when you have many duplicates
        ngrams = [text[i : i + n] for i in range(len(text) - n + 1)]
        return set(ngrams)  # Convert to set only once

    def lsh_candidate_filtering(self, series, threshold=0.8, num_perm=64):
        minhashes = {}
        lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)

        # Single pass: create, store, and insert MinHash objects
        for idx, val in tqdm(
            series.items(), desc="Creating MinHashes", total=len(series)
        ):
            m = self.get_minhash(val, num_perm)
            minhashes[idx] = m
            lsh.insert(idx, m)

        # More efficient candidate pair generation
        candidate_pairs = set()
        for idx in minhashes:
            results = lsh.query(minhashes[idx])
            for j in results:
                if j != idx:
                    # Create ordered pair to avoid duplicates
                    pair = tuple(sorted([idx, j]))
                    candidate_pairs.add(pair)

        return list(candidate_pairs)

    def _get_similarity_pairs(self, X, threshold, series=None):
        if threshold == 1.0 and series is not None:
            return self._get_exact_match_pairs_fast(series)
        sim_matrix = awesome_cossim_topn(
            X, X, self.topn, threshold, use_threads=True, return_best_ntop=False
        )
        rows, cols = sim_matrix.nonzero()
        values = sim_matrix.data
        return [(i, j, v) for i, j, v in zip(rows, cols, values) if i < j]

    def _combine_edge_sets(self, edge_sets):
        if self.combine_method == "AND":
            return set.intersection(*edge_sets)
        else:
            return set.union(*edge_sets)

    # def _get_sparse_similarity_matrix_partitioned(
    #     self, X, threshold, topn=5, block_size=5_000
    # ):
    #     def chunk_indices(n, chunk_size):
    #         return [range(i, min(i + chunk_size, n)) for i in range(0, n, chunk_size)]

    #     n = X.shape[0]
    #     row_chunks = chunk_indices(n, block_size)
    #     col_chunks = chunk_indices(n, block_size)

    #     row_blocks = []
    #     for row_idx in tqdm(
    #         row_chunks, desc="processing blocks", total=len(row_chunks)
    #     ):
    #         row_A = X[row_idx]
    #         col_blocks = []
    #         for col_idx in col_chunks:
    #             col_B = X[col_idx]
    #             sim_block = awesome_cossim_topn(
    #                 row_A,
    #                 col_B.T,
    #                 topn,
    #                 threshold,
    #                 use_threads=True,
    #                 n_jobs=8,
    #                 return_best_ntop=False,
    #             )
    #             # sim_block = sp_matmul_topn(
    #             #     row_A,
    #             #     col_B.T,
    #             #     top_n=topn,
    #             #     threshold=threshold,
    #             #     sort=False,  # Optional: sort matches by descending similarity
    #             #     n_threads=4,  # Optional: set threads explicitly
    #             # )
    #             col_blocks.append(sim_block)
    #         row_blocks.append(hstack(col_blocks))
    #     sim_matrix = vstack(row_blocks)
    #     rows, cols = sim_matrix.nonzero()
    #     values = sim_matrix.data

    #     value_pairs = [
    #         (i, j, round(v, 4)) for i, j, v in zip(rows, cols, values) if i < j
    #     ]
    #     return value_pairs

    def estimate_n_blocks(self, X, max_rows_per_block=4000, max_total_blocks=64):
        """
        Estimate (n_row_blocks, n_col_blocks) based on matrix size.

        Args:
            X (csr_matrix): TF-IDF matrix
            max_rows_per_block (int): Target rows per block (controls granularity)
            max_total_blocks (int): Optional upper limit to prevent over-chunking

        Returns:
            Tuple[int, int]: Number of row and column blocks
        """
        n = X.shape[0]

        n_row_blocks = max(1, round(n / 1_000_000))  # coarse splitting for big corpora
        n_col_blocks = max(1, round(n / max_rows_per_block))

        # Limit to avoid too many tiny blocks
        if n_row_blocks * n_col_blocks > max_total_blocks:
            scale = (n_row_blocks * n_col_blocks) / max_total_blocks
            n_row_blocks = max(1, int(n_row_blocks / scale))
            n_col_blocks = max(1, int(n_col_blocks / scale))

        return (n_row_blocks, n_col_blocks)

    def _get_sparse_similarity_matrix_partitioned(
        self, X, threshold, topn, n_blocks=(4, 4)
    ):
        """
        Computes cosine similarities in block-wise fashion using sp_matmul_topn or awesome_cossim_topn and merges results,
        without using zip_sp_matmul_topn.
        """

        def chunk_indices(n, n_chunks):
            chunk_size = int(np.ceil(n / n_chunks))
            return [
                slice(i * chunk_size, min((i + 1) * chunk_size, n))
                for i in range(n_chunks)
            ]

        row_chunks = chunk_indices(X.shape[0], n_blocks[0])
        col_chunks = chunk_indices(X.shape[0], n_blocks[1])

        block_results = []
        for row_slice in tqdm(
            row_chunks, desc="Processing row blocks", total=len(row_chunks)
        ):
            row_block = X[row_slice]
            row_partial_results = []

            for col_slice in col_chunks:
                col_block = X[col_slice]

                sim_block = awesome_cossim_topn(
                    row_block,
                    col_block.T,
                    topn,
                    threshold,
                    # sort=True,
                    use_threads=True,
                    n_jobs=8,  # Optional: set threads explicitly
                    return_best_ntop=False,
                    # n_threads=max(
                    #     1, self.num_threads if hasattr(self, "num_threads") else 1
                    # ),
                )

                row_partial_results.append(sim_block)

            # Horizontally stack all column chunks (same row block)
            merged_row = hstack(row_partial_results).tocsr()

            # Optional: enforce top-N per row manually
            # (Not strictly necessary unless you want to prune across column chunks)

            block_results.append(merged_row)

        # Vertically stack all row blocks
        full_matrix = vstack(block_results).tocsr()

        # Extract non-zero entries
        rows, cols = full_matrix.nonzero()
        scores = full_matrix.data

        return list(zip(rows, cols, scores))

    def get_simhash(self, text, n=3):
        text = text.strip().lower()
        tokens = (
            [text[i : i + n] for i in range(len(text) - n + 1)]
            if len(text) >= n
            else [text]
        )
        return Simhash(tokens)

    def simhash_candidate_filtering(self, series, max_distance=3, bucket_bits=8):
        """
        Simulates LSH-like querying using simple buckets on SimHash prefixes.
        """
        simhashes = {}
        buckets = defaultdict(set)

        for idx, val in tqdm(
            series.items(), desc="Creating SimHashes", total=len(series)
        ):
            h = self.get_simhash(val)
            simhashes[idx] = h
            prefix = h.value >> (64 - bucket_bits)  # e.g., first 8 bits
            buckets[prefix].add(idx)

        candidate_pairs = set()

        for bucket in tqdm(buckets.values()):
            if len(bucket) < 2:
                continue

            for idx_i, idx_j in combinations(bucket, 2):
                if simhashes[idx_i].distance(simhashes[idx_j]) <= max_distance:
                    candidate_pairs.add((idx_i, idx_j))

        return list(candidate_pairs)

    def apply_blocking_columns(self, df, fields_config):
        df = df.copy()
        for _, config in fields_config.items():
            block_col = config.get("block_on")
            if block_col:
                block_name = f"block__{block_col}"
                series = df[block_col].astype(str).str.strip().str.lower()

                trim_length = config.get("block_trim")
                if trim_length:
                    series = series.str[:trim_length]

                df[block_name] = series
        return df

    def _get_cluster_id(self, component):
        if self.use_hash_cluster_id:
            return hashlib.md5(
                "".join(sorted(map(str, component))).encode()
            ).hexdigest()
        else:
            return sorted(component)[0]

    def run(self):
        if self.df[[x for x in self.df.columns if x != "index"]].duplicated().any():
            raise ValueError(
                f"Duplicate values found in your dataframe. "
                "Please ensure the rows in the DataFrame are unique."
            )

        all_edges_by_field = {}
        all_comparisons = []

        print("Applying blocking columns...")
        self.df = self.apply_blocking_columns(self.df, self.fields_config)
        print("Blocking columns applied.")

        for col, params in self.fields_config.items():
            blocking_col_raw = params.get("block_on")

            if blocking_col_raw:
                blocking_col = f"block__{blocking_col_raw}"
                block_values = field_df[blocking_col].dropna().unique()
            else:
                blocking_col = None
                block_values = [None]  # single pseudo-block
            threshold = params.get("threshold", 0.85)
            vectorizer_type = params.get("vectorizer", "tfidf")
            field_df = self.df.copy()

            # block_values = field_df[blocking_col].dropna().unique()

            field_edges = set()
        BLOCK_LSH_THRESHOLD = 20_000  # Threshold for using LSH filtering

        with tqdm(block_values) as pbar:
            for block_val in pbar:
                pbar.set_description(f"Processing {block_val}")
                if blocking_col is None:
                    g = field_df
                else:
                    g = field_df[field_df[blocking_col] == block_val]
                if len(g) < 2:
                    continue
                if not self.group_nulls:
                    g = g[g[col].notnull()]
                    if len(g) < 2:
                        continue

                g = g.copy()
                series = g[col].fillna("").astype(str).str.strip().str.lower()
                row_ids = g[self.index].tolist()

                use_lsh = len(g) > BLOCK_LSH_THRESHOLD
                use_lsh = False
                if use_lsh:
                    print("Using LSH filtering for large block...")

                    series_for_lsh = pd.Series({i: s for i, s in enumerate(series)})

                    candidate_pairs = self.lsh_candidate_filtering(
                        series_for_lsh, threshold=0.8
                    )
                    # candidate_pairs = self.simhash_candidate_filtering(
                    #     series_for_lsh, max_distance=3
                    # )
                    if not candidate_pairs:
                        continue

                    subset_idx = sorted(
                        set(i for pair in candidate_pairs for i in pair)
                    )
                    series = series.iloc[subset_idx]
                    row_ids = [
                        row_ids[i] for i in subset_idx
                    ]  # filter row_ids accordingly
                else:
                    subset_idx = list(range(len(series)))

                try:
                    subset_X = self._vectorize_column(
                        series, vectorizer_type=vectorizer_type
                    )
                except ValueError:
                    print(
                        f"Skipping column {col} due to ValueError in vectorization. "
                        "This may be due to too many unique values or empty strings."
                    )
                    continue

                subset_id_map = {
                    local_idx: row_ids[global_idx]
                    for local_idx, global_idx in enumerate(range(len(row_ids)))
                }

                if subset_X.shape[0] <= 20_000:
                    value_pairs = self._get_similarity_pairs(subset_X, threshold)
                else:
                    n_blocks = self.estimate_n_blocks(subset_X)
                    value_pairs = self._get_sparse_similarity_matrix_partitioned(
                        subset_X, threshold, self.topn, n_blocks=n_blocks
                    )

                for i, j, score in value_pairs:
                    if i < j:
                        id_i = subset_id_map[i]
                        id_j = subset_id_map[j]
                        edge = (id_i, id_j, round(score, 2))
                        field_edges.add(edge)
                        all_comparisons.append((*edge, col))

                if field_edges:
                    all_edges_by_field[col] = field_edges

        # Combine edges across fields
        if not all_edges_by_field:
            print("No edges found.")
            self.comparison_df = pd.DataFrame(
                columns=["id1", "id2", "similarity_score", "field"]
            )
            return self.df, self.comparison_df

        print("Combining edges across fields...")
        combined_edges = self._combine_edge_sets(list(all_edges_by_field.values()))
        print(f"Total combined edges: {len(combined_edges)}")

        # Run Union-Find
        for i, j, _ in tqdm(combined_edges, desc="Union-Find Clustering"):
            self.uf.union(i, j)

        components = self.uf.components()
        cluster_labels = {
            node: self._get_cluster_id(component)
            for component in components
            for node in component
        }

        self.df["cluster_id"] = self.df[self.index].map(cluster_labels)

        self.comparison_df = pd.DataFrame(
            all_comparisons, columns=["id1", "id2", "similarity_score", "field"]
        )
        # Identify rows with no cluster assignment
        unclustered_mask = self.df["cluster_id"].isna()

        # Option 1: use row_hash (already computed elsewhere)
        if "row_hash" in self.df.columns:
            self.df.loc[unclustered_mask, "cluster_id"] = self.df.loc[
                unclustered_mask, "row_hash"
            ]
        # pdb.set_trace()
        self.df = self.df_original.merge(
            self.df[["row_hash", "cluster_id"]],
            on="row_hash",
            how="left",
        )
        self.df.drop(columns=["row_hash"], inplace=True)
        return self.df, self.comparison_df


if __name__ == "__main__":
    # df = pd.read_csv("test_data/100k.csv", nrows=1_000_000, encoding="latin-1")
    # df = df[["BorrowerName", "BorrowerAddress", "BorrowerCity", "BorrowerState"]]
    # df = df[df["BorrowerState"].notnull()]
    # df["BorrowerCity"] = df["BorrowerCity"].str.strip().str.lower()
    # # df = df[0:100_000].reset_index(drop=True)
    # # pdb.set_trace()
    # df.to_parquet("test_data/100k.parquet", index=False)
    # pdb.set_trace()
    import time

    df = pd.read_parquet("test_data/100k.parquet")
    df = df.drop_duplicates(["BorrowerAddress"])
    df = df[0:100_000]  # .reset_index(drop=True)
    df["BorrowerName"] = df["BorrowerName"].astype(str).str.strip().str.lower()
    # df = df[df["BorrowerState"] == "AZ"]

    fields_config = {
        "BorrowerName": {
            # "block_on": "BorrowerState",
            "threshold": 0.9,
            "vectorizer": "tfidf",
        },
    }

    # from your_package import EntityClustering  # use actual import

    start = time.time()
    df_final, df_comp = EntityClustering(df=df, fields_config=fields_config).run()
    your_time = time.time() - start

    from string_grouper import match_strings, group_similar_strings
    import time

    # Extract just the address column
    # addresses = df["BorrowerAddress"].dropna().drop_duplicates().astype(str)

    # Time the grouping
    start = time.time()
    matches = match_strings(df["BorrowerName"], min_similarity=0.9)
    stringgrouper_time = time.time() - start
    df[["group-id", "name_deduped"]] = group_similar_strings(df["BorrowerName"])
    # Collapse matches into groups – pass both the addresses and the matches
    # groups = group_similar_strings(addresses, matches)

    print(f"EntityClustering time: {your_time:.2f}s")
    print(f"String Grouper time: {stringgrouper_time:.2f}s")

    pdb.set_trace()
    # fields_config = {
    #     "BorrowerCity": {
    #         "block_on": "BorrowerState",
    #         "block_trim": 1,
    #         "threshold": 0.75,
    #         "vectorizer": "count",
    #     },
    # }
    # # pdb.set_trace()
    # df.reset_index(drop=True, inplace=True)
    # df.reset_index(inplace=True)

    # df, df_comp = EntityClustering(
    #     df=df,
    #     fields_config=fields_config,
    #     # index="index",
    # ).run()
    # df.rename(columns={"cluster_id": "BorrowerCity_cluster_id"}, inplace=True)

    fields_config = {
        "BorrowerName": {
            "block_on": "BorrowerState",
            "threshold": 0.8,
            "vectorizer": "count",
        },
    }

    df_final, df_comp = EntityClustering(
        df=df,
        fields_config=fields_config,
        # index="index",
    ).run()
    pdb.set_trace()

    # rows, cols = sim_matrix.nonzero()
    # scores = sim_matrix.data
    # for i, j, score in zip(rows, cols, scores):
    #     if i < j:
    #         id_i = subset_id_map[i]
    #         id_j = subset_id_map[j]
    #         edge = (id_i, id_j, round(score, 2))
    #         field_edges.add(edge)
    #         all_comparisons.append((*edge, col))
    # else:
    #     try:
    #         X = self._vectorize_column(
    #             series, vectorizer_type=vectorizer_type
    #         )
    #     except ValueError:
    #         continue
    #     if X.shape[0] <= 10_000:
    #         value_pairs = self._get_similarity_pairs(X, threshold)
    #     else:
    #         value_pairs = self._get_sparse_similarity_matrix_partitioned(
    #             X, threshold, self.topn
    #         )

    # Add similarity edges
    # for i, j, score in value_pairs:
    #     id_i = row_ids[i]
    #     id_j = row_ids[j]
    #     if id_i != id_j:
    #         edge = (id_i, id_j, round(score, 2))
    #         field_edges.add(edge)
    #         all_comparisons.append((*edge, col))

    #     # Add exact match edges
    #     val_to_ids = defaultdict(list)
    #     for val, idx in zip(series, g[self.index]):
    #         val_to_ids[val].append(idx)

    #     for val, ids in val_to_ids.items():
    #         if len(ids) > 1:
    #             for i, j in combinations(ids, 2):
    #                 edge = (i, j, 1.0)
    #                 field_edges.add(edge)
    #                 all_comparisons.append((*edge, col))

    # def lsh_candidate_filtering_batched(
    #     self, series, threshold=0.8, num_perm=128, batch_size=1000
    # ):
    #     """Process in batches to manage memory usage for large datasets"""
    #     if len(series) <= batch_size:
    #         return self.lsh_candidate_filtering(series, threshold, num_perm)

    #     all_candidates = set()
    #     indices = list(series.index)

    #     for i in range(0, len(indices), batch_size):
    #         batch_indices = indices[i : i + batch_size]
    #         batch_series = series.loc[batch_indices]
    #         batch_candidates = self.lsh_candidate_filtering(
    #             batch_series, threshold, num_perm
    #         )
    #         all_candidates.update(batch_candidates)

    #     return list(all_candidates)
