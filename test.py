# Refactored Flexible Entity Resolution Pipeline with Multiple Field Support and Blocking

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import awesome_cossim_topn
import networkx as nx
from tqdm import tqdm
import re
import hashlib
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import pdb
import pickle
import uuid


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

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.parent, f)

    def load(self, path):
        with open(path, "rb") as f:
            self.parent = pickle.load(f)

    def assign_cluster_or_fallback_id(df, index_col, fields_to_check, unionfind):
        """
        Assigns a cluster ID from unionfind, or falls back to index or a generated hash.

        - If all fields_to_check are null/empty → assign unique hash
        - If index_col is in unionfind.parent → assign cluster via find()
        - Else → use the index_col value itself as fallback cluster ID

        Returns:
        - pandas Series of cluster labels (int or str)
        """

        def is_meaningless(row):
            return all(
                pd.isna(row[col])
                or str(row[col]).strip().lower() in {"", "nan", "none", "__null__"}
                for col in fields_to_check
            )

        def normalize_text(text):
            text = str(text).lower()
            text = re.sub(r"\b(inc|llc|co|corp|group|foundation)\b", "", text)
            return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", text)).strip()

        def is_meaningless(row, fields):
            return all(
                pd.isna(row[f])
                or str(row[f]).strip().lower() in {"", "nan", "none", "__null__"}
                for f in fields
            )

        def row_hash(row, fields):
            if is_meaningless(row, fields):
                return hashlib.sha256(str(uuid.uuid4()).encode()).hexdigest()[:16]
            else:
                key = "|".join(normalize_text(row[f]) for f in fields)
                return hashlib.sha256(key.encode()).hexdigest()[:16]


class EntityClustering:
    def __init__(
        self,
        df,
        fields_config,
        combine_method="OR",
        index=None,
        deduped_clustered_path=None,
    ):
        self.df = df.reset_index(drop=True)
        self.df = self.df.replace("None", None)
        self.df = self.df.replace("NaN", None)
        self.fields_config = fields_config
        self.combine_method = combine_method
        self.threshold = 0.85
        self.topn = 10
        self.group_nulls = True
        self.edge_output_path = None
        self.uf = UnionFind()
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
        new_df_columns = []
        for col, col_config in self.fields_config.items():
            if "block_on" in col_config:
                blocked_column = col_config["block_on"]
                norm_col = f"{blocked_column}_normalized"
                self.df[norm_col] = self.df[blocked_column].apply(self._normalize_text)
                new_df_columns += [blocked_column, norm_col]
            norm_col = f"{col}_normalized"
            self.df[norm_col] = self.df[col].apply(self._normalize_text)
            new_df_columns += [col, norm_col]

        columns_to_hash = list(self.fields_config.keys())
        if index is None:
            self.df["row_hash"] = self.df.apply(
                lambda row: self._row_hash(row, columns_to_hash), axis=1
            )
        else:
            self.df["row_hash"] = self.df[index].copy()

        self.df_original = self.df[["row_hash"] + original_columns].copy()
        self.df = self.df.drop_duplicates(["row_hash"])
        self.df = self.df[["row_hash"] + new_df_columns]

    def _normalize_text(self, text):
        if pd.isna(text):
            return ""
        text = text.lower()
        for word in self.custom_stopwords:
            text = re.sub(rf"\b{word}\b", "", text)
        text = re.sub(r"[^a-z0-9]+", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _row_hash(self, row, columns_to_hash):
        key_fields = [str(row.get(f, "")) for f in columns_to_hash]
        if all(
            f.strip() == "" or f.strip().lower() in {"nan", "__null__", "none"}
            for f in key_fields
        ):
            return hashlib.sha256(str(uuid.uuid4()).encode()).hexdigest()[:16]
        normalized = "|".join(key_fields).lower()
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]

    def _vectorize_column(self, series):
        if len(series) < 2:
            raise ValueError("Not enough records to compute TF-IDF.")
        vectorizer = TfidfVectorizer(
            min_df=1,
            analyzer="char_wb",
            ngram_range=(3, 5),
            stop_words=None,
            norm="l2",
            sublinear_tf=True,
        )
        return vectorizer.fit_transform(series.fillna(""))

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
        if not edge_sets:
            return set()
        if self.combine_method == "AND":
            return set.intersection(*edge_sets)
        return set.union(*edge_sets)

    def _process_block(self, block_df):
        if len(block_df) < 2:
            return
        edge_sets = []
        for col, params in self.fields_config.items():
            normalized_col = block_df[f"{col}_normalized"]
            try:
                X = self._vectorize_column(normalized_col)
            except ValueError:
                continue

            field_threshold = params.get("threshold", self.threshold)
            pairs = self._get_similarity_pairs(X, field_threshold, normalized_col)

            index_map = dict(enumerate(block_df.index))
            local_to_hash = lambda i: self.df.loc[index_map[i], "row_hash"]
            edges = [(local_to_hash(i), local_to_hash(j), s) for i, j, s in pairs]
            edge_sets.append(edges)

        combined = self._combine_edge_sets(
            [{(i, j, s) for i, j, s in es} for es in edge_sets]
        )
        for i, j, _ in combined:
            self.uf.union(i, j)

    def run(self):
        block_columns = set()
        for col, params in self.fields_config.items():
            block_col = params.get("block_on")
            if block_col:
                block_col = f"{block_col}_normalized"
                block_columns.add(block_col)
                self.df[block_col] = self.df[block_col].fillna("__NULL__").astype(str)

        if block_columns:
            self.df["__block_key__"] = self.df.apply(
                lambda row: "_".join(
                    [str(row.get(col, "__NULL__")) for col in block_columns]
                ),
                axis=1,
            )
            grouped = [g for _, g in self.df.groupby("__block_key__")]
        else:
            grouped = [self.df]

        for g in tqdm(grouped, desc="Blocking"):
            self._process_block(g)

    def save_unionfind(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.uf.parent, f)

    def load_unionfind(self, path):
        with open(path, "rb") as f:
            self.uf.parent = pickle.load(f)


class IncrementalEntityClustering(EntityClustering):
    def __init__(
        self,
        df_new,
        fields_config,
        combine_method="OR",
        index=None,
        deduped_clustered_path=None,
    ):
        super().__init__(
            pd.DataFrame(),
            fields_config,
            combine_method,
            index=index,
            deduped_clustered_path=deduped_clustered_path,
        )
        self.df_new = df_new.reset_index(drop=True)
        self.index = index
        self.df_new = self._prepare_df(self.df_new, self.index)

    def _prepare_df(self, df, index):
        df = df.replace("None", None)
        df = df.replace("NaN", None)

        for col, col_config in self.fields_config.items():
            if "block_on" in col_config:
                blocked_column = col_config["block_on"]
                df[f"{blocked_column}_normalized"] = df[blocked_column].apply(
                    self._normalize_text
                )
            df[f"{col}_normalized"] = df[col].apply(self._normalize_text)

        columns_to_hash = list(self.fields_config.keys())
        if index is None:
            df["row_hash"] = df.apply(
                lambda row: self._row_hash(row, columns_to_hash), axis=1
            )
        else:
            df["row_hash"] = df[index].copy()

        return df

    def run_incremental(self):
        if self.deduped_clustered_path and Path(self.deduped_clustered_path).exists():
            df_existing = pd.read_parquet(self.deduped_clustered_path)
            self.df = self._prepare_df(df_existing, index="row_hash")
        else:
            self.df = pd.DataFrame(columns=self.df_new.columns)

        combined_df = pd.concat([self.df, self.df_new], ignore_index=True)

        block_columns = set()
        for col, params in self.fields_config.items():
            block_col = params.get("block_on")
            if block_col:
                block_col = f"{block_col}_normalized"
                block_columns.add(block_col)
                combined_df[block_col] = (
                    combined_df[block_col].fillna("__NULL__").astype(str)
                )

        if block_columns:
            combined_df["__block_key__"] = combined_df.apply(
                lambda row: "_".join(
                    [str(row.get(col, "__NULL__")) for col in block_columns]
                ),
                axis=1,
            )
            grouped = [g for _, g in combined_df.groupby("__block_key__")]
        else:
            grouped = [combined_df]

        for group in grouped:
            if len(group) < 2:
                continue
            edge_sets = []
            for col, params in self.fields_config.items():
                normalized_col = group[f"{col}_normalized"]
                try:
                    X = self._vectorize_column(normalized_col)
                except ValueError:
                    continue
                threshold = params.get("threshold", self.threshold)
                pairs = self._get_similarity_pairs(X, threshold)

                index_map = dict(enumerate(group.index))
                local_to_hash = lambda i: combined_df.loc[index_map[i], "row_hash"]
                edges = [(local_to_hash(i), local_to_hash(j), s) for i, j, s in pairs]
                edge_sets.append(edges)

            combined = self._combine_edge_sets(
                [{(i, j, s) for i, j, s in es} for es in edge_sets]
            )
            for i, j, _ in combined:
                self.uf.union(i, j)

        cluster_labels = {
            node: root for root, nodes in self.uf.components() for node in nodes
        }
        combined_df["cluster_id"] = combined_df["row_hash"].map(cluster_labels)

        if self.deduped_clustered_path:
            new_deduped = combined_df.drop_duplicates("row_hash")
            new_deduped = new_deduped.drop_duplicates("cluster_id")
            if Path(self.deduped_clustered_path).exists():
                old_deduped = pd.read_parquet(self.deduped_clustered_path)
                combined_deduped = (
                    pd.concat([old_deduped, new_deduped])
                    .drop_duplicates("row_hash", keep="last")
                    .drop_duplicates("cluster_id", keep="last")
                    .reset_index(drop=True)
                )
            else:
                combined_deduped = new_deduped
            combined_deduped.to_parquet(self.deduped_clustered_path, index=False)

        return combined_df


iec = IncrementalEntityClustering(
    df_existing=old_df,
    df_new=new_df,
    fields_config=fields_config,
    uf_path="clusters.pkl",
    cluster_map_path="clustered.parquet",
)
iec.run_incremental()


# def normalize_text(text):
#     text = str(text).lower()
#     text = re.sub(r"\b(inc|llc|co|corp|group|foundation)\b", "", text)
#     return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", text)).strip()


# def row_hash(row):
#     fields = ["BorrowerName", "BorrowerAddress"]
#     key = "|".join(normalize_text(row[f]) for f in fields)
#     return hashlib.sha256(key.encode()).hexdigest()[:16]


# df = pd.read_csv(
#     "test_data/100k.csv",
#     usecols=["BorrowerName", "BorrowerAddress", "BorrowerCity"],
#     nrows=200000,
# )


# with open("clusters.uf.pkl", "rb") as f:
#     parent_map = pickle.load(f)

# # Create connected components
# from collections import defaultdict


# def find(x):
#     while parent_map[x] != x:
#         parent_map[x] = parent_map[parent_map[x]]
#         x = parent_map[x]
#     return x


# clusters = defaultdict(set)
# for node in parent_map:
#     clusters[find(node)].add(node)

# hash_to_cluster = {h: i for i, group in enumerate(clusters.values()) for h in group}
# pdb.set_trace()

# df["row_hash"] = df.apply(row_hash, axis=1)
# df["cluster"] = df["row_hash"].map(lambda h: hash_to_cluster.get(h, -1))
