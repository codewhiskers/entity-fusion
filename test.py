import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import normalize

# from sparse_dot_topn import awesome_cossim_topn
from sparse_dot_topn import sp_matmul_topn, zip_sp_matmul_topn
import hashlib
import pdb
import re

# import pickle
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from itertools import combinations
import warnings
from scipy.sparse import vstack, hstack

# from datasketch import MinHash, MinHashLSH
# from simhash import Simhash
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
                lambda row: self._compute_row_hash(row, self.df.columns), axis=1
            )  # changed from columns_used_to_cluster
            self.index = "row_hash"
        else:
            self.index = index

        self.df_original = self.df[original_columns].copy()
        # self.df = self.df[[self.index] + columns_used_to_cluster]
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

    def _get_exact_match_pairs_fast(self, df, col, chunk_size=1000, max_pairs=10000):
        if not self.index:
            raise ValueError("Index column is not set.")

        pairs = set()
        grouped = df.groupby(col)
        group_items = list(grouped)

        for i in range(0, len(group_items), chunk_size):
            chunk = group_items[i : i + chunk_size]
            for _, group in chunk:
                n = len(group)
                if n > 1:
                    if (n * (n - 1)) // 2 > max_pairs:
                        continue
                    group_ids = group[self.index].tolist()
                    for i, j in combinations(range(len(group_ids)), 2):
                        id_i = group_ids[i]
                        id_j = group_ids[j]
                        pairs.add((id_i, id_j, 1.0))

        return list(pairs)

    def _get_similarity_pairs(self, X, col, threshold, row_ids=None):
        if threshold == 1.0:
            return self._get_exact_match_pairs_fast(X, col)
        if X.shape[0] == 0:
            return []

        sim_matrix = sp_matmul_topn(
            X,
            X,
            top_n=self.topn,
            threshold=threshold,
            sort=False,
            n_threads=8,
        )
        rows, cols = sim_matrix.nonzero()
        values = sim_matrix.data

        if row_ids:
            return [
                (row_ids[i], row_ids[j], v)
                for i, j, v in zip(rows, cols, values)
                if i < j
            ]
        else:
            return [(i, j, v) for i, j, v in zip(rows, cols, values) if i < j]

    def _combine_edge_sets(self, edge_sets):
        if self.combine_method == "AND":
            # Only return intersection if all fields produced at least one edge set
            non_empty_sets = [s for s in edge_sets if s]
            if len(non_empty_sets) == len(edge_sets):
                return set.intersection(*non_empty_sets)
            else:
                return set()  # Some field didn't match anything, so AND = empty
        else:
            return set.union(*edge_sets)

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
        self, X, col, threshold, topn, n_blocks=(4, 4), row_ids=None
    ):
        if threshold == 1.0:
            return self._get_exact_match_pairs_fast(X, col)

        def chunk_indices(n, n_chunks):
            chunk_size = int(np.ceil(n / n_chunks))
            return [
                slice(i * chunk_size, min((i + 1) * chunk_size, n))
                for i in range(n_chunks)
            ]

        row_chunks = chunk_indices(X.shape[0], n_blocks[0])
        col_chunks = chunk_indices(X.shape[0], n_blocks[1])

        block_results = []
        for row_slice in row_chunks:
            row_block = X[row_slice]
            row_results = []
            for col_slice in col_chunks:
                col_block = X[col_slice]

                sim_block = sp_matmul_topn(
                    row_block,
                    col_block.T,
                    top_n=topn,
                    threshold=threshold,
                    sort=True,
                    n_threads=8,
                )
                row_results.append(sim_block)

            merged_row = zip_sp_matmul_topn(top_n=topn, C_mats=row_results)
            block_results.append(merged_row)

        full_matrix = vstack(block_results).tocsr()
        rows, cols = full_matrix.nonzero()
        scores = full_matrix.data

        if row_ids:
            return [
                (row_ids[i], row_ids[j], v)
                for i, j, v in zip(rows, cols, scores)
                if i < j
            ]
        else:
            return [(i, j, v) for i, j, v in zip(rows, cols, scores) if i < j]

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
        # if self.df[[x for x in self.df.columns if x != "index"]].duplicated().any():
        #     pdb.set_trace()
        #     raise ValueError(
        #         f"Duplicate values found in your dataframe. "
        #         "Please ensure the rows in the DataFrame are unique."
        #     )

        all_edges_by_field = {}
        all_comparisons = []

        print("Applying blocking columns...")
        self.df = self.apply_blocking_columns(self.df, self.fields_config)
        print("Blocking columns applied.")

        field_df = self.df.copy()

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

            field_edges = set()

            # BLOCK_LSH_THRESHOLD = 20_000  # Threshold for using LSH filtering
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
                    g[col] = g[col].fillna("").astype(str).str.strip().str.lower()
                    series = g[col].copy()
                    row_ids = g[self.index].tolist()
                    # subset_idx = list(range(len(series)))

                    try:
                        if threshold != 1:
                            subset_X = self._vectorize_column(
                                series, vectorizer_type=vectorizer_type
                            )
                        else:
                            subset_X = g[[self.index, col]]
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
                    if subset_X.shape[0] <= 5_000:
                        value_pairs = self._get_similarity_pairs(
                            subset_X, col, threshold, row_ids=row_ids
                        )
                    else:
                        n_blocks = self.estimate_n_blocks(subset_X)
                        value_pairs = self._get_sparse_similarity_matrix_partitioned(
                            subset_X,
                            col,
                            threshold,
                            self.topn,
                            n_blocks=n_blocks,
                            row_ids=row_ids,
                        )

                    for id_i, id_j, score in value_pairs:
                        edge = (id_i, id_j, round(score, 2))
                        field_edges.add(edge)
                        all_comparisons.append((*edge, col))

                    if field_edges:
                        all_edges_by_field[col] = field_edges
        #         pdb.set_trace()
        # Combine edges across fields
        if not all_edges_by_field:
            print("No edges found.")
            self.comparison_df = pd.DataFrame(
                columns=["id1", "id2", "similarity_score", "field"]
            )
            return self.df_original, self.comparison_df

        print("Combining edges across fields...")
        combined_edges = self._combine_edge_sets(list(all_edges_by_field.values()))
        print(f"Total combined edges: {len(combined_edges)}")

        # Run Union-Find
        for i, j, _ in tqdm(combined_edges, desc="Union-Find Clustering"):
            self.uf.union(i, j)

        components = self.uf.components()
        print(f"Total clusters found: {len(components)}")
        cluster_labels = {}
        for component in components:
            cluster_id = self._get_cluster_id(component)
            for node in component:
                cluster_labels[node] = cluster_id

        self.df["cluster_id"] = None
        self.df.set_index(self.index, inplace=True)
        self.df["cluster_id"].update(pd.Series(cluster_labels))
        self.df.reset_index(inplace=True)

        self.comparison_df = pd.DataFrame(
            all_comparisons, columns=["id1", "id2", "similarity_score", "field"]
        )
        # Identify rows with no cluster assignment
        unclustered_mask = self.df["cluster_id"].isna()

        # Option 1: use row_hash (already computed elsewhere)
        if self.index in self.df.columns:
            self.df.loc[unclustered_mask, "cluster_id"] = self.df.loc[
                unclustered_mask, self.index
            ]
        self.df = self.df_original.merge(
            self.df[[self.index, "cluster_id"]],
            on=self.index,
            how="left",
        )
        return self.df, self.comparison_df


# def _vectorize_column(series, vectorizer):
#     vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5))
#     return vectorizer.transform(series.fillna(""))

# vectorizer = TfidfVectorizer().fit(all_text)

# tfidf1 = _vectorize_column(df1[col], vectorizer)
# tfidf2 = _vectorize_column(df2[col], vectorizer)

# sim_matrix = awesome_cossim_topn(tfidf2, tfidf1, 10, 0.95, use_threads=True, return_best_ntop=False)
# matches_coo = sim_matrix.tocoo()
# matched_clusters = {}
# for i, j, score in zip(matches_coo.row, matches_coo.col, matches_coo.data):
# #     print(i, j, score)
#     matched_clusters[i] = df_cluster.iloc[j]['cluster_id']

# assigned_clusters = []
# for i in range(df2.shape[0]):
#     if i in matched_clusters:
#         assigned_clusters.append(matched_clusters[i])
#     else:
#         assigned_clusters.append(-1)
# df2['cluster_id'] = assigned_clusters


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
    df = df.drop_duplicates()
    df = df[0:100_000]  # .reset_index(drop=True)
    # df = df.reset_index()
    df["BorrowerAddress"] = df["BorrowerAddress"].astype(str).str.strip().str.lower()
    df = df[df["BorrowerState"] == "CA"]
    # pdb.set_trace()
    fields_config = {
        "BorrowerAddress": {
            # "block_on": "BorrowerState",
            "threshold": 1,
            "vectorizer": "tfidf",
        },
        "BorrowerName": {
            # "block_on": "BorrowerState",
            "threshold": 0.9,
            "vectorizer": "tfidf",
        },
    }

    # from your_package import EntityClustering  # use actual import

    start = time.time()
    # pdb.set_trace()
    df_final, df_comp = EntityClustering(
        df=df,
        fields_config=fields_config,
    ).run()
    your_time = time.time() - start
    pdb.set_trace()
    # from string_grouper import match_strings, group_similar_strings
    # import time

    # Extract just the address column
    # addresses = df["BorrowerAddress"].dropna().drop_duplicates().astype(str)

    # # Time the grouping
    # start = time.time()
    # matches = match_strings(df["BorrowerAddress"], min_similarity=0.9)
    # stringgrouper_time = time.time() - start
    # df[["group-id", "name_deduped"]] = group_similar_strings(df["BorrowerName"])
    # # Collapse matches into groups – pass both the addresses and the matches
    # # groups = group_similar_strings(addresses, matches)

    # print(f"EntityClustering time: {your_time:.2f}s")
    # print(f"String Grouper time: {stringgrouper_time:.2f}s")

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
