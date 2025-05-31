import pandas as pd
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import normalize
from sparse_dot_topn import awesome_cossim_topn
import hashlib
import pdb
import pickle
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from itertools import combinations
import warnings

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

    # def save(self, path):
    #     with open(path, "wb") as f:
    #         pickle.dump(self.parent, f)

    # def load(self, path):
    #     with open(path, "rb") as f:
    #         self.parent = pickle.load(f)


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

        if index is None:
            raise ValueError("Please include an index field")
        else:
            self.index = index

        original_columns = self.df.columns.tolist()
        columns_used_to_cluster = []
        for col, col_config in self.fields_config.items():
            if "block_on" in col_config:
                blocked_column = col_config["block_on"]
                if blocked_column not in columns_used_to_cluster:
                    columns_used_to_cluster.append(blocked_column)
            if col not in columns_used_to_cluster:
                columns_used_to_cluster.append(col)

        self.df_original = self.df[[self.index] + original_columns].copy()
        self.df = self.df[[self.index] + columns_used_to_cluster]

    def _vectorize_column(self, series, vectorizer_type="count"):
        series = series.fillna("").astype(str)

        if vectorizer_type == "count":
            vectorizer = CountVectorizer(analyzer="char", ngram_range=(1, 4))
        else:
            vectorizer = TfidfVectorizer(
                analyzer="char", ngram_range=(1, 4), norm="l2", sublinear_tf=True
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

    def _get_similarity_pairs(self, X, threshold, series=None):
        if threshold == 1.0 and series is not None:
            return self._get_exact_match_pairs_fast(series)
        sim_matrix = awesome_cossim_topn(
            X, X, self.topn, threshold, use_threads=True, return_best_ntop=False
        )
        rows, cols = sim_matrix.nonzero()
        values = sim_matrix.data
        # pdb.set_trace()
        return [(i, j, v) for i, j, v in zip(rows, cols, values) if i < j]

    def _combine_edge_sets(self, edge_sets):
        if self.combine_method == "AND":
            return set.intersection(*edge_sets)
        else:
            return set.union(*edge_sets)

    def apply_blocking_columns(self, df, fields_config):
        df = df.copy()
        for field, config in fields_config.items():
            block_col = config.get("block_on")
            if block_col:
                block_name = f"block__{field}"
                series = df[block_col].astype(str).str.strip().str.lower()

                trim_length = config.get("block_trim")
                if trim_length:
                    series = series.str[:trim_length]

                df[block_name] = series
        return df

    def _deduplicate_for_comparison(self, df, col):
        """
        Create a deduplicated version for similarity comparison while maintaining
        mapping back to original indices.
        """
        # Group by the column value and collect all indices for each unique value
        grouped = df.groupby(col)[self.index].apply(list).reset_index()
        grouped.columns = [col, "original_indices"]

        # Create a mapping from deduplicated index to original indices
        dedup_to_original = {}
        for idx, row in grouped.iterrows():
            dedup_to_original[idx] = row["original_indices"]

        # Create deduplicated dataframe for comparison
        dedup_df = grouped[[col]].copy()
        dedup_df["dedup_index"] = dedup_df.index

        return dedup_df, dedup_to_original

    def _expand_dedup_pairs_to_original(self, pairs, dedup_to_original):
        """
        Expand similarity pairs from deduplicated indices back to all original indices.
        """
        expanded_pairs = []
        for i, j, score in pairs:
            # Get all original indices for both deduplicated indices
            original_i_list = dedup_to_original[i]
            original_j_list = dedup_to_original[j]

            # Create pairs between all combinations of original indices
            for orig_i in original_i_list:
                for orig_j in original_j_list:
                    if orig_i != orig_j:  # Avoid self-pairs
                        expanded_pairs.append((orig_i, orig_j, score))

        return expanded_pairs

    def _get_cluster_id(self, component):
        if self.use_hash_cluster_id:
            return hashlib.md5(
                "".join(sorted(map(str, component))).encode()
            ).hexdigest()
        else:
            return sorted(component)[0]

    def run(self):
        from collections import defaultdict

        all_edges_by_field = {}
        all_comparisons = []

        print("Applying blocking columns...")
        self.df = self.apply_blocking_columns(self.df, self.fields_config)
        print("Blocking columns applied.")

        for col, params in self.fields_config.items():
            blocking_col = params.get("block_on")
            blocking_col = f"block__{blocking_col}"
            threshold = params.get("threshold", 0.85)
            vectorizer_type = params.get("vectorizer", "count")
            topn = params.get("topn", self.topn)

            field_df = self.df.copy()
            block_values = field_df[blocking_col].dropna().unique()

            field_edges = set()

            for block_val in tqdm(block_values, desc=f"Processing {col}"):
                g = field_df[field_df[blocking_col] == block_val]
                if len(g) < 2:
                    continue
                if not self.group_nulls:
                    g = g[g[col].notnull()]
                    if len(g) < 2:
                        continue

                g = g.copy()

                # Step 1: Normalize and deduplicate values
                series = g[col].fillna("").astype(str).str.strip().str.lower()
                unique_vals = series.drop_duplicates().reset_index(drop=True)

                if len(unique_vals) < 2:
                    continue

                try:
                    X = self._vectorize_column(
                        unique_vals, vectorizer_type=vectorizer_type
                    )
                except ValueError:
                    continue

                value_pairs = self._get_similarity_pairs(X, threshold, unique_vals)

                # Step 2: Cluster values using UnionFind
                idx_to_val = {i: v for i, v in unique_vals.items()}

                uf = UnionFind()
                for i, j, _ in value_pairs:
                    uf.union(i, j)

                components = uf.components()
                # Step 3: Map values to all row indices
                val_to_row_ids = defaultdict(list)
                for i, val in series.items():
                    val_to_row_ids[val].append(g.loc[i, self.index])

                # 👇 Add exact match edges (if enabled) always enabled
                # if params.get("exact_match", False):
                for val, ids in tqdm(val_to_row_ids.items()):
                    if len(ids) > 1:
                        for i, j in combinations(ids, 2):
                            edge = (i, j, 1.0)
                            field_edges.add(edge)
                            all_comparisons.append((i, j, 1.0, col))

                # Step 4: Expand value-pairs to row-pairs (edges)
                for i, j, score in tqdm(value_pairs):
                    val_i = idx_to_val[i]
                    val_j = idx_to_val[j]

                    ids_i = val_to_row_ids[val_i]
                    ids_j = val_to_row_ids[val_j]

                    if val_i == val_j:
                        for row_i, row_j in combinations(ids_i, 2):
                            edge = (row_i, row_j, round(score, 2))
                            field_edges.add(edge)
                            all_comparisons.append((*edge, col))
                    else:
                        for row_i in ids_i:
                            for row_j in ids_j:
                                edge = (row_i, row_j, round(score, 2))
                                field_edges.add(edge)
                                all_comparisons.append((*edge, col))

            if field_edges:
                all_edges_by_field[col] = field_edges

        # Combine edges using AND/OR logic
        if not all_edges_by_field:
            print("No edges found.")
            self.comparison_df = pd.DataFrame(
                columns=["id1", "id2", "similarity_score", "field"]
            )
            return self.df, self.comparison_df
        print("Combining edges across fields...")
        combined_edges = self._combine_edge_sets(list(all_edges_by_field.values()))
        print(f"Total combined edges: {len(combined_edges)}")

        # Run UnionFind on combined edges
        for i, j, _ in tqdm(combined_edges, desc="Union-Find Clustering"):
            self.uf.union(i, j)

        components = self.uf.components()
        cluster_labels = {
            node: self._get_cluster_id(component)
            for component in components
            for node in component
        }

        self.df["cluster_id"] = self.df[self.index].map(cluster_labels)

        # Comparison dataframe
        self.comparison_df = pd.DataFrame(
            all_comparisons, columns=["id1", "id2", "similarity_score", "field"]
        )

        return self.df, self.comparison_df


if __name__ == "__main__":
    df = pd.read_csv("test_data/100k.csv", nrows=100_000, encoding="latin-1")
    df = df[["BorrowerName", "BorrowerAddress", "BorrowerCity"]]
    df["BorrowerCity"] = df["BorrowerCity"].str.strip().str.lower()
    df = df[df["BorrowerCity"].str.contains("^sc", na=False)]
    # df = df.drop_duplicates(subset=["BorrowerCity"]).reset_index(drop=True)
    fields_config = {
        "BorrowerCity": {
            "block_on": "BorrowerCity",
            "block_trim": 1,
            "threshold": 0.75,
            "topn": 5,
            "exact_match": True,
        },
    }
    df.reset_index(drop=True, inplace=True)
    df.reset_index(inplace=True)

    df, df_comp = EntityClustering(
        df=df, fields_config=fields_config, index="index"
    ).run()
    # Merge id1

    pdb.set_trace()
