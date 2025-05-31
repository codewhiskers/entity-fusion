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
        # if "BorrowerName" in columns_used_to_cluster:
        #     pdb.set_trace()
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
        # if len(df) > 4000:
        #     pdb.set_trace()
        df = df.copy()
        for field, config in fields_config.items():
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
            # if col == "BorrowerName":
            #     pdb.set_trace()
            blocking_col = params.get("block_on")
            blocking_col = f"block__{blocking_col}"
            threshold = params.get("threshold", 0.85)
            vectorizer_type = params.get("vectorizer", "tfidf")
            # topn = params.get("topn", self.topn)

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
                series = g[col].fillna("").astype(str).str.strip().str.lower()
                row_ids = g[self.index].tolist()

                try:
                    X = self._vectorize_column(series, vectorizer_type=vectorizer_type)
                except ValueError:
                    continue

                value_pairs = self._get_similarity_pairs(X, threshold)

                # Add similarity edges
                for i, j, score in value_pairs:
                    id_i = row_ids[i]
                    id_j = row_ids[j]
                    if id_i != id_j:
                        edge = (id_i, id_j, round(score, 2))
                        field_edges.add(edge)
                        all_comparisons.append((*edge, col))

                # Add exact match edges
                val_to_ids = defaultdict(list)
                for val, idx in zip(series, g[self.index]):
                    val_to_ids[val].append(idx)

                for val, ids in val_to_ids.items():
                    if len(ids) > 1:
                        for i, j in combinations(ids, 2):
                            edge = (i, j, 1.0)
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
        # pdb.set_trace()
        # Option 1: use row_hash (already computed elsewhere)
        if "row_hash" in self.df.columns:
            self.df.loc[unclustered_mask, "cluster_id"] = self.df.loc[
                unclustered_mask, "row_hash"
            ]

        self.df = self.df_original.merge(
            self.df[["row_hash", "cluster_id"]],
            on="row_hash",
            how="left",
        )
        self.df.drop(columns=["row_hash"], inplace=True)
        return self.df, self.comparison_df


if __name__ == "__main__":
    df = pd.read_csv("test_data/100k.csv", nrows=100_000, encoding="latin-1")
    df = df[["BorrowerName", "BorrowerAddress", "BorrowerCity"]]
    df["BorrowerCity"] = df["BorrowerCity"].str.strip().str.lower()
    # df = df[df["BorrowerCity"].str.contains("^sc", na=False)]
    # df = df.drop_duplicates(subset=["BorrowerCity"]).reset_index(drop=True)
    fields_config = {
        "BorrowerCity": {
            "block_on": "BorrowerCity",
            "block_trim": 1,
            "threshold": 0.75,
            "vectorizer": "count",
        },
    }
    df.reset_index(drop=True, inplace=True)
    df.reset_index(inplace=True)

    df, df_comp = EntityClustering(
        df=df,
        fields_config=fields_config,
        # index="index",
    ).run()
    df.rename(columns={"cluster_id": "BorrowerCity_cluster_id"}, inplace=True)
    # pdb.set_trace()
    fields_config = {
        "BorrowerName": {
            "block_on": "BorrowerCity_cluster_id",
            "threshold": 0.75,
            "vectorizer": "tfidf",
        },
        "BorrowerAddress": {
            "block_on": "BorrowerCity_cluster_id",
            "threshold": 0.9,
            "vectorizer": "tfidf",
        },
    }
    df_final, df_comp = EntityClustering(
        df=df,
        fields_config=fields_config,
        # index="index",
    ).run()
    pdb.set_trace()
