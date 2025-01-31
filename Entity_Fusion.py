import numpy as np
import pandas as pd
from scipy.sparse import lil_matrix, coo_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
import pdb
import re
from collections import defaultdict, deque
from sparse_dot_topn import awesome_cossim_topn, sp_matmul_topn


class SimilarityMatrixGenerator:
    def __init__(self, df, column_thresholds, combine_method="OR"):
        self.df = df
        self.column_thresholds = column_thresholds
        self.df_sim = None
        self.similarity_results = []  # will hold (column, column_results) pairs
        self.graph = defaultdict(set)
        self.clusters = {}
        self.combine_method = combine_method.upper()  # "AND" or "OR"

    def create_similarity_matrices(self):
        """
        Create similarity matrices for specified columns and store them as lists of tuples.
        """
        if not self.column_thresholds:
            raise ValueError(
                "No column thresholds specified for similarity computation."
            )

        similarity_calculator = SimilarityCalculator()
        data_grouper = DataGrouper(self.df)
        similarity_results = []

        for column, params in self.column_thresholds.items():
            if column not in self.df.columns:
                raise ValueError(f"Column '{column}' not found in the dataframe.")

            column_results = []

            # Group and process with a progress bar
            groups = list(
                data_grouper.group_dataframe(params, column)
            )  # Convert generator to list for tqdm
            for group in tqdm(groups, desc=f"Processing groups for column '{column}'"):
                if len(group) <= 1:
                    # Skip single-item groups
                    continue

                group_data = similarity_calculator.initialize_vectorizer(
                    params.get("similarity_method", "tfidf"), group[column]
                )
                result = similarity_calculator.create_similarity_matrix(
                    group_data,
                    group.index,
                    column,
                    params["threshold"],
                    params["similarity_method"],
                )

                if result.shape[0] > 0:
                    column_results.extend(
                        result.tolist()
                    )  # Convert NumPy array to list of tuples

            # Store results for this column as a list of tuples
            if column_results:
                similarity_results.append((column, column_results))

        self.similarity_results = similarity_results

    def _construct_similarity_graph(self):
        """
        Construct a graph from the similarity results stored as lists of tuples,
        combining them via AND or OR logic.
        """
        print("Constructing similarity graph...")

        # 1) Build a list of sets, one set per column
        edges_by_column = []

        for column, data in self.similarity_results:
            # data is a list of [id1, id2, similarity]
            # We'll convert them into a set of (id1, id2) pairs
            column_edges = set()

            for id1, id2, similarity in data:
                # Skip self-loops
                if id1 == id2:
                    continue

                # Ensure (id1, id2) is stored in a canonical order
                if id1 > id2:
                    id1, id2 = id2, id1

                column_edges.add((id1, id2))

            # Only add this set if it's not empty
            if column_edges:
                edges_by_column.append(column_edges)

        # 2) Combine sets with OR or AND
        #    - If no columns produce edges, default to empty set
        if not edges_by_column:
            final_edges = set()
        else:
            if self.combine_method == "AND":
                final_edges = set.intersection(*edges_by_column)
            else:  # default OR
                final_edges = set.union(*edges_by_column)

        # 3) Build the graph from final_edges
        for id1, id2 in final_edges:
            self.graph[id1].add(id2)
            self.graph[id2].add(id1)

        # print("Similarity graph constructed.")

    def _find_clusters_from_graph(self):
        """
        Find clusters using connected components.
        """

        def bfs(graph, start_node, visited):
            cluster = set()
            queue = deque([start_node])
            while queue:
                node = queue.popleft()
                if node not in visited:
                    visited.add(node)
                    cluster.add(node)
                    queue.extend(graph[node] - visited)
            return cluster

        clusters = []
        visited = set()

        for node in self.graph.keys():
            if node not in visited:
                cluster = bfs(self.graph, node, visited)
                clusters.append(cluster)

        # Assign cluster IDs
        cluster_map = {}
        for cluster_id, cluster in enumerate(clusters):
            for node in cluster:
                cluster_map[node] = cluster_id

        self.clusters = cluster_map

    def assign_cluster_labels(self):
        """
        Assign cluster labels to the dataframe based on the graph clusters.
        """
        if not self.graph:
            raise ValueError(
                "Graph not constructed. Run _construct_similarity_graph first."
            )
        if not self.clusters:
            raise ValueError("Clusters not found. Run _find_clusters_from_graph first.")

        # Map cluster labels to the dataframe
        self.df["cluster_label"] = self.df.index.map(self.clusters)

        # Assign unique cluster labels to nodes not in the graph
        unclustered = self.df["cluster_label"].isna()
        self.df.loc[unclustered, "cluster_label"] = range(
            max(self.clusters.values(), default=-1) + 1,
            max(self.clusters.values(), default=-1) + 1 + unclustered.sum(),
        )

    def cluster_data(self):
        """
        Main method to cluster data.
        """
        # Step 1: Create similarity matrices
        self.create_similarity_matrices()
        # pdb.set_trace()
        # Step 2: Construct similarity graph
        self._construct_similarity_graph()

        # Step 3: Find clusters from the graph
        self._find_clusters_from_graph()

        # Step 4: Assign cluster labels
        self.assign_cluster_labels()
        pdb.set_trace()
        return self.df


class SimilarityCalculator:

    def initialize_vectorizer(self, similarity_method, data_column):
        """
        Initialize the vectorizer and transform the data based on the similarity method.

        Args:
            similarity_method (str): The similarity method to use ('tfidf', 'numeric', or 'exact').
            data_column (pd.Series): The column of data to vectorize.

        Returns:
            scipy.sparse matrix or pd.Series: The vectorized data or raw data for exact matching.
        """
        if similarity_method == "numeric":
            # Numeric similarity: tokenizes numbers from the text
            vectorizer = TfidfVectorizer(
                tokenizer=lambda x: re.findall(r"\d+", x),  # Extract numeric tokens
                preprocessor=None,
                lowercase=False,
                # stop_words="english",
            )
            return vectorizer.fit_transform(data_column.values)

        elif similarity_method == "tfidf":
            # TF-IDF similarity: character n-grams
            vectorizer = TfidfVectorizer(
                analyzer="char_wb",
                preprocessor=None,
                lowercase=True,
                ngram_range=(2, 3),  # Bi- and tri-grams
                norm="l2",
                smooth_idf=True,
                use_idf=True,
                stop_words="english",
            )
            return vectorizer.fit_transform(data_column.values)

        elif similarity_method == "exact":
            # Exact similarity: raw data (no vectorization)
            return data_column  # Return the column directly

        else:
            raise ValueError(f"Unsupported similarity method: {similarity_method}")

    def create_similarity_matrix(
        self, group_tfidf, group_ids, column_name, threshold, similarity_method
    ):
        if similarity_method == "exact":
            return self._create_exact_match_matrix(group_tfidf, group_ids, column_name)
        return self._create_cosine_similarity_matrix(
            group_tfidf, group_ids, column_name, threshold
        )

    def _create_exact_match_matrix(self, group_tfidf, group_ids, column_name):
        # Logic for exact matching similarity matrix
        pass

    def _create_cosine_similarity_matrix(
        self, group_tfidf, group_ids, column_name, threshold
    ):
        """
        Create a cosine similarity matrix using `sparse_dot_topn` for large groups
        and the original setup for small groups, without progress bars.
        """
        n_samples = group_tfidf.shape[0]
        chunk_size = 2000
        large_group_threshold = 500  # Use sparse_dot_topn for groups larger than this

        if n_samples > large_group_threshold:
            top_n = 100  # retain top-N similarities

            cos_sim_sparse = sp_matmul_topn(
                group_tfidf, group_tfidf, top_n=top_n, threshold=threshold, n_threads=-1
            )
        # Use the original setup for small groups
        else:
            cos_sim_sparse = lil_matrix((n_samples, n_samples), dtype=np.float32)

            for start_idx in range(0, n_samples, chunk_size):
                end_idx = min(start_idx + chunk_size, n_samples)
                chunk_matrix = self._compute_cosine_similarity_chunk(
                    start_idx, end_idx, group_tfidf, threshold
                )
                cos_sim_sparse[start_idx:end_idx, :] = chunk_matrix

        return self._finalize_similarity_matrix(cos_sim_sparse, group_ids, column_name)

    def _compute_cosine_similarity_chunk(
        self, start_idx, end_idx, group_tfidf, threshold
    ):
        chunk_matrix = cosine_similarity(group_tfidf[start_idx:end_idx], group_tfidf)
        mask = chunk_matrix >= threshold
        return np.where(mask, chunk_matrix, 0)

    def _finalize_similarity_matrix(self, cos_sim_sparse, group_ids, column_name):
        """
        Finalize the similarity matrix by filtering out self-similarities
        and converting to a lightweight format.
        """
        coo = coo_matrix(cos_sim_sparse)
        rows, cols, values = coo.row, coo.col, coo.data

        # Filter out self-similarities
        mask = rows != cols
        filtered_rows = rows[mask]
        filtered_cols = cols[mask]
        filtered_values = values[mask]

        # Construct NumPy array
        group_ids = np.array(group_ids)
        similarities = np.vstack(
            (
                group_ids[filtered_rows],
                group_ids[filtered_cols],
                filtered_values,
            )
        ).T

        return similarities  # Return as NumPy array


class DataGrouper:
    def __init__(self, df):
        self.df = df

    def group_dataframe(self, params, column):
        df = self._preprocess_column(column)
        blocking_criteria = params.get("blocking_criteria", None)

        if not blocking_criteria:
            return [(None, df)]

        grouped_data = [df]
        for criterion in blocking_criteria:
            grouped_data = self._apply_blocking_criterion(
                grouped_data, criterion, params, column
            )
        return grouped_data

    def _preprocess_column(self, column):
        return self.df[
            (self.df[column].notnull())
            & (self.df[column] != "")
            & (self.df[column].str.lower().isin(["unknown", "nan", "none"]) == False)
        ]

    def _apply_blocking_criterion(self, grouped_data, criterion, params, column):
        new_groups = []
        for group in grouped_data:
            if criterion == "first_letter":
                # Group by the first letter of the column
                new_groups.extend(list(group.groupby(group[column].str[0], sort=False)))
            elif criterion == "blocking_column":
                blocking_columns = params.get("blocking_column")
                if isinstance(blocking_columns, list):
                    # Group by multiple columns
                    new_groups.extend(list(group.groupby(blocking_columns, sort=False)))
                elif isinstance(blocking_columns, str):
                    # Group by a single column
                    new_groups.extend(list(group.groupby(blocking_columns, sort=False)))
                else:
                    raise ValueError(
                        f"Invalid blocking_columns type: {type(blocking_columns)}"
                    )
            else:
                raise ValueError(f"Unsupported blocking criterion: {criterion}")
        return [grp for _, grp in new_groups if len(grp) > 1]


class TwoDFMatcher:
    """
    A helper class to match rows between two DataFrames using
    column-level thresholds, blocking criteria, and a TF-IDF similarity method.
    """

    # def __init__(self, tfidf_min_value=0.0):
    #     """
    #     Args:
    #         tfidf_min_value (float): If > 0, will prune TF-IDF entries below this value.
    #     """
    #     self.tfidf_min_value = tfidf_min_value

    def match_two_dataframes_blocking(
        self,
        df1,
        df2,
        column_thresholds,
        top_n=1,
        global_threshold=0.0,
        combine_method="OR",
    ):
        """
        Main method to match two DataFrames using per-column thresholds and blocking criteria.

        Args:
            df1 (pd.DataFrame): First DataFrame
            df2 (pd.DataFrame): Second DataFrame
            column_thresholds (dict): A dict like:
                {
                  "BorrowerName": {
                    "threshold": 0.9,
                    "similarity_method": "tfidf",
                    "blocking_criteria": ["blocking_column"],
                    "blocking_column": ["BorrowerCity"],
                  },
                  "BorrowerAddress": {
                    "threshold": 0.8,
                    "similarity_method": "tfidf",
                    "blocking_criteria": ["first_letter"],
                  },
                  ...
                }
            top_n (int): Number of top matches to retain per row within each block.
            global_threshold (float): If > 0, a final similarity cutoff to apply after combining columns.
            combine_method (str): "OR" or "AND" to combine matches across columns.

        Returns:
            pd.DataFrame: Combined matches with columns [df1_index, df2_index, similarity_score, column]
        """
        all_matches_by_column = []

        # Wrap columns iteration in a progress bar:
        for col_name, params in tqdm(
            column_thresholds.items(), desc="Processing columns"
        ):
            threshold = params["threshold"]
            similarity_method = params.get("similarity_method", "tfidf")
            blocking_criteria = params.get("blocking_criteria", [])

            # Generate block pairs
            block_pairs = list(
                self._generate_block_pairs(
                    df1, df2, col_name, blocking_criteria, params
                )
            )
            column_matches = []

            # Wrap block pairs in an inner progress bar
            for sub_df1, sub_df2 in tqdm(
                block_pairs, desc=f"Blocks for {col_name}", leave=False
            ):
                if sub_df1.empty or sub_df2.empty:
                    continue

                block_result_df = self._match_subdataframes(
                    sub_df1, sub_df2, col_name, similarity_method, threshold, top_n
                )

                if not block_result_df.empty:
                    block_result_df["column"] = col_name
                    column_matches.append(block_result_df)

            if column_matches:
                column_df = pd.concat(column_matches, ignore_index=True)
                all_matches_by_column.append(column_df)

        # Combine results across columns, apply AND/OR logic, etc. (same as before)
        if not all_matches_by_column:
            return pd.DataFrame(
                columns=["df1_index", "df2_index", "similarity_score", "column"]
            )

        combined_df = pd.concat(all_matches_by_column, ignore_index=True)

        if combine_method == "AND":
            group_cols = ["df1_index", "df2_index"]
            pair_counts = (
                combined_df.groupby(group_cols)["column"].nunique().reset_index()
            )
            n_columns = len(column_thresholds)
            valid_pairs = pair_counts[pair_counts["column"] == n_columns]
            combined_df = combined_df.merge(
                valid_pairs[group_cols], on=group_cols, how="inner"
            )

        if global_threshold > 0:
            combined_df = combined_df[
                combined_df["similarity_score"] >= global_threshold
            ]

        return combined_df

    def _generate_block_pairs(self, df1, df2, col_name, blocking_criteria, params):
        """
        Produce (sub_df1, sub_df2) pairs that share the same blocking key(s),
        according to the blocking criteria for this column.

        If no blocking criteria, we yield the entire (df1, df2).
        """
        if not blocking_criteria:
            yield (df1, df2)
            return

        new_groups_1 = [df1]
        new_groups_2 = [df2]

        # If there's a chain of criteria (e.g. ["blocking_column", "first_letter"]),
        # apply them in sequence.
        for criterion in blocking_criteria:
            tmp_groups_1 = []
            for gdf1 in new_groups_1:
                tmp_groups_1.extend(
                    self._apply_blocking_criterion(gdf1, criterion, params, col_name)
                )

            tmp_groups_2 = []
            for gdf2 in new_groups_2:
                tmp_groups_2.extend(
                    self._apply_blocking_criterion(gdf2, criterion, params, col_name)
                )

            new_groups_1 = tmp_groups_1
            new_groups_2 = tmp_groups_2

        # Now we have lists of grouped sub-dataframes for df1 and df2.
        # We need to match sub-dataframes that share the same group key(s).
        # We'll build a dictionary keyed by each group's name or first row's grouping key, etc.

        def group_by_block_id(group_list):
            """
            group_list is a list of (group_key, sub_df).
            group_key can be a single value or tuple if we grouped by multiple columns.
            We'll store them in a dict to match keys across df1 and df2.
            """
            result = {}
            for block_key, gdf in group_list:
                result[block_key] = gdf
            return result

        # But note that _apply_blocking_criterion returns a list of (key, sub_df).
        # If your code returns only sub_dfs, adapt accordingly.

        if isinstance(new_groups_1[0], tuple):
            dict1 = group_by_block_id(new_groups_1)
        else:
            # If your code returns just sub_dfs, no keys. Then you need a different approach:
            # For now, assume we have (key, sub_df).
            dict1 = {i: g for i, g in enumerate(new_groups_1)}

        if isinstance(new_groups_2[0], tuple):
            dict2 = group_by_block_id(new_groups_2)
        else:
            dict2 = {i: g for i, g in enumerate(new_groups_2)}

        # Match keys that appear in both
        for block_key, sub1 in dict1.items():
            if block_key in dict2:
                sub2 = dict2[block_key]
                yield (sub1, sub2)

    def _apply_blocking_criterion(self, df, criterion, params, col_name):
        """
        Example from your DataGrouper:
            - 'first_letter': group by first letter of column
            - 'blocking_column': group by one or more columns
        Return a list of (group_key, sub_df).
        """
        if criterion == "first_letter":
            # Group by the first letter of the col_name
            grouped = df.groupby(df[col_name].astype(str).str[0], dropna=False)
            return list(grouped)  # each item is (group_key, sub_df)

        elif criterion == "blocking_column":
            blocking_columns = params.get("blocking_column")
            if isinstance(blocking_columns, list):
                for bc in blocking_columns:
                    df[bc] = df[bc].astype(str).fillna("NA")
                grouped = df.groupby(blocking_columns, dropna=False)
                return list(grouped)
            elif isinstance(blocking_columns, str):
                grouped = df.groupby(blocking_columns, dropna=False)
                return list(grouped)
            else:
                raise ValueError(f"Invalid blocking_columns: {blocking_columns}")

        else:
            raise ValueError(f"Unsupported blocking criterion: {criterion}")

    def _match_subdataframes(
        self, sub_df1, sub_df2, col_name, similarity_method, threshold, top_n
    ):
        """
        For the given sub-dataframes that share a blocking key,
        compute top-n similarities above threshold on col_name.
        Returns a DataFrame [df1_index, df2_index, similarity_score].
        """
        # Build & fit a TfidfVectorizer on combined text
        vectorizer = self._create_vectorizer(similarity_method)

        combined_text = pd.concat([sub_df1[col_name], sub_df2[col_name]]).astype(str)
        vectorizer.fit(combined_text)

        tfidf_a = vectorizer.transform(sub_df1[col_name].astype(str))
        tfidf_b = vectorizer.transform(sub_df2[col_name].astype(str))

        # Optional pruning
        # if similarity_method in ("tfidf", "numeric") and self.tfidf_min_value > 0:
        #     tfidf_a = self._prune_tfidf_matrix(tfidf_a)
        #     tfidf_b = self._prune_tfidf_matrix(tfidf_b)

        # Sparse dot product top-n
        results_sparse = sp_matmul_topn(
            tfidf_a, tfidf_b.T, top_n=top_n, threshold=threshold, n_threads=-1
        )
        result_df = self._sparse_results_to_df(
            results_sparse, sub_df1.index, sub_df2.index
        )
        return result_df

    def _create_vectorizer(self, similarity_method):
        """
        Creates and returns a TfidfVectorizer (not the final matrix) based on the method.
        """
        if similarity_method == "numeric":
            return TfidfVectorizer(
                tokenizer=lambda x: re.findall(r"\d+", x),
                preprocessor=None,
                lowercase=False,
                # stop_words="english",
            )
        elif similarity_method == "tfidf":
            return TfidfVectorizer(
                analyzer="char_wb",
                preprocessor=None,
                lowercase=True,
                ngram_range=(2, 3),
                norm="l2",
                smooth_idf=True,
                use_idf=True,
                # stop_words="english",
            )
        else:
            raise ValueError(f"Unsupported similarity method: {similarity_method}")

    def _sparse_results_to_df(self, sparse_matrix, index_a, index_b):
        """
        Convert a sparse (n_rows_a x n_rows_b) matrix of dot products
        to a DataFrame: [df1_index, df2_index, similarity_score].
        """
        coo = sparse_matrix.tocoo()
        rows, cols, data = coo.row, coo.col, coo.data

        df1_indices = index_a.to_numpy()[rows]
        df2_indices = index_b.to_numpy()[cols]

        result_df = pd.DataFrame(
            {
                "df1_index": df1_indices,
                "df2_index": df2_indices,
                "similarity_score": data,
            }
        )
        return result_df

    def _prune_tfidf_matrix(self, tfidf_matrix):
        """
        If self.tfidf_min_value > 0, zero out entries below that and re-sparsify.
        """
        mask = tfidf_matrix.data < self.tfidf_min_value
        tfidf_matrix.data[mask] = 0
        tfidf_matrix.eliminate_zeros()
        return tfidf_matrix


if __name__ == "__main__":
    df = pd.read_csv("test_data/100k.csv", nrows=500_000, encoding="latin-1")
    df = df[["BorrowerName", "BorrowerAddress", "BorrowerCity"]]
    df.reset_index(inplace=True)
    df1 = df.iloc[:25_000]
    df2 = df.iloc[25_000:100_000]

    column_thresholds = {
        "BorrowerName": {
            "threshold": 0.8,
            "blocking_column": ["BorrowerCity"],
            "blocking_criteria": ["blocking_column"],
            "similarity_method": "tfidf",
        },
        "BorrowerAddress": {
            "threshold": 0.9,
            "blocking_column": ["BorrowerCity"],
            "blocking_criteria": ["blocking_column"],
            "similarity_method": "tfidf",
        },
    }

    matcher = TwoDFMatcher()

    # Let's keep top 3 matches per row, apply a final global threshold if needed
    results = matcher.match_two_dataframes_blocking(
        df1=df1,
        df2=df2,
        column_thresholds=column_thresholds,
        top_n=3,
        global_threshold=0.0,
        combine_method="OR",
    )

    print("----- MATCHING RESULTS -----")
    print(results)

    pivoted = results.pivot_table(
        index=["df1_index", "df2_index"], columns="column", values="similarity_score"
    ).reset_index()

    pivoted.rename(
        columns=lambda c: (
            f"similarity_{c}" if c not in ["df1_index", "df2_index"] else c
        ),
        inplace=True,
    )

    merged = pivoted.merge(
        df1, left_on="df1_index", right_index=True, how="left"
    ).merge(
        df2,
        left_on="df2_index",
        right_index=True,
        how="left",
        suffixes=("_df1", "_df2"),
    )
    pdb.set_trace()
    column_thresholds = {
        "BorrowerName": {
            "threshold": 0.6,
            "blocking_column": ["BorrowerCity"],
            "blocking_criteria": ["blocking_column"],
            "similarity_method": "tfidf",
        },
        "BorrowerAddress": {
            "threshold": 0.9,
            "blocking_column": ["BorrowerCity"],
            "blocking_criteria": ["blocking_column"],
            "similarity_method": "tfidf",
        },
    }
    pdb.set_trace()
    EF = SimilarityMatrixGenerator(df, column_thresholds, combine_method="AND")
    clustered_df = EF.cluster_data()
