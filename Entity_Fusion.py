import numpy as np
import pandas as pd
from scipy.sparse import lil_matrix, coo_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
import pdb
import re
from collections import defaultdict, deque


class SimilarityMatrixGenerator:
    def __init__(self, df, column_thresholds):
        self.df = df
        self.column_thresholds = column_thresholds
        self.df_sim = None
        self.graph = defaultdict(set)
        self.clusters = {}


    def create_similarity_matrices(self):
        """
        Create similarity matrices for specified columns and store them as lists of tuples.
        """
        if not self.column_thresholds:
            raise ValueError("No column thresholds specified for similarity computation.")

        similarity_calculator = SimilarityCalculator()
        data_grouper = DataGrouper(self.df)
        similarity_results = []

        for column, params in self.column_thresholds.items():
            if column not in self.df.columns:
                raise ValueError(f"Column '{column}' not found in the dataframe.")

            column_results = []

            # Group and process with a progress bar
            groups = list(data_grouper.group_dataframe(params, column))  # Convert generator to list for tqdm
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
                    column_results.extend(result.tolist())  # Convert NumPy array to list of tuples

            # Store results for this column as a list of tuples
            if column_results:
                similarity_results.append((column, column_results))

        self.similarity_results = similarity_results





    def _construct_similarity_graph(self):
        """
        Construct a graph from the similarity results stored as lists of tuples.
        """
        print("Constructing similarity graph...")
        
        for column, data in self.similarity_results:
            for id1, id2, similarity in data:
                if id1 != id2:
                    self.graph[id1].add(id2)
                    self.graph[id2].add(id1)



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
            raise ValueError("Graph not constructed. Run _construct_similarity_graph first.")
        if not self.clusters:
            raise ValueError("Clusters not found. Run _find_clusters_from_graph first.")

        # Map cluster labels to the dataframe
        self.df["cluster_label"] = self.df.index.map(self.clusters)

        # Assign unique cluster labels to nodes not in the graph
        unclustered = self.df["cluster_label"].isna()
        self.df.loc[unclustered, "cluster_label"] = range(
            max(self.clusters.values(), default=-1) + 1,
            max(self.clusters.values(), default=-1) + 1 + unclustered.sum()
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
                stop_words="english",
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

    def create_similarity_matrix(self, group_tfidf, group_ids, column_name, threshold, similarity_method):
        if similarity_method == "exact":
            return self._create_exact_match_matrix(group_tfidf, group_ids, column_name)
        return self._create_cosine_similarity_matrix(group_tfidf, group_ids, column_name, threshold)

    def _create_exact_match_matrix(self, group_tfidf, group_ids, column_name):
        # Logic for exact matching similarity matrix
        pass


    def _create_cosine_similarity_matrix(self, group_tfidf, group_ids, column_name, threshold):
        """
        Create a cosine similarity matrix with a progress bar for chunks,
        shown only if the number of chunks exceeds a threshold.
        """
        n_samples = group_tfidf.shape[0]
        chunk_size = 2_000
        num_chunks = (n_samples + chunk_size - 1) // chunk_size  # Calculate total chunks
        show_progress_bar = num_chunks > 3  # Show progress bar only if chunks exceed threshold
        cos_sim_sparse = lil_matrix((n_samples, n_samples), dtype=np.float32)

        # Outer progress bar, shown only when needed
        progress_bar = tqdm(
            total=n_samples,
            desc=f"Processing chunks for {column_name}",
            leave=False,
            disable=not show_progress_bar,
        )

        try:
            for start_idx in range(0, n_samples, chunk_size):
                end_idx = min(start_idx + chunk_size, n_samples)
                chunk_matrix = self._compute_cosine_similarity_chunk(
                    start_idx, end_idx, group_tfidf, threshold
                )
                cos_sim_sparse[start_idx:end_idx, :] = chunk_matrix

                # Update the progress bar
                progress_bar.update(end_idx - start_idx)
        finally:
            progress_bar.close()  # Ensure progress bar is properly closed

        return self._finalize_similarity_matrix(cos_sim_sparse, group_ids, column_name)


    def _compute_cosine_similarity_chunk(self, start_idx, end_idx, group_tfidf, threshold):
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
        similarities = np.vstack((
            group_ids[filtered_rows],
            group_ids[filtered_cols],
            filtered_values,
        )).T

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
            grouped_data = self._apply_blocking_criterion(grouped_data, criterion, params, column)
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
                    raise ValueError(f"Invalid blocking_columns type: {type(blocking_columns)}")
            else:
                raise ValueError(f"Unsupported blocking criterion: {criterion}")
        return [grp for _, grp in new_groups if len(grp) > 1]



if __name__ == "__main__":
    df = pd.read_csv("public_up_to_150k_1_240930.csv", nrows=500_000, encoding="latin-1")
    # df = df_ppp[0:5_000].copy()
    df = df[["BorrowerName", "BorrowerAddress", "BorrowerCity"]]
    df.reset_index(inplace=True)
    column_threshold = {
        "BorrowerName": {
            "threshold": 0.9,
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
    EF = SimilarityMatrixGenerator(df, column_threshold)
    clustered_df = EF.cluster_data()
    print(clustered_df.head())