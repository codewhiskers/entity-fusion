import pandas as pd
import numpy as np

np.seterr(divide="ignore", invalid="ignore")  # need to fix this later
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from scipy.sparse import lil_matrix, coo_matrix
import networkx as nx
import plotly.graph_objects as go
from IPython.display import display
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import re
from collections import deque, Counter, defaultdict
import pickle
import os
import datetime
import pdb
import json


class Entity_Fusion:

    def cluster_data(self):
        # Check if the parameters have been initialized
        if self.df is None or self.column_thresholds is None or self.id_column is None:
            raise ValueError(
                "Parameters have not been initialized. Please call initialize_parameters() first."
            )

        # Separate the DataFrame into pre-clustered and unclustered parts
        pre_clustered_df = self.df[self.df["cluster_label"].notnull()].copy()
        unclustered_df = self.df[self.df["cluster_label"].isnull()].copy()
        if unclustered_df.empty:
            print("No unclustered rows to process.")
            return self.df
        self.df = unclustered_df.reset_index(drop=True)
        self.create_similarity_matrices()
        if not pre_clustered_df.empty:
            pre_clustered_df = pre_clustered_df.reset_index(drop=True)
            self.df.reset_index(drop=True, inplace=True)
            combined_df = pd.concat([pre_clustered_df, self.df], ignore_index=True)

            pre_clustered_indices = pre_clustered_df.index
            unclustered_indices = self.df.index + len(pre_clustered_df)

            self.df_sim = self._create_combined_similarity_matrix(
                combined_df, pre_clustered_indices, unclustered_indices
            )
            pdb.set_trace()
            self._construct_similarity_graph()

        self.create_similarity_matrices()
        self._construct_similarity_graph()
        self.clusters = self._find_clusters_from_graph(self.graph)

        self.df["cluster_label"] = self.df[self.id_column].map(self.clusters)
        self.df = self.find_unclustered(self.df)
        if self.compare:
            matched = self.df.groupby("cluster_label")["df"].nunique().reset_index()
            matched = matched.rename(columns={"df": "matched"})
            matched["matched"] = np.where(matched["matched"] == 2, True, False)
            self.df = self.df.merge(matched, on="cluster_label", how="left")

        return self.df

    def _create_similarity_matrix(
        self, group_tfidf, group_ids, column_name, threshold, similarity_method
    ):
        if similarity_method == "exact":
            return self._create_exact_match_matrix(group_tfidf, group_ids, column_name)

        def compute_cosine_similarity_chunk(start_idx, end_idx, group_tfidf, threshold):
            chunk_matrix = cosine_similarity(
                group_tfidf[start_idx:end_idx], group_tfidf
            )
            mask = chunk_matrix >= threshold
            chunk_matrix = np.where(mask, chunk_matrix, 0)
            return start_idx, end_idx, chunk_matrix

        chunk_size = 2_000
        n_samples = group_tfidf.shape[0]

        if n_samples > 5_000:
            use_progress_bar = True
        else:
            use_progress_bar = False

        cos_sim_sparse = lil_matrix((n_samples, n_samples), dtype=np.float32)
        iter_range = range(0, n_samples, chunk_size)
        if use_progress_bar:
            iter_range = tqdm(
                iter_range,
                desc=f"Computing cosine similarity in chunks for {column_name}",
                leave=False,
            )
        for start_idx in iter_range:
            end_idx = min(start_idx + chunk_size, n_samples)
            start_idx, end_idx, chunk_matrix = compute_cosine_similarity_chunk(
                start_idx, end_idx, group_tfidf, threshold
            )
            cos_sim_sparse[start_idx:end_idx, :] = chunk_matrix

        cos_sim_sparse = cos_sim_sparse.tocsr()
        coo = coo_matrix(cos_sim_sparse)
        rows, cols, values = coo.row, coo.col, coo.data

        group_ids = np.array(group_ids)

        mask = rows != cols
        filtered_rows = rows[mask]
        filtered_cols = cols[mask]
        filtered_values = values[mask]

        all_similarities = np.vstack(
            (
                group_ids[filtered_rows],
                group_ids[filtered_cols],
                filtered_values,
            )
        ).T

        sim_df = pd.DataFrame(
            all_similarities,
            columns=[
                f"{column_name}_1_id",
                f"{column_name}_2_id",
                f"{column_name}_similarity",
            ],
        )
        return sim_df

    def _create_combined_similarity_matrix(
        self,
        pre_clustered_df,
        unclustered_df,
        pre_clustered_indices,
        unclustered_indices,
    ):
        processed_dfs = []

        for column, params in self.column_thresholds.items():
            pre_clustered_df[column] = pre_clustered_df[column].astype(str)
            unclustered_df[column] = unclustered_df[column].astype(str)
            combined_df = pd.concat(
                [pre_clustered_df, unclustered_df], ignore_index=True
            )
            similarity_method = params.get("similarity_method", "tfidf")

            if similarity_method == "numeric":
                vectorizer = TfidfVectorizer(
                    tokenizer=lambda x: re.findall(r"\d+", x),
                    preprocessor=None,
                    lowercase=False,
                    stop_words="english",
                )
                X_tfidf = vectorizer.fit_transform(combined_df[column])
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
                X_tfidf = vectorizer.fit_transform(combined_df[column])
            elif similarity_method == "exact":
                vectorizer = None
                X_tfidf = combined_df
                params["threshold"] = 1

            grouped_data = self.group_dataframe(combined_df, params, column)

            grouped_processed_dfs_list = []
            for _, group in tqdm(grouped_data, desc=f"Processing groups for {column}"):
                group_tfidf = (
                    X_tfidf[group.index, :] if vectorizer is not None else group[column]
                )

                pre_group_tfidf = group_tfidf[
                    np.isin(group.index, pre_clustered_indices), :
                ]
                un_group_tfidf = group_tfidf[
                    np.isin(group.index, unclustered_indices), :
                ]

                # Process the similarities between pre-clustered and unclustered
                result = self.process_group(
                    group,
                    column,
                    (pre_group_tfidf, un_group_tfidf),
                    similarity_method,
                    params["threshold"],
                    pre_clustered_indices,
                    unclustered_indices,
                )
                grouped_processed_dfs_list.append(result)

            grouped_processed_dfs = pd.concat(
                grouped_processed_dfs_list, ignore_index=True
            )

            processed_dfs.append(grouped_processed_dfs)

        if not processed_dfs:
            raise ValueError("No processed DataFrames to merge.")

        # Initialize merged DataFrame
        df_sim = processed_dfs[0]
        for i in range(1, len(processed_dfs)):
            df_sim = self.merge_dataframes(df_sim, processed_dfs[i])
        df_sim = df_sim.fillna(0)
        return df_sim

    def create_similarity_matrices(self):
        processed_dfs = []
        for column, params in self.column_thresholds.items():
            df = self.df.copy()
            df[column] = df[column].astype(str)
            similarity_method = params.get("similarity_method", "tfidf")
            data = df[column].tolist()

            if similarity_method == "numeric":
                vectorizer = TfidfVectorizer(
                    tokenizer=lambda x: re.findall(r"\d+", x),
                    preprocessor=None,
                    lowercase=False,
                    stop_words="english",
                )
                X_tfidf = vectorizer.fit_transform(data)
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
                X_tfidf = vectorizer.fit_transform(data)
            elif similarity_method == "exact":
                vectorizer = None
                X_tfidf = df
                params["threshold"] = 1

            grouped_data = self.group_dataframe(df, params, column)

            grouped_processed_dfs_list = []
            for group_name, group in tqdm(
                grouped_data, desc=f"Processing groups for {column}"
            ):
                result = self.process_group(
                    group_name,
                    group,
                    column,
                    (
                        X_tfidf[group.index, :]
                        if similarity_method in ["tfidf", "numeric"]
                        else group[column]
                    ),
                    similarity_method,
                    params["threshold"],
                )
                grouped_processed_dfs_list.append(result)

            grouped_processed_dfs = pd.concat(
                grouped_processed_dfs_list, ignore_index=True
            )

            processed_dfs.append(grouped_processed_dfs)

        if not processed_dfs:
            raise ValueError("No processed DataFrames to merge.")

        # Initialize merged DataFrame
        df_sim = processed_dfs[0]
        for i in range(1, len(processed_dfs)):
            df_sim = self.merge_dataframes(df_sim, processed_dfs[i])
        df_sim = df_sim.fillna(0)
        self.df_sim = df_sim
        return df_sim

    def process_group(
        self, group_name, group, column, X_tfidf, similarity_method, threshold
    ):
        group_ids = group[self.id_column].tolist()

        if similarity_method == "tfidf" or similarity_method == "numeric":
            group_tfidf = X_tfidf
        elif similarity_method == "exact":
            group_tfidf = group[column].tolist()

        grouped_processed_df = self._create_similarity_matrix(
            group_tfidf, group_ids, column, threshold, similarity_method
        )

        if not grouped_processed_df.empty:
            grouped_processed_df.rename(
                columns={
                    f"{column}_1_id": "id1",
                    f"{column}_2_id": "id2",
                    f"{column}_similarity": f"{column}_similarity",
                },
                inplace=True,
            )
            grouped_processed_df = grouped_processed_df[
                ["id1", "id2", f"{column}_similarity"]
            ]
            return grouped_processed_df
        else:
            return pd.DataFrame(columns=["id1", "id2", f"{column}_similarity"])

    def process_combined_group(
        self,
        # group_name,
        group,
        column,
        group_tfidf,
        similarity_method,
        threshold,
        pre_clustered_indices,
        unclustered_indices,
    ):
        pre_group_tfidf, un_group_tfidf = group_tfidf

        group_ids = group[self.id_column].tolist()
        pre_group_ids = [
            group_ids[i]
            for i in range(len(group_ids))
            if np.isin(group.index[i], pre_clustered_indices)
        ]
        un_group_ids = [
            group_ids[i]
            for i in range(len(group_ids))
            if np.isin(group.index[i], unclustered_indices)
        ]

        if similarity_method == "tfidf" or similarity_method == "numeric":
            similarity_matrix = cosine_similarity(un_group_tfidf, pre_group_tfidf)
        elif similarity_method == "exact":
            similarity_matrix = self._create_exact_match_matrix(
                group[column].tolist(), group_ids, column
            )
            return similarity_matrix

        threshold_matrix = similarity_matrix >= threshold
        sim_values = similarity_matrix[threshold_matrix]

        idx_pairs = np.argwhere(threshold_matrix)
        id1s = [un_group_ids[idx[0]] for idx in idx_pairs]
        id2s = [pre_group_ids[idx[1]] for idx in idx_pairs]

        result_df = pd.DataFrame(
            {
                f"{column}_1_id": id1s,
                f"{column}_2_id": id2s,
                f"{column}_similarity": sim_values,
            }
        )

        result_df.rename(
            columns={
                f"{column}_1_id": "id1",
                f"{column}_2_id": "id2",
                f"{column}_similarity": f"{column}_similarity",
            },
            inplace=True,
        )

        return result_df

    def _construct_similarity_graph(self):
        print("Computing similarity graph...")

        if self.graph is None:
            self.graph = defaultdict(set)

        masks = []
        for col, params in self.column_thresholds.items():
            masks.append(
                self.df_sim[f"{col}_similarity"].astype(float) >= params["threshold"]
            )

        if self.conditional == "AND":
            final_mask = np.logical_and.reduce(masks)
        else:
            final_mask = np.logical_or.reduce(masks)

        filtered_df = self.df_sim[final_mask]

        if self.pre_clustered_df is not None:
            exclude_set = set(
                zip(
                    self.pre_clustered_df[self.pre_clustered_df["match"] == False][
                        "id1"
                    ],
                    self.pre_clustered_df[self.pre_clustered_df["match"] == False][
                        "id2"
                    ],
                )
            )
            reverse_exclude_set = set((y, x) for x, y in exclude_set)
            exclude_set.update(reverse_exclude_set)

            include_set = set(
                zip(
                    self.pre_clustered_df[self.pre_clustered_df["match"] == True][
                        "id1"
                    ],
                    self.pre_clustered_df[self.pre_clustered_df["match"] == True][
                        "id2"
                    ],
                )
            )
            reverse_include_set = set((y, x) for x, y in include_set)
            include_set.update(reverse_include_set)
        else:
            exclude_set = set()
            include_set = set()

        idx1 = filtered_df["id1"].values
        idx2 = filtered_df["id2"].values
        edges = list(zip(idx1, idx2))

        # Create a hash map (dictionary) for fast ID lookup
        # id_map = self.df[self.id_column].to_dict()

        for edge in tqdm(edges, desc="Adding edges to the graph"):
            if edge[0] is None or edge[1] is None:
                print(f"Invalid edge found: {edge}")
                continue
            if (
                edge[0] is not None
                and edge[1] is not None
                and (edge[0], edge[1]) not in exclude_set
                and (edge[1], edge[0]) not in exclude_set
            ):
                self.graph[edge[0]].add(edge[1])
                self.graph[edge[1]].add(edge[0])

        for id1, id2 in include_set:
            try:  # Need to fix this later
                node1 = self.df[self.df[self.id_column] == id1][self.id_column].values[
                    0
                ]
                node2 = self.df[self.df[self.id_column] == id2][self.id_column].values[
                    0
                ]
                self.graph[node1].add(node2)
                self.graph[node2].add(node1)
            except:
                continue

        print("Similarity graph constructed.")
        return self.graph

    def group_dataframe(self, df, params, column):
        # Combine filtering conditions into a single operation
        df = df[
            (df[column].notnull())
            & (df[column] != "")
            & (df[column] != "Unknown")
            & (df[column].str.lower() != "nan")
            & (df[column].str.lower() != "none")
        ]

        blocking_criteria = params.get("blocking_criteria", None)

        if blocking_criteria is not None:
            grouped_data = [df]

            for criterion in blocking_criteria:
                new_groups = []
                for group in grouped_data:
                    if criterion == "first_letter":
                        new_groups.extend(
                            list(group.groupby(group[column].str[0], sort=False))
                        )
                    elif criterion == "blocking_column":
                        blocking_columns = params.get("blocking_column")
                        if isinstance(blocking_columns, list):
                            new_groups.extend(
                                list(
                                    group.groupby(
                                        [group[col] for col in blocking_columns],
                                        sort=False,
                                    )
                                )
                            )
                        else:
                            new_groups.extend(
                                list(group.groupby(group[blocking_columns], sort=False))
                            )
                    else:
                        raise ValueError(f"Unsupported criterion: {criterion}")

                # Filter out groups with a single entry
                grouped_data = [grp for _, grp in new_groups if len(grp) > 1]

            return [(group_name, group) for group_name, group in new_groups]
        else:
            return [(None, df)]

    def initialize_parameters(
        self,
        df,
        id_column,
        column_thresholds,
        df2=None,
        conditional="OR",
        pre_clustered_df=None,
    ):
        # Ensure ID column is specified and unique
        if id_column not in df.columns:
            raise ValueError(
                f"The ID column '{id_column}' is not present in the dataframe."
            )
        # Check for unique IDs in the first dataframe
        if not df[id_column].is_unique:
            duplicated_ids = df[id_column][df[id_column].duplicated()].unique()
            raise ValueError(
                f"The ID column '{id_column}' must contain unique values. Duplicated IDs: {duplicated_ids}"
            )

        self.column_thresholds = column_thresholds
        self.id_column = id_column if id_column else "id"
        self.conditional = conditional
        self.pre_clustered_df = pre_clustered_df
        self.df_sim = None
        self.graph = None
        self.clusters = None
        self.stopwords = set(ENGLISH_STOP_WORDS)
        self.df = df.reset_index(drop=True)

        if df2 is not None:
            self.compare = True
            self.df2 = df2.reset_index(drop=True)
            self.df["df"] = 1
            self.df2["df"] = 2
            self.df = pd.concat([self.df, self.df2], ignore_index=True)
            if not self.df[id_column].is_unique:
                duplicated_ids = df[id_column][df[id_column].duplicated()].unique()
                raise ValueError(
                    f"The ID column '{id_column}' must contain unique values. Duplicated IDs: {duplicated_ids}"
                )
        else:
            self.compare = False
        if "cluster_label" not in self.df.columns:
            self.df["cluster_label"] = np.nan

    def _find_common_prefixes_and_postfixes(self, data, min_length=2):
        threshold = 5
        all_words = [word for text in data for word in text.split()]
        word_counts = Counter(all_words)
        common_affixes = [
            word
            for word, count in word_counts.items()
            if count >= threshold and len(word) >= min_length
        ]
        return common_affixes

    def find_unclustered(self, df):
        # Check if the 'cluster_label' column exists, if not, create it
        if "cluster_label" not in df.columns:
            df["cluster_label"] = np.nan

        # Get the maximum cluster label currently in the dataframe
        max_label = int(
            df["cluster_label"].max() if pd.notnull(df["cluster_label"].max()) else -1
        )

        # Find all rows where the cluster label is NaN
        unclustered_mask = df["cluster_label"].isnull()

        # Count the number of unclustered rows
        num_unclustered = int(unclustered_mask.sum())

        # Assign new cluster labels to unclustered rows
        df.loc[unclustered_mask, "cluster_label"] = range(
            max_label + 1, max_label + 1 + num_unclustered
        )

        return df

    def merge_dataframes(self, left_df, right_df):
        return pd.merge(left_df, right_df, on=["id1", "id2"], how="outer")

    def _create_exact_match_matrix(self, data, group_ids, column_name):
        matches = []
        value_to_indices = defaultdict(list)

        for idx, value in enumerate(data):
            value_to_indices[value].append(group_ids[idx])

        for indices in tqdm(
            value_to_indices.values(),
            desc=f"Processing exact matches for {column_name}",
        ):
            if len(indices) > 1:
                for i in range(len(indices)):
                    for j in range(i + 1, len(indices)):
                        matches.append([indices[i], indices[j], 1])

        match_df = pd.DataFrame(
            matches,
            columns=[
                f"{column_name}_1_id",
                f"{column_name}_2_id",
                f"{column_name}_similarity",
            ],
        )

        return match_df

    def update_clusters_with_new_data(self, old_df, new_df, old_cluster_label_col):
        if old_cluster_label_col not in old_df.columns:
            raise ValueError(
                f"Old dataframe does not have '{old_cluster_label_col}' column"
            )

        new_df["cluster_label"] = np.nan
        self.df = pd.concat([old_df, new_df], ignore_index=True)
        self.df = self.find_unclustered(self.df)

        # Only process the new data
        self.df_sim = None  # Reset similarity matrix
        self.create_similarity_matrices()
        self._construct_similarity_graph()
        new_clusters = self._find_clusters_from_graph(self.graph)

        # Map old cluster labels to new ones
        old_to_new_cluster_map = {}
        for _, row in self.df.iterrows():
            old_cluster = row[old_cluster_label_col]
            new_cluster = new_clusters.get(row[self.id_column], None)
            if pd.notna(old_cluster) and new_cluster is not None:
                if old_cluster not in old_to_new_cluster_map:
                    old_to_new_cluster_map[old_cluster] = new_cluster
        self.df[old_cluster_label_col] = (
            self.df[old_cluster_label_col]
            .map(old_to_new_cluster_map)
            .fillna(self.df["cluster_label"])
        )

        # Update self.clusters with the old cluster IDs
        for old_cluster, new_cluster in old_to_new_cluster_map.items():
            for node, cluster_id in new_clusters.items():
                if cluster_id == new_cluster:
                    new_clusters[node] = old_cluster
        self.clusters = new_clusters

        return self.df

    def return_cluster_data_logic_dataframe(self):
        print("Creating cluster data logic DataFrame...")
        # Reset index for the original DataFrame
        original_df = self.df.reset_index()

        # Use a single loop to update idx1 and idx2 mappings and cluster labels
        for column in self.column_thresholds.keys():
            index_to_col = original_df.set_index("index")[column]
            # Map the values in one go
            self.df_sim[f"{column}_id1"] = self.df_sim["id1"].map(index_to_col)
            self.df_sim[f"{column}_id2"] = self.df_sim["id2"].map(index_to_col)

        # Map cluster labels for both idx1 and idx2
        self.df_sim["cluster_label_id1"] = self.df_sim["id1"].map(self.clusters)
        self.df_sim["cluster_label_id2"] = self.df_sim["id2"].map(self.clusters)

        # Vectorized operation to create the new column
        self.df_sim["cluster_label"] = np.where(
            self.df_sim["cluster_label_id1"] == self.df_sim["cluster_label_id2"],
            self.df_sim["cluster_label_id1"],
            None,
        )
        self.df_sim.drop(
            columns=["cluster_label_id1", "cluster_label_id2"], inplace=True
        )
        print("Cluster data logic DataFrame created.")
        return self.df_sim

    # Function to visualize a specific cluster interactively
    def visualize_cluster(self, cluster_id, hover_columns=None):
        if self.graph is None or self.clusters is None:
            print("Graph or clusters not initialized.")
            return

        # Create a NetworkX graph from the clusters and edges
        G = nx.Graph()

        # Add edges to the graph
        for node, neighbors in self.graph.items():
            for neighbor in neighbors:
                G.add_edge(node, neighbor)

        # Extract nodes in the cluster
        nodes_in_cluster = [
            node for node, cluster in self.clusters.items() if cluster == cluster_id
        ]
        subgraph = G.subgraph(nodes_in_cluster)

        pos = nx.spring_layout(subgraph)

        edge_x = []
        edge_y = []
        for edge in subgraph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.append(x0)
            edge_x.append(x1)
            edge_x.append(None)
            edge_y.append(y0)
            edge_y.append(y1)
            edge_y.append(None)

        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            line=dict(width=1, color="#888"),
            hoverinfo="none",
            mode="lines",
        )

        node_x = []
        node_y = []
        hover_texts = []
        for node in subgraph.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)

            # Generate custom hover text
            hover_text = f"Node: {node}"
            if hover_columns:
                for col in hover_columns:
                    if col in self.df.columns:
                        col_value = self.df[self.df[self.id_column] == node][
                            col
                        ].values[0]
                        hover_text += f"<br>{col}: {col_value}"
            hover_texts.append(hover_text)

        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=[str(node) for node in subgraph.nodes()],
            textposition="bottom center",
            hovertext=hover_texts,
            hoverinfo="text",
            marker=dict(
                showscale=True,
                colorscale="YlGnBu",
                size=10,
                colorbar=dict(
                    thickness=15,
                    title="Node Connections",
                    xanchor="left",
                    titleside="right",
                ),
            ),
        )

        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title=f"<br>Cluster {cluster_id} Graph",
                titlefont_size=16,
                showlegend=False,
                hovermode="closest",
                margin=dict(b=20, l=5, r=5, t=40),
                annotations=[
                    dict(
                        text="Interactive graph where nodes can be moved",
                        showarrow=False,
                        xref="paper",
                        yref="paper",
                        x=0.005,
                        y=-0.002,
                    )
                ],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            ),
        )
        display(fig)

    def generate_cluster_javascript(
        self, cluster_id, hover_columns=None, threshold=3, size_multiplier=10
    ):
        """
        Generate a JavaScript file for a specific cluster with supernodes and individual nodes.

        Parameters:
            cluster_id (int): The cluster ID to extract.
            hover_columns (list): List of columns to include as hover attributes.
            threshold (int): Minimum number of neighbors to form a super cluster.
            size_multiplier (int): Multiplier to determine node size based on group size.
        """

        # Create a NetworkX graph
        G = nx.Graph()

        # Build the graph from `self.graph`
        for node, neighbors in self.graph.items():
            for neighbor in neighbors:
                G.add_edge(node, neighbor)

        # Identify super clusters
        super_clusters = defaultdict(list)
        visited = set()

        for node in G.nodes:
            if node not in visited and self.clusters.get(node) == cluster_id:
                # Find neighbors within the same cluster
                neighbors = [
                    n for n in G.neighbors(node) if self.clusters.get(n) == cluster_id
                ]
                if len(neighbors) >= threshold:
                    # Mark node and neighbors as part of the super cluster
                    super_clusters[cluster_id].append([node] + neighbors)
                    visited.update([node] + neighbors)

        # Create Cytoscape elements
        elements = {"nodes": [], "edges": []}
        node_to_supernode = {}

        # Add supernodes
        for cluster_id, groups in super_clusters.items():
            for i, group in enumerate(groups):
                supernode_id = f"Super_{cluster_id}_{i}"
                node_to_supernode.update({n: supernode_id for n in group})

                # Aggregate dynamic attributes for the supernode
                aggregated_attributes = {col: [] for col in hover_columns or []}
                for node in group:
                    for col in hover_columns or []:
                        if col in self.df.columns:
                            col_value = self.df[self.df[self.id_column] == node][
                                col
                            ].values
                            if len(col_value) > 0:
                                aggregated_attributes[col].append(col_value[0])

                # Add attributes to supernode
                attributes = {
                    col: aggregated_attributes[col] for col in aggregated_attributes
                }
                elements["nodes"].append(
                    {
                        "data": {
                            "id": supernode_id,
                            "label": f"Supernode {cluster_id}_{i}",
                            "size": len(group)
                            * size_multiplier,  # Size based on number of nodes in the cluster
                            **attributes,
                        }
                    }
                )

        # Add individual nodes not part of any super cluster
        remaining_nodes = {
            node for node in G.nodes if self.clusters.get(node) == cluster_id
        } - visited
        for node in remaining_nodes:
            # Extract attribute values dynamically from the dataframe
            attributes = {}
            for col in hover_columns or []:
                if col in self.df.columns:
                    col_value = self.df[self.df[self.id_column] == node][col].values
                    attributes[col] = col_value[0] if len(col_value) > 0 else "N/A"

            elements["nodes"].append(
                {
                    "data": {
                        "id": str(node),
                        "label": str(node),
                        "size": size_multiplier,  # Default size for individual nodes
                        **attributes,
                    }
                }
            )

        # Add edges
        for u, v in G.edges:
            if (
                self.clusters.get(u) == cluster_id
                and self.clusters.get(v) == cluster_id
            ):
                # Map nodes to supernodes if applicable
                source = node_to_supernode.get(u, str(u))
                target = node_to_supernode.get(v, str(v))

                # Avoid duplicate edges between supernodes
                if source != target:
                    elements["edges"].append(
                        {"data": {"source": source, "target": target}}
                    )

        # Save as JavaScript for use in Cytoscape
        output_path = f"cluster_{cluster_id}_graph.js"
        with open(output_path, "w") as f:
            f.write(f"const data = {json.dumps(elements, indent=4)};")

        print(f"Cluster graph for cluster {cluster_id} saved as {output_path}")

    def _find_clusters_from_graph(self, graph):
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
        nodes = list(graph.keys())

        for node in tqdm(nodes, desc="Processing nodes"):
            if node not in visited:
                cluster = bfs(graph, node, visited)
                clusters.append(cluster)

        cluster_map = {}
        for cluster_id, cluster in enumerate(clusters):
            for node in cluster:
                cluster_map[node] = cluster_id
        return cluster_map
