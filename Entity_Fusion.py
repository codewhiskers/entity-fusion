import pandas as pd
import numpy as np

np.seterr(divide="ignore", invalid="ignore")  # need to fix this later
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.auto import tqdm
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


class Entity_Fusion:

    def initialize_parameters(
        self,
        df,
        id_column,
        column_thresholds,
        df2=None,
        conditional="OR",
        pre_clustered_df=None,
        clustered_csv_path="cluster_files/clustered_data.csv",
        graph_path="cluster_files/graph.pkl",
        save_copies=True,
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
            
        if df2 is not None:
            self.compare = True
            if id_column not in df2.columns:
                raise ValueError(
                    f"The ID column '{id_column}' is not present in the second dataframe."
                )
            if not df2[id_column].is_unique:
                duplicated_ids = df2[id_column][df2[id_column].duplicated()].unique()
                raise ValueError(
                    f"The ID column '{id_column}' must contain unique values. Duplicated IDs: {duplicated_ids}"
                )
            conflicting_ids = set(df[id_column]).intersection(
                set(df2[id_column])
            )
            if conflicting_ids:
                raise ValueError(
                    f"ID conflict detected between df1 and df2. Conflicting IDs: {conflicting_ids}"
                )
            self.df2 = df2
        else:
            self.compare = False

        self.column_thresholds = column_thresholds
        self.id_column = id_column if id_column else "id"
        self.conditional = conditional
        self.pre_clustered_df = pre_clustered_df
        self.df_sim = None
        self.graph = None
        self.clusters = None
        self.clustered_csv_path = clustered_csv_path
        self.graph_path = graph_path
        self.save_copies = save_copies
        self.stopwords = set(ENGLISH_STOP_WORDS)

        # Load existing clustered data and graph if they exist
        if os.path.exists(self.clustered_csv_path):
            existing_data = pd.read_csv(self.clustered_csv_path)
            with open(self.graph_path, "rb") as file:
                self.graph = pickle.load(file)

            # Verify that IDs do not conflict
            conflicting_ids = set(df[self.id_column]).intersection(
                set(existing_data[self.id_column])
            )
            if conflicting_ids:
                raise ValueError(
                    f"ID conflict detected between new data and existing clustered data. Conflicting IDs: {conflicting_ids}"
                )

            self.df = pd.concat([existing_data, df], ignore_index=True)
        else:
            self.df = df.reset_index(drop=True)
        # Stack indices for the two dataframes
        if self.compare:
            self.df['df'] = 1  # Add a column to differentiate between the two dataframes
            self.df2['df'] = 2  # Add a column to differentiate between the two dataframes
            joined_for_indices = pd.concat([self.df, self.df2], ignore_index=True)  
            joined_for_indices = joined_for_indices.reset_index(drop=True)
            self.df = joined_for_indices[joined_for_indices['df'] == 1]
            self.df2 = joined_for_indices[joined_for_indices['df'] == 2]
        
        

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
            df["cluster_label"].max()
            if pd.notnull(df["cluster_label"].max())
            else -1
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
        return pd.merge(left_df, right_df, on=["idx1", "idx2"], how="outer")
        
    def _create_exact_match_matrix(self, data, group_indices, column_name, data2=None, group_indices2=None):
        matches = []
        value_to_indices = defaultdict(list)
        
        for idx, value in enumerate(data):
            value_to_indices[value].append(group_indices[idx])
        
        if data2 is not None:
            for idx, value in enumerate(data2):
                value_to_indices[value].append(group_indices2[idx])
        
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
                f"{column_name}_1_index",
                f"{column_name}_2_index",
                f"{column_name}_similarity",
            ],
        )
        
        return match_df
    
    def _create_similarity_matrix(
        self,
        group_tfidf,
        group_indices,
        column_name,
        threshold,
        similarity_method,
        group_tfidf2=None,
        group_indices2=None
    ):
        if similarity_method == "numeric_exact":
            return self._create_exact_match_matrix(
                group_tfidf, group_indices, column_name, group_tfidf2, group_indices2
            )

        def compute_cosine_similarity_chunk(start_idx, end_idx, group_tfidf, threshold, group_tfidf2=None):
            if group_tfidf2 is not None:
                chunk_matrix = cosine_similarity(
                    group_tfidf[start_idx:end_idx], group_tfidf2
                )
            else:
                chunk_matrix = cosine_similarity(
                    group_tfidf[start_idx:end_idx], group_tfidf
                )
            mask = chunk_matrix >= threshold
            chunk_matrix = np.where(mask, chunk_matrix, 0)
            return start_idx, end_idx, chunk_matrix

        chunk_size = 2_000
        n_samples = group_tfidf.shape[0]

        if group_tfidf2 is not None:
            n_samples2 = group_tfidf2.shape[0]
            cos_sim_sparse = lil_matrix((n_samples, n_samples2), dtype=np.float32)
            for start_idx in tqdm(range(0, n_samples, chunk_size), desc=f"Computing cosine similarity in chunks for {column_name}", leave=False):
                end_idx = min(start_idx + chunk_size, n_samples)
                start_idx, end_idx, chunk_matrix = compute_cosine_similarity_chunk(
                    start_idx, end_idx, group_tfidf, threshold, group_tfidf2
                )
                cos_sim_sparse[start_idx:end_idx, :] = chunk_matrix
        else:
            cos_sim_sparse = lil_matrix((n_samples, n_samples), dtype=np.float32)
            for start_idx in tqdm(range(0, n_samples, chunk_size), desc=f"Computing cosine similarity in chunks for {column_name}", leave=False):
                end_idx = min(start_idx + chunk_size, n_samples)
                start_idx, end_idx, chunk_matrix = compute_cosine_similarity_chunk(
                    start_idx, end_idx, group_tfidf, threshold
                )
                cos_sim_sparse[start_idx:end_idx, :] = chunk_matrix

        cos_sim_sparse = cos_sim_sparse.tocsr()
        coo = coo_matrix(cos_sim_sparse)
        rows, cols, values = coo.row, coo.col, coo.data

        group_indices = np.array(group_indices)
        if group_tfidf2 is not None:
            group_indices2 = np.array(group_indices2)
            all_similarities = np.vstack(
                (
                    group_indices[rows],
                    group_indices2[cols],
                    values,
                )
            ).T
        else:
            mask = rows != cols
            filtered_rows = rows[mask]
            filtered_cols = cols[mask]
            filtered_values = values[mask]

            all_similarities = np.vstack(
                (
                    group_indices[filtered_rows],
                    group_indices[filtered_cols],
                    filtered_values,
                )
            ).T

        sim_df = pd.DataFrame(
            all_similarities,
            columns=[
                f"{column_name}_1_index",
                f"{column_name}_2_index",
                f"{column_name}_similarity",
            ],
        )
        return sim_df

    def process_group(
        self,
        group_name,
        group,
        column,
        X_tfidf,
        similarity_method,
        threshold,
        group2=None,
        # X_tfidf2=None
    ):
        group_indices = group.index.tolist()
        # pdb.set_trace()
        if group2 is not None:
            group_indices2 = group2.index.tolist()
            if similarity_method == "tfidf" or similarity_method == "numeric":
                group_tfidf = X_tfidf[group_indices, :]
                group_tfidf2 = X_tfidf[group_indices2, :]
            elif similarity_method == "numeric_exact":
                group_tfidf = group[column].tolist()
                group_tfidf2 = group2[column].tolist()

        else:
            group_tfidf2 = None
            if similarity_method == "tfidf" or similarity_method == "numeric":
                group_tfidf = X_tfidf[group_indices, :]
            elif similarity_method == "numeric_exact":
                group_tfidf = group[column].tolist()
        # if not group[group['corporation_name'].str.contains('K & J')].empty:
        #     pdb.set_trace()
        # pdb.set_trace()
        grouped_processed_df = self._create_similarity_matrix(
            group_tfidf,
            group_indices,
            column,
            threshold,
            similarity_method,
            group_tfidf2,
            group_indices2
        )

        if not grouped_processed_df.empty:
            grouped_processed_df.rename(
                columns={
                    f"{column}_1_index": "idx1",
                    f"{column}_2_index": "idx2",
                    f"{column}_similarity": f"{column}_similarity",
                },
                inplace=True,
            )
            grouped_processed_df = grouped_processed_df[
                ["idx1", "idx2", f"{column}_similarity"]
            ]
            return grouped_processed_df
        else:
            return pd.DataFrame(columns=["idx1", "idx2", f"{column}_similarity"])



    def group_dataframe(self, df, params, column):
        df = df[df[column].notnull()]
        df = df[df[column] != ""]
        df = df[df[column] != "nan"]
        df = df[df[column] != "None"]
        
        blocking_criteria = params.get("blocking_criteria", None)

        if blocking_criteria is not None:
            grouped_data = [df]

            for criterion in blocking_criteria:
                new_groups = []
                for group in grouped_data:
                    if criterion == "first_letter":
                        new_groups.extend(list(group.groupby(group[column].str[0])))
                    elif criterion == "blocking_column":
                        blocking_columns = params.get("blocking_column")
                        if isinstance(blocking_columns, list):
                            new_groups.extend(
                                list(
                                    group.groupby(
                                        [group[col] for col in blocking_columns]
                                    )
                                )
                            )
                        else:
                            new_groups.extend(
                                list(group.groupby(group[blocking_columns]))
                            )
                    else:
                        raise ValueError(f"Unsupported criterion: {criterion}")

                grouped_data = [grp for _, grp in new_groups if len(grp) > 1]

            return [(group_name, group) for group_name, group in new_groups]
        else:
            return [(None, df)]

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
                    preprocessor=None,
                    lowercase=False,
                    ngram_range=(2, 3),
                    norm="l2",
                    smooth_idf=True,
                    use_idf=True,
                    stop_words="english",
                )
                X_tfidf = vectorizer.fit_transform(data)
            elif similarity_method == "numeric_exact":
                X_tfidf = df
            
            if self.df2 is not None:
                df2 = self.df2.copy()
                df2[column] = df2[column].astype(str)
                data2 = df2[column].tolist()
                X_tfidf = vectorizer.transform(data + data2) if similarity_method in ["tfidf", "numeric"] else df2
                grouped_data1 = self.group_dataframe(df, params, column)
                grouped_data2 = self.group_dataframe(df2, params, column)

                grouped_processed_dfs_list = []

                for (group_name1, group1) in grouped_data1:
                    for (group_name2, group2) in grouped_data2:
                        if group_name1 == group_name2:
                            result = self.process_group(
                                group_name1,
                                group1,
                                column,
                                X_tfidf,
                                similarity_method,
                                params["threshold"],
                                group2,
                                # X_tfidf2
                            )
                            grouped_processed_dfs_list.append(result)

                grouped_processed_dfs = pd.concat(
                    grouped_processed_dfs_list, ignore_index=True
                )

                processed_dfs.append(grouped_processed_dfs)
            else:
                grouped_data = self.group_dataframe(df, params, column)

                grouped_processed_dfs_list = []
                for group_name, group in tqdm(
                    grouped_data, desc=f"Processing groups for {column}"
                ):
                    result = self.process_group(
                        group_name,
                        group,
                        column,
                        X_tfidf,
                        similarity_method,
                        params["threshold"]
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




    def _construct_similarity_graph(self):
        print("Computing similarity graph...")

        if self.graph is None:
            self.graph = defaultdict(set)

        masks = []
        for col, params in self.column_thresholds.items():
            masks.append(self.df_sim[f"{col}_similarity"] >= params["threshold"])

        if self.conditional == "AND":
            final_mask = np.logical_and.reduce(masks)
        else:
            final_mask = np.logical_or.reduce(masks)

        filtered_df = self.df_sim[final_mask]

        if self.pre_clustered_df is not None:
            exclude_set = set(
                zip(
                    self.pre_clustered_df[self.pre_clustered_df["match"] == False]["id1"],
                    self.pre_clustered_df[self.pre_clustered_df["match"] == False]["id2"],
                )
            )
            reverse_exclude_set = set((y, x) for x, y in exclude_set)
            exclude_set.update(reverse_exclude_set)

            include_set = set(
                zip(
                    self.pre_clustered_df[self.pre_clustered_df["match"] == True]["id1"],
                    self.pre_clustered_df[self.pre_clustered_df["match"] == True]["id2"],
                )
            )
            reverse_include_set = set((y, x) for x, y in include_set)
            include_set.update(reverse_include_set)
        else:
            exclude_set = set()
            include_set = set()

        idx1 = filtered_df["idx1"].astype(int).values
        idx2 = filtered_df["idx2"].astype(int).values
        edges = list(zip(idx1, idx2))

        # Create a hash map (dictionary) for fast ID lookup
        id_map = {}
        if self.df2 is not None:
            combined_df = pd.concat([self.df, self.df2], ignore_index=True)
        else:
            combined_df = self.df
        
        id_map.update(combined_df[self.id_column].to_dict())

        for edge in tqdm(edges, desc="Adding edges to the graph"):
            if edge[0] is None or edge[1] is None:
                print(f"Invalid edge found: {edge}")
                continue
            id1 = id_map.get(edge[0])
            id2 = id_map.get(edge[1])
            if (
                id1 is not None
                and id2 is not None
                and (id1, id2) not in exclude_set
                and (id2, id1) not in exclude_set
            ):
                self.graph[edge[0]].add(edge[1])
                self.graph[edge[1]].add(edge[0])

        for id1, id2 in include_set:
            node1 = combined_df[combined_df[self.id_column].astype(str) == str(id1)].index[0]
            node2 = combined_df[combined_df[self.id_column].astype(str) == str(id2)].index[0]
            self.graph[node1].add(node2)
            self.graph[node2].add(node1)

        print("Similarity graph constructed.")
        return self.graph

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

    def cluster_data(self):
        # Check if the parameters have been initialized
        
        if self.df is None or self.column_thresholds is None or self.id_column is None:
            raise ValueError(
                "Parameters have not been initialized. Please call initialize_parameters() first."
            )

        self.create_similarity_matrices()
        self._construct_similarity_graph()
        self.clusters = self._find_clusters_from_graph(self.graph)

        # Apply clusters to df
        # self.df["cluster_label"] = self.df[self.id_column].map(self.clusters)
        # self.df = self.find_unclustered(self.df)

        if self.df2 is not None:
            # Get the last cluster label from df
            # last_cluster_label = int(self.df["cluster_label"].max()) + 1
            self.df = pd.concat([self.df, self.df2], ignore_index=True)
            # Apply clusters to df2 starting from the last cluster label from df
            # self.df2["cluster_label"] = self.df2[self.id_column].map(self.clusters)
            # self.df2 = self.find_unclustered(self.df2, start_label=last_cluster_label)
            
            # Concatenate df and df2
            # concatenated_df = pd.concat([self.df, self.df2], ignore_index=True)
        # else:
        
        self.df["cluster_label"] = self.df.index.map(self.clusters)
        # concatenated_df = self.df
        pdb.set_trace()
        self.df = self.find_unclustered(self.df)
        if self.compare:
            matched = self.df.groupby("cluster_label")["df"].nunique().reset_index()
            matched = matched.rename(columns={"df": "matched"})
            matched["matched"] = np.where(matched["matched"] == 2, True, False)
            self.df = self.df.merge(matched, on="cluster_label", how="left")

        if self.save_copies:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            clustered_csv_path = (
                f"{os.path.splitext(self.clustered_csv_path)[0]}_{timestamp}.csv"
            )
            graph_path = f"{os.path.splitext(self.graph_path)[0]}_{timestamp}.pkl"
        else:
            clustered_csv_path = self.clustered_csv_path
            graph_path = self.graph_path
        pdb.set_trace()
        self.df.to_csv(clustered_csv_path, index=False)
        with open(graph_path, "wb") as file:
            pickle.dump(self.graph, file)

        return self.df

    def return_cluster_data_logic_dataframe(self):
        print("Creating cluster data logic DataFrame...")
        # Reset index for the original DataFrame
        original_df = self.df.reset_index()

        # Use a single loop to update idx1 and idx2 mappings and cluster labels
        for column in self.column_thresholds.keys():
            index_to_col = original_df.set_index("index")[column]
            # Map the values in one go
            self.df_sim[f"{column}_idx1"] = self.df_sim["idx1"].map(index_to_col)
            self.df_sim[f"{column}_idx2"] = self.df_sim["idx2"].map(index_to_col)

        # Map cluster labels for both idx1 and idx2
        self.df_sim["cluster_label_idx1"] = self.df_sim["idx1"].map(self.clusters)
        self.df_sim["cluster_label_idx2"] = self.df_sim["idx2"].map(self.clusters)

        # Vectorized operation to create the new column
        self.df_sim["cluster_label"] = np.where(
            self.df_sim["cluster_label_idx1"] == self.df_sim["cluster_label_idx2"],
            self.df_sim["cluster_label_idx1"],
            None,
        )
        self.df_sim.drop(
            columns=["cluster_label_idx1", "cluster_label_idx2"], inplace=True
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
                        hover_text += f"<br>{col}: {self.df.loc[node, col]}"
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

    # def _create_similarity_matrix(self, group_tfidf, group_indices, column_name, threshold, similarity_method, blocking_value=None, progress_bar=True):
    #     if similarity_method == 'numeric_exact':
    #         return self._create_exact_match_matrix(group_tfidf, group_indices, column_name)

    #     # Determine the top_n based on the threshold
    #     top_n = 10  # Adjust this based on your requirement or make it a parameter

    #     # Use multiple threads to compute top-N cosine similarities
    #     n_threads = 4  # Adjust this based on your machine's capability
    #     cos_sim_sparse = sp_matmul_topn(group_tfidf, group_tfidf.T, top_n=top_n, threshold=threshold, n_threads=n_threads)

    #     coo = cos_sim_sparse.tocoo()
    #     rows, cols, values = coo.row, coo.col, coo.data

    #     group_indices = np.array(group_indices)  # Convert to NumPy array for faster indexing

    #     # Vectorized operation to filter out self-similarities
    #     mask = rows != cols
    #     filtered_rows = rows[mask]
    #     filtered_cols = cols[mask]
    #     filtered_values = values[mask]

    #     all_similarities = np.vstack((
    #         group_indices[filtered_rows],
    #         group_indices[filtered_cols],
    #         filtered_values
    #     )).T

    #     sim_df = pd.DataFrame(
    #         all_similarities,
    #         columns=[
    #             f"{column_name}_1_index",
    #             f"{column_name}_2_index",
    #             f"{column_name}_similarity",
    #         ],
    #     )
    #     return sim_df
