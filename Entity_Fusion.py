import pandas as pd
import numpy as np
np.seterr(divide='ignore', invalid='ignore') # need to fix this later
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
# from sklearn.decomposition import TruncatedSVD
import pdb
from sparse_dot_topn import sp_matmul_topn
from scipy.sparse import lil_matrix, coo_matrix
import networkx as nx
import plotly.graph_objects as go
# import plotly.io as pio
from IPython.display import display
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import re
# import random
from collections import deque, Counter, defaultdict
import pickle
import os
import datetime


class EntityFusion:
    def __init__(self):
        self.df = None
        self.column_thresholds = None
        self.id_column = None
        self.conditional = None
        self.pre_clustered_df = None
        self.df_sim = None
        self.graph = None
        self.clusters = None
        self.clustered_csv_path = None
        self.graph_path = None
        self.save_copies = None
        self.stopwords = set(ENGLISH_STOP_WORDS)
        self.compare = False
        
    def initialize_parameters(self, 
                 df, 
                 id_column,
                 column_thresholds, 
                 df2=None, 
                 conditional='OR', 
                 pre_clustered_df=None, 
                 clustered_csv_path='cluster_files/clustered_data.csv', 
                 graph_path='cluster_files/graph.pkl',
                 save_copies=True):
        # Ensure ID column is specified and unique
        if id_column not in df.columns:
            raise ValueError(f"The ID column '{id_column}' is not present in the dataframe.")
        
        if df2 is not None:
            # Need to add warning that the 'existing data' will be treated as dataframe 1 if it's included
            self.compare = True
            df['df'] = 1
            df2['df'] = 2
            if id_column not in df2.columns:
                raise ValueError(f"The ID column '{id_column}' is not present in the second dataframe.")
            
            df = pd.concat([df, df2], ignore_index=True)
        else:
            self.compare = False
            df = df.copy()
            
         # Check for unique IDs in the first dataframe
        if not df[id_column].is_unique:
            duplicated_ids = df[id_column][df[id_column].duplicated()].unique()
            raise ValueError(f"The ID column '{id_column}' must contain unique values. Duplicated IDs: {duplicated_ids}")
        
        self.column_thresholds = column_thresholds
        self.id_column = id_column if id_column else 'id'
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
            with open(self.graph_path, 'rb') as file:
                self.graph = pickle.load(file)
            
            # Verify that IDs do not conflict
            conflicting_ids = set(df[self.id_column]).intersection(set(existing_data[self.id_column]))
            if conflicting_ids:
                raise ValueError(f"ID conflict detected between new data and existing clustered data. Conflicting IDs: {conflicting_ids}")
            
            self.df = pd.concat([existing_data, df], ignore_index=True)
        else:
            self.df = df.reset_index(drop=True)
                

    def _find_common_prefixes_and_postfixes(self, data, min_length=2):
        threshold = 5
        all_words = [word for text in data for word in text.split()]
        word_counts = Counter(all_words)
        common_affixes = [word for word, count in word_counts.items() if count >= threshold and len(word) >= min_length]
        return common_affixes

    def find_unclustered(self):
        max_label = int(self.df['cluster_label'].max() if self.df['cluster_label'].max() is not None else -1)
        unclustered_mask = self.df['cluster_label'].isnull()
        num_unclustered = int(unclustered_mask.sum())
        self.df.loc[unclustered_mask, 'cluster_label'] = range(max_label + 1, max_label + 1 + num_unclustered)

    def _create_exact_match_matrix(self, data, group_indices, column_name):
        matches = []
        value_to_indices = defaultdict(list)
        for idx, value in enumerate(data):
            value_to_indices[value].append(group_indices[idx])
        for indices in tqdm(value_to_indices.values(), desc=f"Processing exact matches for {column_name}"):
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



    def _create_similarity_matrix(self, group_tfidf, group_indices, column_name, threshold, similarity_method, blocking_value=None, progress_bar=True):
        if similarity_method == 'numeric_exact':
            return self._create_exact_match_matrix(group_tfidf, group_indices, column_name)

        # Determine the top_n based on the threshold
        top_n = 10  # Adjust this based on your requirement or make it a parameter

        # Use multiple threads to compute top-N cosine similarities
        n_threads = 4  # Adjust this based on your machine's capability
        cos_sim_sparse = sp_matmul_topn(group_tfidf, group_tfidf.T, top_n=top_n, threshold=threshold, n_threads=n_threads)

        coo = cos_sim_sparse.tocoo()
        rows, cols, values = coo.row, coo.col, coo.data

        group_indices = np.array(group_indices)  # Convert to NumPy array for faster indexing

        # Vectorized operation to filter out self-similarities
        mask = rows != cols
        filtered_rows = rows[mask]
        filtered_cols = cols[mask]
        filtered_values = values[mask]

        all_similarities = np.vstack((
            group_indices[filtered_rows],
            group_indices[filtered_cols],
            filtered_values
        )).T

        sim_df = pd.DataFrame(
            all_similarities,
            columns=[
                f"{column_name}_1_index",
                f"{column_name}_2_index",
                f"{column_name}_similarity",
            ],
        )
        return sim_df

    def process_group(self, group_name, group, column, X_tfidf, similarity_method, threshold, blocking_value):
        group = group[group[column].notnull()]
        group = group[group[column].str.contains(r'\d')]
        group = group[group[column] != '']
        group = group[group[column] != 'nan']
        group = group[group[column] != 'None']

        group_indices = group.index.tolist()

        if similarity_method == 'tfidf' or similarity_method == 'numeric':
            group_tfidf = X_tfidf[group_indices, :]
        elif similarity_method == 'numeric_exact':
            group_tfidf = X_tfidf.loc[group_indices, :]
            group_tfidf = group_tfidf[column].tolist()

        grouped_processed_df = self._create_similarity_matrix(group_tfidf, group_indices, column, threshold, similarity_method, blocking_value=group_name, progress_bar=False)

        if not grouped_processed_df.empty:
            grouped_processed_df.rename(columns={
                f"{column}_1_index": "idx1",
                f"{column}_2_index": "idx2",
                f"{column}_similarity": f"{column}_similarity"
            }, inplace=True)
            grouped_processed_df = grouped_processed_df[['idx1', 'idx2', f"{column}_similarity"]]
            return grouped_processed_df
        else:
            return pd.DataFrame(columns=['idx1', 'idx2', f"{column}_similarity"])

    def merge_dataframes(self, left_df, right_df):
        return pd.merge(
            left_df,
            right_df,
            on=["idx1", "idx2"],
            how="outer"
        )

    def group_dataframe(self, df, params, column):
        blocking_criteria = params.get('blocking_criteria', None)

        if blocking_criteria is not None:
            grouped_data = [df]

            for criterion in blocking_criteria:
                new_groups = []
                for group in grouped_data:
                    if criterion == 'first_letter':
                        new_groups.extend(list(group.groupby(group[column].str[0])))
                    elif criterion == 'blocking_column':
                        blocking_columns = params.get('blocking_column')
                        if isinstance(blocking_columns, list):
                            new_groups.extend(list(group.groupby([group[col] for col in blocking_columns])))
                        else:
                            new_groups.extend(list(group.groupby(group[blocking_columns])))
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
            similarity_method = params.get('similarity_method', 'tfidf')
            
            data = df[column].tolist()
            if similarity_method == 'numeric':
                vectorizer = TfidfVectorizer(tokenizer=lambda x: re.findall(r'\d+', x), preprocessor=None, lowercase=False, stop_words='english')
                X_tfidf = vectorizer.fit_transform(data)
            elif similarity_method == 'tfidf':
                vectorizer = TfidfVectorizer(preprocessor=None, lowercase=False, ngram_range=(2, 3), norm='l2', smooth_idf=True, use_idf=True, stop_words='english')
                X_tfidf = vectorizer.fit_transform(data)
            elif similarity_method == 'numeric_exact':
                X_tfidf = df
            grouped_data = self.group_dataframe(df, params, column)
            
            grouped_processed_dfs_list = []
            for group_name, group in tqdm(grouped_data, desc=f"Processing groups for {column}"):
                result = self.process_group(group_name, group, column, X_tfidf, similarity_method, params['threshold'], group_name)
                grouped_processed_dfs_list.append(result)
            
            grouped_processed_dfs = pd.concat(grouped_processed_dfs_list, ignore_index=True)
            
            processed_dfs.append(grouped_processed_dfs)
        
        if not processed_dfs:
            raise ValueError("No processed DataFrames to merge.")
        
        # Initialize merged DataFrame
        df_sim = processed_dfs[0]
        for i in range(1, len(processed_dfs)):
            df_sim = self.merge_dataframes(df_sim, processed_dfs[i])
        pdb.set_trace()
        df_sim = df_sim.fillna(0)
        self.df_sim = df_sim
        
        return df_sim


    def _construct_similarity_graph(self, multiprocessing=True):
        print('Computing similarity graph...')
        
        if self.graph is None:
            self.graph = defaultdict(set)

        masks = []
        for col, params in self.column_thresholds.items():
            masks.append(self.df_sim[f"{col}_similarity"] >= params['threshold'])

        if self.conditional == 'AND':
            final_mask = np.logical_and.reduce(masks)
        else:
            final_mask = np.logical_or.reduce(masks)

        filtered_df = self.df_sim[final_mask]

        if self.pre_clustered_df is not None:
            exclude_set = set(zip(self.pre_clustered_df[self.pre_clustered_df['match'] == False]['id1'],
                                self.pre_clustered_df[self.pre_clustered_df['match'] == False]['id2']))
            reverse_exclude_set = set((y, x) for x, y in exclude_set)
            exclude_set.update(reverse_exclude_set)

            include_set = set(zip(self.pre_clustered_df[self.pre_clustered_df['match'] == True]['id1'],
                                self.pre_clustered_df[self.pre_clustered_df['match'] == True]['id2']))
            reverse_include_set = set((y, x) for x, y in include_set)
            include_set.update(reverse_include_set)
        else:
            exclude_set = set()
            include_set = set()

        idx1 = filtered_df["idx1"].astype(int).values
        idx2 = filtered_df["idx2"].astype(int).values
        edges = list(zip(idx1, idx2))

        # Create a hash map (dictionary) for fast ID lookup
        id_map = self.df[self.id_column].to_dict()

        for edge in tqdm(edges, desc="Adding edges to the graph"):
            if edge[0] is None or edge[1] is None:
                print(f"Invalid edge found: {edge}")
                continue
            id1 = id_map.get(edge[0])
            id2 = id_map.get(edge[1])
            if id1 is not None and id2 is not None and (id1, id2) not in exclude_set and (id2, id1) not in exclude_set:
                self.graph[edge[0]].add(edge[1])
                self.graph[edge[1]].add(edge[0])

        for id1, id2 in include_set:
            node1 = self.df[self.df[self.id_column].astype(str) == str(id1)].index[0]
            node2 = self.df[self.df[self.id_column].astype(str) == str(id2)].index[0]
            self.graph[node1].add(node2)
            self.graph[node2].add(node1)

        print('Similarity graph constructed.')
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
            raise ValueError("Parameters have not been initialized. Please call initialize_parameters() first.")

        self.create_similarity_matrices()
        self._construct_similarity_graph()
        self.clusters = self._find_clusters_from_graph(self.graph)
        self.df["cluster_label"] = self.df.index.map(self.clusters)
        self.find_unclustered()
        if self.compare:
            matched = self.df.groupby('cluster_label')['df'].nunique().reset_index()
            matched = matched.rename(columns={'df': 'matched'})
            matched['matched'] = np.where(matched['matched'] == 2, True, False)
            self.df = self.df.merge(matched, on='cluster_label', how='left')

        if self.save_copies:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            clustered_csv_path = f"{os.path.splitext(self.clustered_csv_path)[0]}_{timestamp}.csv"
            graph_path = f"{os.path.splitext(self.graph_path)[0]}_{timestamp}.pkl"
        else:
            clustered_csv_path = self.clustered_csv_path
            graph_path = self.graph_path

        self.df.to_csv(clustered_csv_path, index=False)
        with open(graph_path, 'wb') as file:
            pickle.dump(self.graph, file)
        return self.df


    
    def return_cluster_data_logic_dataframe(self):
        print('Creating cluster data logic DataFrame...')
        # Reset index for the original DataFrame
        original_df = self.df.reset_index()

        # Use a single loop to update idx1 and idx2 mappings and cluster labels
        for column in self.column_thresholds.keys():
            index_to_col = original_df.set_index("index")[column]
            # Map the values in one go
            self.df_sim[f"{column}_idx1"] = self.df_sim["idx1"].map(index_to_col)
            self.df_sim[f"{column}_idx2"] = self.df_sim["idx2"].map(index_to_col)
            
        # Map cluster labels for both idx1 and idx2
        self.df_sim['cluster_label_idx1'] = self.df_sim['idx1'].map(self.clusters)
        self.df_sim['cluster_label_idx2'] = self.df_sim['idx2'].map(self.clusters)

        # Vectorized operation to create the new column
        self.df_sim['cluster_label'] = np.where(
            self.df_sim['cluster_label_idx1'] == self.df_sim['cluster_label_idx2'],
            self.df_sim['cluster_label_idx1'],
            None
        )
        self.df_sim.drop(columns=['cluster_label_idx1', 'cluster_label_idx2'], inplace=True)
        print('Cluster data logic DataFrame created.')
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
        nodes_in_cluster = [node for node, cluster in self.clusters.items() if cluster == cluster_id]
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
            x=edge_x, y=edge_y,
            line=dict(width=1, color='#888'),
            hoverinfo='none',
            mode='lines'
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
            x=node_x, y=node_y,
            mode='markers+text',
            text=[str(node) for node in subgraph.nodes()],
            textposition="bottom center",
            hovertext=hover_texts,
            hoverinfo='text',
            marker=dict(
                showscale=True,
                colorscale='YlGnBu',
                size=10,
                colorbar=dict(
                    thickness=15,
                    title='Node Connections',
                    xanchor='left',
                    titleside='right'
                )
            )
        )
        
        fig = go.Figure(data=[edge_trace, node_trace],
                        layout=go.Layout(
                            title=f'<br>Cluster {cluster_id} Graph',
                            titlefont_size=16,
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=20, l=5, r=5, t=40),
                            annotations=[dict(
                                text="Interactive graph where nodes can be moved",
                                showarrow=False,
                                xref="paper", yref="paper",
                                x=0.005, y=-0.002)],
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                        )
        display(fig)
        
    # def _create_similarity_matrix(self, group_tfidf, group_indices, column_name, threshold, similarity_method, blocking_value=None, progress_bar=True):
    #     if similarity_method == 'numeric_exact':
    #         return self._create_exact_match_matrix(group_tfidf, group_indices, column_name)
    #     if group_tfidf.shape[0] > 5_000:
    #         progress_bar = True

    #     def compute_cosine_similarity_chunk(start_idx, end_idx, group_tfidf, threshold):
    #         chunk_matrix = cosine_similarity(group_tfidf[start_idx:end_idx], group_tfidf)
    #         mask = chunk_matrix >= threshold
    #         chunk_matrix = np.where(mask, chunk_matrix, 0)
    #         return start_idx, end_idx, chunk_matrix

    #     chunk_size = 2_000
    #     n_samples = group_tfidf.shape[0]
    #     cos_sim_sparse = lil_matrix((n_samples, n_samples), dtype=np.float32)
    #     if blocking_value:
    #         cos_sim_desc = f"Computing cosine similarity in chunks for {column_name} (Blocking: {blocking_value})"
    #     else:
    #         cos_sim_desc = f"Computing cosine similarity in chunks for {column_name}"
    #     loop_range = tqdm(range(0, n_samples, chunk_size), desc=cos_sim_desc, leave=False) if progress_bar else range(0, n_samples, chunk_size)

    #     for start_idx in loop_range:
    #         end_idx = min(start_idx + chunk_size, n_samples)
    #         start_idx, end_idx, chunk_matrix = compute_cosine_similarity_chunk(start_idx, end_idx, group_tfidf, threshold)
    #         cos_sim_sparse[start_idx:end_idx] = chunk_matrix

    #     cos_sim_sparse = cos_sim_sparse.tocsr()
    #     coo = coo_matrix(cos_sim_sparse)
    #     rows, cols, values = coo.row, coo.col, coo.data

    #     all_similarities = []
    #     for i, j, value in zip(rows, cols, values):
    #         if i != j:
    #             all_similarities.append([group_indices[i], group_indices[j], value])

    #     sim_df = pd.DataFrame(
    #         all_similarities,
    #         columns=[
    #             f"{column_name}_1_index",
    #             f"{column_name}_2_index",
    #             f"{column_name}_similarity",
    #         ],
    #     )
    #     return sim_df