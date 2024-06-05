import pandas as pd
import numpy as np
np.seterr(divide='ignore', invalid='ignore') # need to fix this later
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import multiprocessing as mp
from sklearn.decomposition import TruncatedSVD
import pdb
from scipy.sparse import lil_matrix, coo_matrix
import networkx as nx
import plotly.graph_objects as go
# import plotly.io as pio
from IPython.display import display
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import re
import random
from collections import deque, Counter, defaultdict
from sparse_dot_topn import sp_matmul_topn


class Entity_Fusion:
    
    def __init__(self, df, column_thresholds, id_column=None, conditional='OR', pre_clustered_df=None):
        self.df = df.reset_index(drop=True)
        self.column_thresholds = column_thresholds
        self.id_column = id_column if id_column else 'id'
        self.conditional = conditional
        self.pre_clustered_df = pre_clustered_df
        self.df_sim = None
        self.graph = None
        self.clusters = None
        # self.tfidf_scores = None
        self.stopwords = set(ENGLISH_STOP_WORDS)
        if self.id_column not in self.df.columns:
            self.df[self.id_column] = range(1, len(self.df) + 1)

    def _find_common_prefixes_and_postfixes(self, data, min_length=2):
        threshold = 5
        all_words = [word for text in data for word in text.split()]
        word_counts = Counter(all_words)
        common_affixes = [word for word, count in word_counts.items() if count >= threshold and len(word) >= min_length]
        return common_affixes#common_prefixes, common_postfixes

    def find_unclustered(self):
        # Find the maximum existing cluster label
        max_label = int(self.df['cluster_label'].max() if self.df['cluster_label'].max() is not None else -1)
        # Assign new unique labels to unclustered fields
        unclustered_mask = self.df['cluster_label'].isnull()
        num_unclustered = int(unclustered_mask.sum())
        self.df.loc[unclustered_mask, 'cluster_label'] = range(max_label + 1, max_label + 1 + num_unclustered)  # Assign new labels

    def _create_similarity_matrix(self, group_tfidf, group_indices, column_name, threshold, blocking_value=None, progress_bar=True):
        if group_tfidf.shape[0] > 5_000:  # Turn progress bar on since data is large
            progress_bar = True

        def compute_cosine_similarity_chunk(start_idx, end_idx, X_reduced, threshold):
            chunk_matrix = cosine_similarity(X_reduced[start_idx:end_idx], X_reduced)
            mask = chunk_matrix >= threshold
            chunk_matrix = np.where(mask, chunk_matrix, 0)
            return start_idx, end_idx, chunk_matrix

        # def compute_cosine_similarity_chunk(start_idx, end_idx, group_tfidf, threshold, top_n=5, n_threads=2):
        #     X_chunk = group_tfidf[start_idx:end_idx]
        #     chunk_matrix = sp_matmul_topn(X_chunk, group_tfidf.T, top_n=top_n, threshold=threshold, n_threads=n_threads)
            
        #     return start_idx, end_idx, chunk_matrix

        chunk_size = 2_000
        n_samples = group_tfidf.shape[0]
        cos_sim_sparse = lil_matrix((n_samples, n_samples), dtype=np.float32)
        if blocking_value:
            cos_sim_desc = f"Computing cosine similarity in chunks for {column_name} (Blocking: {blocking_value})"
        else:
            cos_sim_desc = f"Computing cosine similarity in chunks for {column_name}"
        loop_range = tqdm(range(0, n_samples, chunk_size), desc=cos_sim_desc, leave=False) if progress_bar else range(0, n_samples, chunk_size)

        for start_idx in loop_range:
            end_idx = min(start_idx + chunk_size, n_samples)
            start_idx, end_idx, chunk_matrix = compute_cosine_similarity_chunk(start_idx, end_idx, group_tfidf, threshold)
            cos_sim_sparse[start_idx:end_idx] = chunk_matrix

        cos_sim_sparse = cos_sim_sparse.tocsr()
        coo = coo_matrix(cos_sim_sparse)
        rows, cols, values = coo.row, coo.col, coo.data

        all_similarities = []
        for i, j, value in zip(rows, cols, values):
            if i != j:
                all_similarities.append([group_indices[i], group_indices[j], value])

        sim_df = pd.DataFrame(
            all_similarities,
            columns=[
                f"{column_name}_1_index",
                f"{column_name}_2_index",
                f"{column_name}_similarity",
            ],
        )
        return sim_df
    
    def create_similarity_matrices(self):
        processed_dfs = []
        for column, params in self.column_thresholds.items():
            # df = self.df[self.df[column].notnull()].reset_index()
            self.df.fillna('None', inplace=True)
            df = self.df.copy()
            similarity_method = params.get('similarity_method', 'tfidf')
            data = df[column].tolist()
            if similarity_method == 'numeric':
                vectorizer = TfidfVectorizer(tokenizer=lambda x: re.findall(r'\d+', x), preprocessor=None, lowercase=False, stop_words='english')
            elif similarity_method == 'tfidf':
                vectorizer = TfidfVectorizer(preprocessor=None, lowercase=False, ngram_range=(2, 3), norm='l2', smooth_idf=True, use_idf=True, stop_words='english')
            X_tfidf = vectorizer.fit_transform(data)
            
            def group_dataframe(df, params):
                blocking_criteria = params.get('blocking_criteria', None)
                
                if blocking_criteria is not None:
                    grouped_data = [df]

                    for criterion in blocking_criteria:
                        new_groups = []
                        for group in grouped_data:
                            if criterion == 'first_letter':
                                new_groups.extend(list(group.groupby(group[params['column']].str[0])))
                            elif criterion == 'blocking_column':
                                blocking_columns = params.get('blocking_column')
                                if isinstance(blocking_columns, list):
                                    new_groups.extend(list(group.groupby([group[col] for col in blocking_columns])))
                                else:
                                    new_groups.extend(list(group.groupby(group[blocking_columns])))
                            else:
                                raise ValueError(f"Unsupported criterion: {criterion}")
                        
                        grouped_data = [grp for _, grp in new_groups if len(grp) > 1]  # group size must be greater than 1
                    
                    # Return grouped_data as a list of DataFrames and their respective group names
                    return [(group_name, group) for group_name, group in new_groups]
                else:
                    return [(None, df)]

                
            grouped_data = group_dataframe(df, params)
            grouped_processed_dfs = pd.DataFrame(columns=['idx1', 'idx2', f"{column}_similarity"])
            
            for group_name, group in tqdm(grouped_data, desc=f"Processing groups for {column}"):
                group_indices = group.index.tolist()
                group_tfidf = X_tfidf[group_indices, :]
                grouped_processed_df = self._create_similarity_matrix(group_tfidf, group_indices, column, params['threshold'], blocking_value=group_name, progress_bar=False)
                if not grouped_processed_df.empty:
                    grouped_processed_df.rename(columns={
                        f"{column}_1_index": "idx1",
                        f"{column}_2_index": "idx2",
                        f"{column}_similarity": f"{column}_similarity"
                    }, inplace=True)
                    grouped_processed_df = grouped_processed_df[['idx1', 'idx2', f"{column}_similarity"]]
                    grouped_processed_dfs = pd.concat([grouped_processed_dfs, grouped_processed_df], ignore_index=True)
                else:
                    continue

            processed_dfs.append(grouped_processed_dfs)
        
        # Incremental merge with debugging
        def merge_dataframes(left_df, right_df):
            print(f"Merging dataframes: {left_df.shape} with {right_df.shape}")
            return pd.merge(
                left_df,
                right_df,
                on=["idx1", "idx2"],
                how="outer"
            )
        
        if not processed_dfs:
            raise ValueError("No processed DataFrames to merge.")
        
        # Initialize merged DataFrame
        df_sim = processed_dfs[0]
        for i in range(1, len(processed_dfs)):
            df_sim = merge_dataframes(df_sim, processed_dfs[i])
            print(f"Merged DataFrame {i}: {df_sim.shape}")
        df_sim = df_sim.fillna(0)
        self.df_sim = df_sim
        return df_sim

    def _construct_similarity_graph(self):
        print('Computing similarity graph...')
        self.graph = defaultdict(set)

        # Create boolean masks for the conditions
        masks = []
        for col, params in self.column_thresholds.items():
            print(f"Processing column: {col}")
            masks.append(self.df_sim[f"{col}_similarity"] >= params['threshold'])
        
        if self.conditional == 'AND':
            final_mask = np.logical_and.reduce(masks)
        else:  # self.conditional == 'OR'
            final_mask = np.logical_or.reduce(masks)

        print(f"Final mask: {final_mask}")

        # Use the final mask to filter the DataFrame
        filtered_df = self.df_sim[final_mask]
        print(f"Filtered DataFrame: {filtered_df.shape}")

        # Convert pre-clustered DataFrame to sets for quick lookup
        if self.pre_clustered_df is not None:
            exclude_set = set(zip(self.pre_clustered_df[self.pre_clustered_df['match'] == False]['id1'],
                                self.pre_clustered_df[self.pre_clustered_df['match'] == False]['id2']))
            reverse_exclude_set = set((y, x) for x, y in exclude_set)  # Create reverse pairs
            exclude_set.update(reverse_exclude_set)
            print(f"Exclude set: {exclude_set}")
            
            include_set = set(zip(self.pre_clustered_df[self.pre_clustered_df['match'] == True]['id1'],
                                self.pre_clustered_df[self.pre_clustered_df['match'] == True]['id2']))
            reverse_include_set = set((y, x) for x, y in include_set)  # Create reverse pairs
            include_set.update(reverse_include_set)
            print(f"Include set: {include_set}")
        else:
            exclude_set = set()
            include_set = set()

        # Extract edges using vectorized operations
        idx1 = filtered_df["idx1"].astype(int)
        idx2 = filtered_df["idx2"].astype(int)
        edges = list(zip(idx1, idx2))

        # Add edges to the graph with a progress bar
        for edge in tqdm(edges, desc="Adding edges to the graph"):
            if edge[0] is None or edge[1] is None:
                print(f"Invalid edge found: {edge}")
                continue

            id1 = self.df.loc[edge[0], self.id_column]
            id2 = self.df.loc[edge[1], self.id_column]
            if (id1, id2) not in exclude_set:
                self.graph[edge[0]].add(edge[1])
                self.graph[edge[1]].add(edge[0])
        
        # Add edges from the include set
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
        self.create_similarity_matrices()
        self._construct_similarity_graph()
        self.clusters = self._find_clusters_from_graph(self.graph)
        self.df["cluster_label"] = self.df.index.map(self.clusters)
        self.find_unclustered()
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
        
        
        
    # def update_clusters_with_post_clustered(self, df_post_clustered):
    #     # Define key columns (all except the last column which is assumed to be 'cluster_label')
    #     key_columns = df_post_clustered.columns[:-1].tolist()
        
    #     # Create a dictionary mapping the key column values to the cluster_label
    #     # Use tuples of key columns for creating the mapping
    #     post_clustered_map = df_post_clustered.set_index(key_columns)['cluster_label'].to_dict()
        
    #     # Function to create a tuple of key column values for each row
    #     def create_key_tuple(row):
    #         return tuple(row[key_columns])
        
    #     # Apply the function to create a tuple key for each row in the original DataFrame
    #     self.df['key_tuple'] = self.df.apply(create_key_tuple, axis=1)
        
    #     # Update the cluster labels in the original DataFrame based on the post-clustered DataFrame
    #     self.df['cluster_label'] = self.df['key_tuple'].map(post_clustered_map).combine_first(self.df['cluster_label'])
        
    #     # Clean up the key tuple column
    #     self.df.drop(columns=['key_tuple'], inplace=True)
    
    
    
    # def _custom_tokenizer(self, text):
    #     words = self._split_string_with_spaces(text) 
    #     total_words = len(words)
        
    #     # # Use a more gradual dropoff for shorter strings
    #     # base = np.log(total_words + 1)
        
    #     # # Calculate initial weights using logarithm and base
    #     # initial_weights = [1 / (np.log(i + 1) + base) ** 2 for i in range(1, total_words + 1)]
        
    #     # # Normalize weights so that the first weight is 1
    #     # first_weight = initial_weights[0]
    #     # word_weights = [weight / first_weight for weight in initial_weights]
    #     word_weights = [1 for i in range(1, total_words + 1)]

    #     weighted_tokens = []
    #     for word, weight in zip(words, word_weights):
    #         # Remove stopwords... 
    #         # if word.lower().strip() in self.stopwords:
    #         #     continue
    #         # Reduce weight for common postfixes
    #         if word.strip() in self.common_affixes:
    #             if word == words[-1]:
    #                 weight *= 0.5
    #             if word == words[0]:
    #                 weight *= 0.75
                    
    #         # Generate 2-word and 3-word tokens including spaces
    #         if len(word) > 2:  # Ensure the word length is valid for bi-gram generation
    #             tokens = [word[i:i+3] for i in range(len(word) - 2)]
    #             for token in tokens:
    #                 weighted_tokens.extend([token] * int(weight * 10))# * 100))  # Adjust weight scaling as needed
    #     return weighted_tokens
    
        # def _split_string_with_spaces(self, input_string):
        # """
        # Split a string and add spaces before and after words that are in the middle.
        
        # Parameters:
        # input_string (str): The input string to split.
        
        # Returns:
        # list: A list of words with added spaces before and after for middle words.
        # """
        # # Split the string by spaces
        # words = input_string.split()
        
        # # Process each word to add spaces
        # result = []
        # for i, word in enumerate(words):
        #     if i == 0:
        #         result.append(word + ' ')
        #     elif i == len(words) - 1:
        #         result.append(' ' + word)
        #     else:
        #         result.append(' ' + word + ' ')
        # return result
