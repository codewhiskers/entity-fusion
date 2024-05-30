import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import multiprocessing as mp
from sklearn.decomposition import TruncatedSVD
import pdb
from scipy.sparse import lil_matrix, coo_matrix
import networkx as nx
from functools import reduce
import plotly.graph_objects as go
from collections import Counter
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import re


class Entity_Fusion:
    
    def __init__(self, df, column_thresholds, post_clustered_df=None):
        self.df = df
        self.column_thresholds = column_thresholds
        self.post_clustered_df = post_clustered_df
        self.df_sim = None
        self.graph = None
        self.clusters = None
        self.stopwords = set(ENGLISH_STOP_WORDS)


    def find_common_prefixes_and_postfixes(self, data, min_length=2):
        threshold = 5
        all_words = [word for text in data for word in text.split()]
        word_counts = Counter(all_words)
        # common_prefixes = [word for word, count in word_counts.items() if count >= threshold and len(word) >= min_length]
        # pdb.set_trace()
        common_affixes = [word for word, count in word_counts.items() if count >= threshold and len(word) >= min_length]
        return common_affixes#common_prefixes, common_postfixes

    def split_string_with_spaces(self, input_string):
        """
        Split a string and add spaces before and after words that are in the middle.
        
        Parameters:
        input_string (str): The input string to split.
        
        Returns:
        list: A list of words with added spaces before and after for middle words.
        """
        # Split the string by spaces
        words = input_string.split()
        
        # Process each word to add spaces
        result = []
        for i, word in enumerate(words):
            if i == 0:
                result.append(word + ' ')
            elif i == len(words) - 1:
                result.append(' ' + word)
            else:
                result.append(' ' + word + ' ')
        # pdb.set_trace()
        return result

    def _create_tfidf_matrix(self, data):
        vectorizer = TfidfVectorizer(
            tokenizer=lambda text: self.split_string_with_spaces(text),
            preprocessor=None,
            lowercase=False,
            sublinear_tf=True,
            norm=None
        )
        # X_tfidf = vectorizer.fit_transform(data)
        # feature_names = vectorizer.get_feature_names_out()
        # tfidf_scores = dict(zip(feature_names, X_tfidf.mean(axis=0).tolist()[0]))
        # scaling_factor = 5  # Choose a factor to scale up the scores
        # tfidf_scores = {term: score * scaling_factor for term, score in tfidf_scores.items()}
        # Print term frequencies
        # term_frequencies = X_tfidf.sum(axis=0).A1
        # term_frequencies_dict = dict(zip(feature_names, term_frequencies))

        # # Print document frequencies
        # document_frequencies = (X_tfidf > 0).sum(axis=0).A1
        # document_frequencies_dict = dict(zip(feature_names, document_frequencies))
        
        vectorizer.fit(data)

        # Get feature names
        feature_names = vectorizer.get_feature_names_out()

        # Get IDF values
        idf_values = vectorizer.idf_
        idf_scores = dict(zip(feature_names, idf_values))
        # scaling_factor = 5  # Choose a factor to scale up the scores
        idf_scores = {term: score for term, score in idf_scores.items()}
        return idf_scores

    def custom_tokenizer(self, text, common_affixes=None, tfidf_scores=None):
        # pdb.set_trace()
        # words = text.split()
        words = self.split_string_with_spaces(text) 
        total_words = len(words)
        
        # Use a more gradual dropoff for shorter strings
        # base = np.log(total_words + 1)  # Base for logarithmic scaling
        # base = np.log(total_words + 1)  # Adjust the base to control the steepness
        # word_weights = [1 for i in range(total_words)]
        base = np.log(total_words + 1)
        word_weights = [1 / (np.log(i + 1) + base) for i in range(1, total_words + 1)]

        weighted_tokens = []
        for word, weight in zip(words, word_weights):
            # Remove stopwords... 
            if tfidf_scores is not None:
                tfidf_weight = tfidf_scores.get(word, 1)
                weight *= tfidf_weight

            # Reduce weight for common postfixes
            if word.strip() in common_affixes:
                if word == words[-1]:
                    weight *= 0.5
                if word == words[0]:
                    weight *= 0.5
                    
            # Generate 2-word and 3-word tokens including spaces
            if len(word) > 1:  # Ensure the word length is valid for bi-gram generation
                tokens = [word[i:i+2] for i in range(len(word) - 1)]
                for token in tokens:
                    weighted_tokens.extend([token] * int(weight))# * 100))  # Adjust weight scaling as needed
                    
        return weighted_tokens

    def _create_similarity_matrix(self, df, column_name, threshold):
        data = df[column_name].tolist()
        common_affixes = self.find_common_prefixes_and_postfixes(data)

        tfidf_scores = self._create_tfidf_matrix(data)

        # Use CountVectorizer to create a term-frequency matrix with the custom tokenizer
        vectorizer = CountVectorizer(tokenizer=lambda text: self.custom_tokenizer(text, 
                                                                                  common_affixes,
                                                                                  tfidf_scores), 
                                     preprocessor=None, lowercase=False)
        X_counts = vectorizer.fit_transform(data)

        # Transform the term-frequency matrix to a tf-idf representation
        tfidf_transformer = TfidfTransformer(norm='l2', smooth_idf=True)
        X_tfidf = tfidf_transformer.fit_transform(X_counts)

        n_features = X_tfidf.shape[1]
        n_features = 1000 if n_features > 2000 else n_features
        svd = TruncatedSVD(n_components=n_features)
        X_reduced = svd.fit_transform(X_tfidf)

        def compute_cosine_similarity_chunk(start_idx, end_idx, X_reduced, threshold):
            chunk_matrix = cosine_similarity(X_reduced[start_idx:end_idx], X_reduced)
            mask = chunk_matrix >= threshold
            chunk_matrix = np.where(mask, chunk_matrix, 0)
            return start_idx, end_idx, chunk_matrix

        chunk_size = 500
        n_samples = X_reduced.shape[0]
        cos_sim_sparse = lil_matrix((n_samples, n_samples), dtype=np.float32)
        cos_sim_desc = f"Computing cosine similarity in chunks for {column_name}"
        for start_idx in tqdm(range(0, n_samples, chunk_size), desc=cos_sim_desc):
            end_idx = min(start_idx + chunk_size, n_samples)
            start_idx, end_idx, chunk_matrix = compute_cosine_similarity_chunk(start_idx, end_idx, X_reduced, threshold)
            cos_sim_sparse[start_idx:end_idx] = chunk_matrix
        cos_sim_sparse = cos_sim_sparse.tocsr()

        coo = coo_matrix(cos_sim_sparse)
        rows, cols, values = coo.row, coo.col, coo.data

        process_sim_desc = f"Processing similarities for {column_name}"
        all_similarities = []
        for i, j, value in tqdm(
            zip(rows, cols, values),
            total=len(values),
            desc=process_sim_desc,
        ):
            if i != j:
                all_similarities.append([i, j, value])

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
        for column, threshold in self.column_thresholds.items():
            processed_df = self._create_similarity_matrix(self.df, column, threshold)
            processed_dfs.append(processed_df)
            
        def merge_dataframes(left_df, right_df, left_col, right_col):
            return pd.merge(
                left_df,
                right_df,
                left_on=[f"{left_col}_1_index", f"{left_col}_2_index"],
                right_on=[f"{right_col}_1_index", f"{right_col}_2_index"],
                how="outer",
                suffixes=(f"_{left_col}", f"_{right_col}")
            )
        # pdb.set_trace()
        column_names = list(self.column_thresholds.keys())
        df_sim = reduce(lambda left, right: merge_dataframes(left, right, column_names[0], column_names[1]), processed_dfs)
        for i in range(1, 3):
            columns_to_check = [x for x in df_sim.columns if f"{i}_index" in x]
            df_sim[f"idx{i}"] = df_sim[columns_to_check].bfill(axis=1).iloc[:, 0]
            df_sim.drop(columns=columns_to_check, inplace=True)
        # pdb.set_trace()
        df_sim = df_sim.fillna(0)
        self.df_sim = df_sim
        return df_sim
        
    def _construct_similarity_graph(self):
        G = nx.Graph()
        for _, row in self.df_sim.iterrows():
            idx1 = int(row["idx1"])
            idx2 = int(row["idx2"])
            condition = any(row[f"{col}_similarity"] > threshold for col, threshold in self.column_thresholds.items())
            if condition:
                G.add_edge(idx1, idx2)
        self.graph = G
        return G
    
    def _find_clusters(self):
        clusters = list(nx.connected_components(self.graph))
        cluster_map = {}
        for cluster_id, cluster in enumerate(clusters):
            for node in cluster:
                cluster_map[node] = cluster_id
        self.clusters = cluster_map
        return cluster_map
    
    def add_pre_clustered_data(self, pre_clustered_df):
        pre_clustered_map = {}
        for column in self.column_thresholds.keys():
            column_map = pre_clustered_df.set_index(column)['cluster_label'].to_dict()
            pre_clustered_map[column] = column_map

        def map_pre_clusters(row):
            for column in self.column_thresholds.keys():
                if row[column] in pre_clustered_map[column]:
                    return pre_clustered_map[column][row[column]]
            return None

        self.df['pre_cluster_label'] = self.df.apply(map_pre_clusters, axis=1)
        self.df['cluster_label'] = self.df['pre_cluster_label'].combine_first(self.df.index.map(self.clusters))
        self.df_sim['pre_cluster_label'] = self.df_sim['idx1'].map(self.df['pre_cluster_label'])
        self.df_sim['cluster_label'] = self.df_sim['pre_cluster_label'].combine_first(self.df_sim['idx1'].map(self.clusters))
    
    def add_post_clustered_data(self, df, post_clustered_df):
        post_clustered_df['cluster_label'] = post_clustered_df['cluster_label'].apply(lambda x: f"M-{x}")
        df = pd.merge(df, post_clustered_df, how='left')
        return df
    
    def update_clusters_with_post_clustered(self, df_post_clustered):
        # Define key columns (all except the last column which is assumed to be 'cluster_label')
        key_columns = df_post_clustered.columns[:-1].tolist()
        
        # Create a dictionary mapping the key column values to the cluster_label
        # Use tuples of key columns for creating the mapping
        post_clustered_map = df_post_clustered.set_index(key_columns)['cluster_label'].to_dict()
        
        # Function to create a tuple of key column values for each row
        def create_key_tuple(row):
            return tuple(row[key_columns])
        
        # Apply the function to create a tuple key for each row in the original DataFrame
        self.df['key_tuple'] = self.df.apply(create_key_tuple, axis=1)
        
        # Update the cluster labels in the original DataFrame based on the post-clustered DataFrame
        self.df['cluster_label'] = self.df['key_tuple'].map(post_clustered_map).combine_first(self.df['cluster_label'])
        
        # Clean up the key tuple column
        self.df.drop(columns=['key_tuple'], inplace=True)
        
        # Update the cluster labels in the similarity matrix
        # self.df_sim['cluster_label'] = self.df_sim['idx1'].map(self.df['cluster_label'])

        
    
    def cluster_data(self):
        self.create_similarity_matrices()
        self._construct_similarity_graph()
        self._find_clusters()
        self.df["cluster_label"] = self.df.index.map(self.clusters)
        if self.post_clustered_df is not None:
            self.update_clusters_with_post_clustered(self.post_clustered_df)
        return self.df
    
    def return_cluster_data_logic_dataframe(self):
        # Reset index for the original DataFrame
        original_df = self.df.reset_index()
        for column in self.column_thresholds.keys():
            # Create a mapping dictionary
            index_to_col = original_df.set_index("index")[column].to_dict()
            # Replace idx1 and idx2 with the corresponding column values
            self.df_sim[f"{column}_idx1"] = self.df_sim["idx1"].map(index_to_col)
            self.df_sim[f"{column}_idx2"] = self.df_sim["idx2"].map(index_to_col)  
            
        self.df_sim['cluster_label'] = self.df_sim['idx1'].map(self.clusters)
        return self.df_sim
    
    
    # Function to visualize a specific cluster interactively
    # def visualize_cluster(graph, clusters, cluster_id):
    def visualize_cluster(self, cluster_id):
        nodes_in_cluster = [node for node, cluster in self.clusters.items() if cluster == cluster_id]
        subgraph = self.graph.subgraph(nodes_in_cluster)
        
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
        for node in subgraph.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            text=[str(node) for node in subgraph.nodes()],
            textposition="bottom center",
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
                            margin=dict(b=20,l=5,r=5,t=40),
                            annotations=[ dict(
                                text="Interactive graph where nodes can be moved",
                                showarrow=False,
                                xref="paper", yref="paper",
                                x=0.005, y=-0.002 ) ],
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                        )
        fig.show()
        
        
#     def _create_similarity_matrix(self, df, column_name, threshold):
        # data = df[column_name].tolist()

        # def custom_tokenizer(text):
        #     words = text.split()
        #     word_weights = [(1 / (i + 1)) for i in range(len(words))]  # Custom word-level weighting
        #     weighted_tokens = []
        #     for word, weight in zip(words, word_weights):
        #         tokens = [word[i:i+2] for i in range(len(word) - 1)]
        #         for token in tokens:
        #             weighted_tokens.extend([token] * int(weight * 100))  # Adjust weight scaling as needed
        #     return weighted_tokens


        # # Use CountVectorizer to create a term-frequency matrix with the custom tokenizer
        # vectorizer = CountVectorizer(tokenizer=custom_tokenizer, preprocessor=None, lowercase=False)
        # X_counts = vectorizer.fit_transform(data)

        # # Transform the term-frequency matrix to a tf-idf representation
        # tfidf_transformer = TfidfTransformer(norm='l2', smooth_idf=True)
        # X_tfidf = tfidf_transformer.fit_transform(X_counts)

        # n_features = X_tfidf.shape[1]
        # n_features = 1000 if n_features > 2000 else n_features
        # svd = TruncatedSVD(n_components=n_features)
        # X_reduced = svd.fit_transform(X_tfidf)

        # def compute_cosine_similarity_chunk(start_idx, end_idx, X_reduced, threshold):
        #     chunk_matrix = cosine_similarity(X_reduced[start_idx:end_idx], X_reduced)
        #     mask = chunk_matrix >= threshold
        #     chunk_matrix = np.where(mask, chunk_matrix, 0)
        #     return start_idx, end_idx, chunk_matrix

        # chunk_size = 500
        # n_samples = X_reduced.shape[0]
        # cos_sim_sparse = lil_matrix((n_samples, n_samples), dtype=np.float32)
        # cos_sim_desc = f"Computing cosine similarity in chunks for {column_name}"
        # for start_idx in tqdm(range(0, n_samples, chunk_size), desc=cos_sim_desc):
        #     end_idx = min(start_idx + chunk_size, n_samples)
        #     start_idx, end_idx, chunk_matrix = compute_cosine_similarity_chunk(start_idx, end_idx, X_reduced, threshold)
        #     cos_sim_sparse[start_idx:end_idx] = chunk_matrix
        # cos_sim_sparse = cos_sim_sparse.tocsr()

        # coo = coo_matrix(cos_sim_sparse)
        # rows, cols, values = coo.row, coo.col, coo.data

        # process_sim_desc = f"Processing similarities for {column_name}"
        # all_similarities = []
        # for i, j, value in tqdm(
        #     zip(rows, cols, values),
        #     total=len(values),
        #     desc=process_sim_desc,
        # ):
        #     if i != j:
        #         all_similarities.append([i, j, value])

        # sim_df = pd.DataFrame(
        #     all_similarities,
        #     columns=[
        #         f"{column_name}_1_index",
        #         f"{column_name}_2_index",
        #         f"{column_name}_similarity",
        #     ],
        # )
        # return sim_df