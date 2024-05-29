import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import multiprocessing as mp
from sklearn.decomposition import TruncatedSVD
import pdb
from scipy.sparse import lil_matrix, coo_matrix
import networkx as nx
from functools import reduce
import plotly.graph_objects as go


class Entity_Fusion:
    
    def __init__(self, df, column_thresholds):
        self.df = df
        self.column_thresholds = column_thresholds
        self.df_sim = None
        self.graph = None
        self.clusters = None


    def _create_similarity_matrix(self, df, column_name, threshold):
        data = df[column_name].tolist()
        # Create the vectorizer
        vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 2), 
                                     norm='l2', max_df = 0.5)
        # Fit and transform the data
        X = vectorizer.fit_transform(data)
        n_features = X.shape[1]
        n_samples = X.shape[0]
        # n_components = n_features if n_samples <= 1000 else 1000
        print(n_features)
        n_features = 1000 if n_features > 2000 else n_features
        # Dimensionality reduction using Truncated SVD
        svd = TruncatedSVD(n_components=n_features)
        X_reduced = svd.fit_transform(X)

        # Function to compute cosine similarity for chunks and apply threshold
        def compute_cosine_similarity_chunk(start_idx, end_idx, X_reduced, threshold):
            chunk_matrix = cosine_similarity(X_reduced[start_idx:end_idx], X_reduced)
            mask = chunk_matrix >= threshold
            chunk_matrix = np.where(mask, chunk_matrix, 0)
            return start_idx, end_idx, chunk_matrix

        chunk_size = 500  # Adjust the chunk size based on your memory constraints
        n_samples = X_reduced.shape[0]
        # Initialize a lil_matrix for cosine similarities
        cos_sim_sparse = lil_matrix((n_samples, n_samples), dtype=np.float32)
        # Compute cosine similarity chunk by chunk and apply the threshold
        cos_sim_desc = f"Computing cosine similarity in chunks for {column_name}"
        for start_idx in tqdm(range(0, n_samples, chunk_size), desc=cos_sim_desc):
            end_idx = min(start_idx + chunk_size, n_samples)
            start_idx, end_idx, chunk_matrix = compute_cosine_similarity_chunk(start_idx, end_idx, X_reduced, threshold)
            cos_sim_sparse[start_idx:end_idx] = chunk_matrix
        # Convert the lil_matrix to csr_matrix for efficient arithmetic operations
        cos_sim_sparse = cos_sim_sparse.tocsr()

        # Get the non-zero indices and values
        coo = coo_matrix(cos_sim_sparse)
        rows, cols, values = coo.row, coo.col, coo.data

        process_sim_desc = f"Processing similarities for {column_name}"
        # Store all similarities above the threshold
        all_similarities = []
        # Add tqdm to the loop for progress tracking
        for i, j, value in tqdm(
            zip(rows, cols, values),
            total=len(values),
            desc=process_sim_desc,
        ):
            if i != j:  # Exclude self-similarity
                all_similarities.append([i, j, value])

        # Create a DataFrame for all similarities
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
    
    
    def cluster_data(self):
        self.create_similarity_matrices()
        self._construct_similarity_graph()
        self._find_clusters()
        self.df["cluster_label"] = self.df.index.map(self.clusters)
        # Reset index for the original DataFrame
        original_df = self.df.reset_index()
        for column in self.column_thresholds.keys():
            # Create a mapping dictionary
            index_to_col = original_df.set_index("index")[column].to_dict()
            # Replace idx1 and idx2 with the corresponding column values
            self.df_sim[f"{column}_idx1"] = self.df_sim["idx1"].map(index_to_col)
            self.df_sim[f"{column}_idx2"] = self.df_sim["idx2"].map(index_to_col)  
            
        self.df_sim['cluster_label'] = self.df_sim['idx1'].map(self.clusters)
        pdb.set_trace()
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