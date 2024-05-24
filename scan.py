import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import multiprocessing as mp
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import lil_matrix, csr_matrix
from sklearn.cluster import DBSCAN
import pdb
from scipy.sparse import lil_matrix, csr_matrix, coo_matrix
import networkx as nx
from sklearn.preprocessing import MinMaxScaler

# df_ppp = pd.read_csv('public_150k_plus_230930.csv')
# df_ppp.to_csv('100k.csv', index=False)



# Function to process each column and compute similarities
def process_column(data, column_name, threshold):
    # Create the vectorizer
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 2), norm=None)
    
    # Fit and transform the data
    X = vectorizer.fit_transform(data)
    
    # Dimensionality reduction using Truncated SVD
    svd = TruncatedSVD(n_components=100)  # Adjust n_components based on memory constraints and desired accuracy
    X_reduced = svd.fit_transform(X)
    
    # Function to compute cosine similarity for chunks and apply threshold
    def compute_cosine_similarity_chunk(start_idx, end_idx, X_reduced, threshold):
        chunk_matrix = cosine_similarity(X_reduced[start_idx:end_idx], X_reduced)
        mask = chunk_matrix >= threshold
        chunk_matrix = np.where(mask, chunk_matrix, 0)
        return start_idx, end_idx, chunk_matrix

    # Parameters
    chunk_size = 500  # Adjust the chunk size based on your memory constraints
    n_samples = X_reduced.shape[0]

    # Initialize a lil_matrix for cosine similarities
    cos_sim_sparse = lil_matrix((n_samples, n_samples), dtype=np.float32)

    # Compute cosine similarity chunk by chunk and apply the threshold
    for start_idx in tqdm(range(0, n_samples, chunk_size), desc=f"Computing cosine similarity in chunks for {column_name}"):
        end_idx = min(start_idx + chunk_size, n_samples)
        start_idx, end_idx, chunk_matrix = compute_cosine_similarity_chunk(start_idx, end_idx, X_reduced, threshold)
        cos_sim_sparse[start_idx:end_idx] = chunk_matrix
    # pdb.set_trace()
 # Convert the lil_matrix to csr_matrix for efficient arithmetic operations
    cos_sim_sparse = cos_sim_sparse.tocsr()

    # Get the non-zero indices and values
    coo = coo_matrix(cos_sim_sparse)
    rows, cols, values = coo.row, coo.col, coo.data

    # Store all similarities above the threshold
    all_similarities = []

    # Add tqdm to the loop for progress tracking
    for i, j, value in tqdm(zip(rows, cols, values), total=len(values), desc=f"Processing similarities for {column_name}"):
        if i != j:  # Exclude self-similarity
            all_similarities.append([i, j, value])
    
    # Create a DataFrame for all similarities
    sim_df = pd.DataFrame(all_similarities, columns=[f'{column_name}_1_index', f'{column_name}_2_index', f'{column_name}_similarity'])
    
    return sim_df

df_ppp = pd.read_csv('100k.csv')

df = df_ppp[0:1000].copy()
df = df[['BorrowerName', 'BorrowerAddress']]
df = df.rename(columns={'BorrowerName': 'col1',
                                  'BorrowerAddress' : 'col2'})
df = df.dropna()

df = df.reset_index(drop=True)

# df = {
#     'col1' : ['Bank of America', 'Bank of Amerigo', 'Sun Bank', 'yowza bank'],
#     'col2' : ['56-87202', 'Yabadababa', '56-87202', 'zipzang']
# }
# df = pd.DataFrame(df)

# Extract columns
data_col1 = df['col1'].tolist()
data_col2 = df['col2'].tolist()




# Process the columns
sim_df_col1 = process_column(df['col1'].tolist(), 'col1', 0.7)
sim_df_col2 = process_column(df['col2'].tolist(), 'col2', 0.8)

# Merge similarity DataFrames on indices
similarity_df = pd.merge(sim_df_col1, sim_df_col2, left_on=['col1_1_index', 'col1_2_index'], right_on=['col2_1_index', 'col2_2_index'], how='outer', suffixes=('_col1', '_col2'))

# Handle idx1 and idx2
similarity_df['idx1'] = np.where(similarity_df['col1_1_index'].isnull(), similarity_df['col2_1_index'], similarity_df['col1_1_index'])
similarity_df['idx2'] = np.where(similarity_df['col1_2_index'].isnull(), similarity_df['col2_2_index'], similarity_df['col1_2_index'])

# Drop the old index columns
similarity_df = similarity_df.drop(columns=['col1_1_index', 'col1_2_index', 'col2_1_index', 'col2_2_index'])

# Fill NaN similarities with 0
similarity_df = similarity_df.fillna(0)
# pdb.set_trace()

# Function to construct a graph from the similarity DataFrame
def construct_similarity_graph(df, threshold=0.8):
    G = nx.Graph()
    
    # Add edges based on similarities above the threshold
    for _, row in df.iterrows():
        idx1 = int(row['idx1'])
        idx2 = int(row['idx2'])
        if row['col1_similarity'] > threshold or row['col2_similarity'] > 0.9:
            G.add_edge(idx1, idx2)
    
    return G

# Function to find clusters using connected components
def find_clusters(G):
    clusters = list(nx.connected_components(G))
    cluster_map = {}
    for cluster_id, cluster in enumerate(clusters):
        for node in cluster:
            cluster_map[node] = cluster_id
    return cluster_map

pdb.set_trace()
similarity_graph = construct_similarity_graph(similarity_df, threshold=0.5)
clusters = find_clusters(similarity_graph)
pdb.set_trace()
df['cluster_label'] = df.index.map(clusters)

# pdb.set_trace()
# Reset index for the original DataFrame
original_df = df.reset_index()

# Create a mapping dictionary
index_to_col1 = original_df.set_index('index')['col1'].to_dict()

# Replace idx1 and idx2 with the corresponding col1 values
similarity_df['col1_idx1'] = similarity_df['idx1'].map(index_to_col1)
similarity_df['col1_idx2'] = similarity_df['idx2'].map(index_to_col1)


index_to_col1 = original_df.set_index('index')['col2'].to_dict()
similarity_df['col2_idx1'] = similarity_df['idx1'].map(index_to_col1)
similarity_df['col2_idx2'] = similarity_df['idx2'].map(index_to_col1)

pdb.set_trace()

def cluster_data(df, column_thresholds, eps=0.7, min_samples=1):
    combined_cos_sim = None
    # weight = 1 / len(column_thresholds)
    
    for col, threshold in column_thresholds.items():
        # Vectorize each column separately
        vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 2), norm='l2')
        X = vectorizer.fit_transform(df[col])
        
        # Compute cosine similarity matrix
        cos_sim = cosine_similarity(X)
        
        # Apply threshold to the similarity matrix
        thresholded_sim = np.where(cos_sim >= threshold, cos_sim, 0)
        
        # Combine the thresholded similarity matrices using the maximum similarity
        if combined_cos_sim is None:
            combined_cos_sim = thresholded_sim
        else:
            combined_cos_sim = np.maximum(combined_cos_sim, thresholded_sim)
    
    # Convert combined cosine similarity to a distance matrix (1 - similarity)
    distance_matrix = 1 - combined_cos_sim
    
    # Ensure all distances are non-negative (remove small negative values due to floating-point precision)
    distance_matrix = np.clip(distance_matrix, 0, None)
    
    # Use DBSCAN for clustering based on the distance matrix
    dbscan = DBSCAN(metric='precomputed', eps=eps, min_samples=min_samples)
    dbscan.fit(distance_matrix)
    
    # Assign cluster IDs
    df['cluster_id'] = dbscan.labels_
    
    return df





# # Process both columns separately
# sim_df_col1 = process_column(data_col1, 'col1', 0.7)
# sim_df_col2 = process_column(data_col2, 'col2', 0.8)

# similarity_df = pd.merge(sim_df_col1, sim_df_col2, left_on=['col1_1_index', 'col1_2_index'], right_on=['col2_1_index', 'col2_2_index'], how='outer', suffixes=('_col1', '_col2'))


# similarity_df['idx1'] = np.where(similarity_df['col1_1_index'].isnull(), similarity_df['col2_1_index'], similarity_df['col1_1_index'])
# similarity_df['idx2'] = np.where(similarity_df['col1_2_index'].isnull(), similarity_df['col2_2_index'], similarity_df['col1_2_index'])
# similarity_df = similarity_df.drop(columns=['col1_1_index', 'col1_2_index', 'col2_1_index', 'col2_2_index'])
# similarity_df = similarity_df.fillna(0)

# original_df = df.reset_index()
# index_to_col1 = original_df.set_index('index')['col1'].to_dict()

# similarity_df['idx1'] = similarity_df['idx1'].map(index_to_col1)
# similarity_df['idx2'] = similarity_df['idx2'].map(index_to_col1)
# pdb.set_trace()

# df_merged1 = pd.merge(original_df[['index', 'col1', 'col2']], similarity_df, left_on='index', right_on='idx1', how='inner')
# df_merged2 = pd.merge(original_df[['index', 'col1', 'col2']], similarity_df, left_on='index', right_on='idx2', how='inner')


# pdb.set_trace()
# df_merged1 = pd.merge(df[['col1']], sim_df_col1, left_index=True, right_on=['col1_1_index'], how='inner')
# df_merged2 = pd.merge(df_merged1[['col1']], sim_df_col1, left_index=True, right_on=['col1_2_index'], how='inner')
# df_merged = pd.merge(df_merged1, df_merged2, on=['col1_1_index', 'col1_2_index'], how='inner')
# pdb.set_trace()