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

# from sklearn.preprocessing import MinMaxScaler
from Entity_Fusion import Entity_Fusion
from CompanyCleaner import CompanyCleaner
import pandas as pd


# df_ppp = pd.read_csv("public_150k_plus_230930.csv", nrows=10_000)
# df = df_ppp[0:10_000].copy()
# df = df[["BorrowerName", "BorrowerAddress", 'BorrowerCity']]
# # df = df.rename(columns={"BorrowerName": "col1", "BorrowerAddress": "col2"})
# df = df.dropna()
# df = df.reset_index(drop=True)

# pre_clustered_df = pd.DataFrame({
#     'id1': [638, 1],
#     'id2': [4023, 9035],
#     'match': [False, True]
# })

# df = pd.read_pickle("cordata8_0.pkl")
# df1 = df[0:10_000].copy()
# # df2 = df[80_000:120_000].copy()

# # pdb.set_trace()
# def subset_and_clean_data(df):
#     df = df[["corporation_name", "address_1", "city", "state", "fei_number", 'registered_agent_name']].copy()
#     df['registered_agent_name'] = df['registered_agent_name'].apply(lambda x: ' '.join(str(x).split()))
#     df["city"] = df["city"].str.upper()
#     df["state"] = df["state"].str.upper()
#     # df = CompanyCleaner(df, "corporation_name").clean_entity_names()
#     return df
# # df = df[["corporation_name", "address_1", "city", "state", "fei_number", 'registered_agent_name']]

# # df['registered_agent_name'] = df['registered_agent_name'].apply(lambda x: ' '.join(str(x).split()))
# # df["city"] = df["city"].str.upper()
# # df["state"] = df["state"].str.upper()
# # df = CompanyCleaner(df, "corporation_name").clean_entity_names()

# df1 = subset_and_clean_data(df1)
# df2 = subset_and_clean_data(df2)    

# # post_clustered_data = pd.read_csv('post_clustered_data.csv')

# EF = Entity_Fusion(
#     df1,
#     {
#         "corporation_name_CLN": {
#             "threshold": 0.5,
#             "blocking_column": ["city", 'state'],
#             "blocking_criteria": ["blocking_column", 'first_letter'],
#             "similarity_method": "tfidf",
#         },
#         # "registered_agent_name": {
#         #     "threshold": 0.9,
#         #     "block": True,
#         #     "blocking_column": "city",
#         #     "criteria": ["first_letter", "blocking_column"],
#         #     "similarity_method": "tfidf",
#         # },
#         "fei_number": {
#             "threshold": 1,
#             "similarity_method": "numeric",
#         },
#     }, 
#     df2
#     # pre_clustered_df=pre_clustered_df
# )

# df = EF.cluster_data()
# pdb.set_trace()

# if __name__ == '__main__':
df = pd.read_pickle("cordata8_0.pkl")
df1 = df[0:100_000].copy()
# df2 = df[80_000:120_000].copy()

# pdb.set_trace()
def subset_and_clean_data(df):
    df = df[["corporation_name", "address_1", "city", "state", "fei_number", 'registered_agent_name']].copy()
    df['registered_agent_name'] = df['registered_agent_name'].apply(lambda x: ' '.join(str(x).split()))
    df["city"] = df["city"].str.upper()
    df["state"] = df["state"].str.upper()
    # df = CompanyCleaner(df, "corporation_name").clean_entity_names()
    return df
# df = df[["corporation_name", "address_1", "city", "state", "fei_number", 'registered_agent_name']]

# df['registered_agent_name'] = df['registered_agent_name'].apply(lambda x: ' '.join(str(x).split()))
# df["city"] = df["city"].str.upper()
# df["state"] = df["state"].str.upper()
# df = CompanyCleaner(df, "corporation_name").clean_entity_names()
df1 = subset_and_clean_data(df1)
from Entity_Fusion import Entity_Fusion

EF = Entity_Fusion(
    df1,
    {
        "corporation_name": {
            "threshold": 0.5,
            "blocking_column": ["city", 'state'],
            "blocking_criteria": ["blocking_column", 'first_letter'],
            "similarity_method": "tfidf",
        },
        # "registered_agent_name": {
        #     "threshold": 0.9,
        #     "block": True,
        #     "blocking_column": "city",
        #     "criteria": ["first_letter", "blocking_column"],
        #     "similarity_method": "tfidf",
        # },
        "fei_number": {
            "threshold": 1,
            "similarity_method": "numeric_exact",
        },
    }, 
    # pre_clustered_df=pre_clustered_df
)
# pdb.set_trace()

df = EF.create_similarity_matrices()
# df_sim = EF.return_cluster_data_logic_dataframe()


print('hello')
# # Function to process each column and compute similarities
# def process_column(data, column_name, threshold):
#     # Create the vectorizer
#     vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 2), norm=None)
pdb.set_trace()
#     # Fit and transform the data
#     X = vectorizer.fit_transform(data)

#     n_features = X.shape[1]
#     n_samples = X.shape[0]
#     n_components = n_features if n_samples <= 1000 else 1000
#     print(n_features)
#     # Dimensionality reduction using Truncated SVD
#     svd = TruncatedSVD(
#         n_components=n_features
#     )  # Adjust n_components based on memory constraints and desired accuracy
#     X_reduced = svd.fit_transform(X)

#     # Function to compute cosine similarity for chunks and apply threshold
#     def compute_cosine_similarity_chunk(start_idx, end_idx, X_reduced, threshold):
#         chunk_matrix = cosine_similarity(X_reduced[start_idx:end_idx], X_reduced)
#         mask = chunk_matrix >= threshold
#         chunk_matrix = np.where(mask, chunk_matrix, 0)
#         return start_idx, end_idx, chunk_matrix

#     # Parameters
#     chunk_size = 500  # Adjust the chunk size based on your memory constraints
#     n_samples = X_reduced.shape[0]

#     # Initialize a lil_matrix for cosine similarities
#     cos_sim_sparse = lil_matrix((n_samples, n_samples), dtype=np.float32)

#     # Compute cosine similarity chunk by chunk and apply the threshold
#     for start_idx in tqdm(
#         range(0, n_samples, chunk_size),
#         desc=f"Computing cosine similarity in chunks for {column_name}",
#     ):
#         end_idx = min(start_idx + chunk_size, n_samples)
#         start_idx, end_idx, chunk_matrix = compute_cosine_similarity_chunk(
#             start_idx, end_idx, X_reduced, threshold
#         )
#         cos_sim_sparse[start_idx:end_idx] = chunk_matrix
#     # pdb.set_trace()
#     # Convert the lil_matrix to csr_matrix for efficient arithmetic operations
#     cos_sim_sparse = cos_sim_sparse.tocsr()

#     # Get the non-zero indices and values
#     coo = coo_matrix(cos_sim_sparse)
#     rows, cols, values = coo.row, coo.col, coo.data

#     # Store all similarities above the threshold
#     all_similarities = []

#     # Add tqdm to the loop for progress tracking
#     for i, j, value in tqdm(
#         zip(rows, cols, values),
#         total=len(values),
#         desc=f"Processing similarities for {column_name}",
#     ):
#         if i != j:  # Exclude self-similarity
#             all_similarities.append([i, j, value])

#     # Create a DataFrame for all similarities
#     sim_df = pd.DataFrame(
#         all_similarities,
#         columns=[
#             f"{column_name}_1_index",
#             f"{column_name}_2_index",
#             f"{column_name}_similarity",
#         ],
#     )

#     return sim_df


# df_ppp = pd.read_csv("100k.csv")
# df = df_ppp[0:1000].copy()
# df = df[["BorrowerName", "BorrowerAddress"]]
# df = df.rename(columns={"BorrowerName": "col1", "BorrowerAddress": "col2"})
# df = df.dropna()
# df = df.reset_index(drop=True)

# # Extract columns
# data_col1 = df["col1"].tolist()
# data_col2 = df["col2"].tolist()

# # Process the columns
# sim_df_col1 = process_column(df["col1"].tolist(), "col1", 0.7)
# sim_df_col2 = process_column(df["col2"].tolist(), "col2", 0.8)

# # Merge similarity DataFrames on indices
# similarity_df = pd.merge(
#     sim_df_col1,
#     sim_df_col2,
#     left_on=["col1_1_index", "col1_2_index"],
#     right_on=["col2_1_index", "col2_2_index"],
#     how="outer",
#     suffixes=("_col1", "_col2"),
# )

# # Handle idx1 and idx2
# similarity_df["idx1"] = np.where(
#     similarity_df["col1_1_index"].isnull(),
#     similarity_df["col2_1_index"],
#     similarity_df["col1_1_index"],
# )
# similarity_df["idx2"] = np.where(
#     similarity_df["col1_2_index"].isnull(),
#     similarity_df["col2_2_index"],
#     similarity_df["col1_2_index"],
# )

# # Drop the old index columns
# similarity_df = similarity_df.drop(
#     columns=["col1_1_index", "col1_2_index", "col2_1_index", "col2_2_index"]
# )

# # Fill NaN similarities with 0
# similarity_df = similarity_df.fillna(0)
# pdb.set_trace()


# # Function to construct a graph from the similarity DataFrame
# def construct_similarity_graph(df, threshold=0.8):
#     G = nx.Graph()
#     # Add edges based on similarities above the threshold
#     for _, row in df.iterrows():
#         idx1 = int(row["idx1"])
#         idx2 = int(row["idx2"])
#         if row["col1_similarity"] > threshold or row["col2_similarity"] > 0.9:
#             G.add_edge(idx1, idx2)
#     return G


# # Function to find clusters using connected components
# def find_clusters(G):
#     clusters = list(nx.connected_components(G))
#     cluster_map = {}
#     for cluster_id, cluster in enumerate(clusters):
#         for node in cluster:
#             cluster_map[node] = cluster_id
#     return cluster_map


# similarity_graph = construct_similarity_graph(similarity_df, threshold=0.5)
# clusters = find_clusters(similarity_graph)
# df["cluster_label"] = df.index.map(clusters)
# # pdb.set_trace()

# # Reset index for the original DataFrame
# original_df = df.reset_index()

# # Create a mapping dictionary
# index_to_col1 = original_df.set_index("index")["col1"].to_dict()

# # Replace idx1 and idx2 with the corresponding col1 values
# similarity_df["col1_idx1"] = similarity_df["idx1"].map(index_to_col1)
# similarity_df["col1_idx2"] = similarity_df["idx2"].map(index_to_col1)


# index_to_col1 = original_df.set_index("index")["col2"].to_dict()
# similarity_df["col2_idx1"] = similarity_df["idx1"].map(index_to_col1)
# similarity_df["col2_idx2"] = similarity_df["idx2"].map(index_to_col1)
# # similarity_df["cluster_label"] = similarity_df.index.map(clusters)
# pdb.set_trace()
