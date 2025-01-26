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

    def _validate_dataframe(self, df, id_column):
        """
        Validates the input dataframe to ensure it has the required ID column
        with unique values.
        """
        if id_column not in df.columns:
            raise ValueError(
                f"The ID column '{id_column}' is not present in the dataframe."
            )
        if not df[id_column].is_unique:
            duplicated_ids = df[id_column][df[id_column].duplicated()].unique()
            raise ValueError(
                f"The ID column '{id_column}' must contain unique values. Duplicated IDs: {duplicated_ids}"
            )

    def _setup_dataframe(self, df, id_column):
        """
        Prepares the main dataframe
        """
        if "cluster_label" not in df.columns:
            df["cluster_label"] = np.nan
        self._validate_dataframe(df, id_column)
        return df

    def initialize_parameters(self, df, id_column, column_thresholds):
        """
        Sets up parameters and validates the input for the Entity_Fusion object.
        """
        self._validate_dataframe(df, id_column)  # Validate main dataframe
        self.column_thresholds = column_thresholds
        self.id_column = id_column
        self.df_sim = None
        self.graph = None
        self.clusters = None
        self.stopwords = set(ENGLISH_STOP_WORDS)
        self.df = self._setup_dataframe(df, id_column)

    def cluster_data(self):
        # Check if the parameters have been initialized
        if self.df is None or self.column_thresholds is None or self.id_column is None:
            raise ValueError(
                "Parameters have not been initialized. Please call initialize_parameters() first."
            )

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


if __name__ == "__main__":
    df_ppp = pd.read_csv("test_data/100k.csv")
    df = df_ppp[0:5_000].copy()
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
    EF = Entity_Fusion()
    EF.initialize_parameters(df, id_column="index", column_thresholds=column_threshold)
    pdb.set_trace()
