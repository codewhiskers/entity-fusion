import numpy as np
import pandas as pd
from scipy.sparse import lil_matrix, coo_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
import pdb
from collections import defaultdict
import numpy as np
import re
from collections import defaultdict, deque
from sparse_dot_topn import awesome_cossim_topn, sp_matmul_topn
from itertools import combinations


class SimilarityMatrixGenerator:
    def __init__(self, df, conditions, must_links=None, cannot_links=None):
        """
        Args:
            df (pd.DataFrame): The DataFrame to cluster
            conditions (dict): Nested dictionary specifying AND/OR logic
            must_links (set of tuples): Pairs (id1, id2) that must be clustered together
            cannot_links (set of tuples): Pairs (id1, id2) that must NOT be in same cluster
        """
        self.df = df
        self.conditions = conditions
        self.must_links = must_links if must_links else set()
        self.cannot_links = cannot_links if cannot_links else set()

        self.graph = defaultdict(set)
        self.clusters = {}
        self.similarity_calculator = SimilarityCalculator()
        self.data_grouper = DataGrouper(self.df)

    def create_similarity_matrices(self):
        """
        Create similarity matrices for specified columns and store them as lists of tuples.
        """
        if not self.column_thresholds:
            raise ValueError(
                "No column thresholds specified for similarity computation."
            )

        similarity_calculator = SimilarityCalculator()
        data_grouper = DataGrouper(self.df)
        similarity_results = []

        for column, params in self.column_thresholds.items():
            if column not in self.df.columns:
                raise ValueError(f"Column '{column}' not found in the dataframe.")

            column_results = []

            # Group and process with a progress bar
            groups = list(
                data_grouper.group_dataframe(params, column)
            )  # Convert generator to list for tqdm
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
                    column_results.extend(
                        result.tolist()
                    )  # Convert NumPy array to list of tuples

            # Store results for this column as a list of tuples
            if column_results:
                similarity_results.append((column, column_results))

        self.similarity_results = similarity_results

    def _construct_similarity_graph(self):
        """
        Construct a graph from the similarity results stored as lists of tuples,
        combining them via AND or OR logic.
        """
        print("Constructing similarity graph...")

        # 1) Build a list of sets, one set per column
        edges_by_column = []

        for column, data in self.similarity_results:
            # data is a list of [id1, id2, similarity]
            # We'll convert them into a set of (id1, id2) pairs
            column_edges = set()

            for id1, id2, similarity in data:
                # Skip self-loops
                if id1 == id2:
                    continue

                # Ensure (id1, id2) is stored in a canonical order
                if id1 > id2:
                    id1, id2 = id2, id1

                column_edges.add((id1, id2))

            # Only add this set if it's not empty
            if column_edges:
                edges_by_column.append(column_edges)

        # 2) Combine sets with OR or AND
        #    - If no columns produce edges, default to empty set
        if not edges_by_column:
            final_edges = set()
        else:
            if self.combine_method == "AND":
                final_edges = set.intersection(*edges_by_column)
            else:  # default OR
                final_edges = set.union(*edges_by_column)

        # 3) Build the graph from final_edges
        for id1, id2 in final_edges:
            self.graph[id1].add(id2)
            self.graph[id2].add(id1)

        # print("Similarity graph constructed.")

    def _find_clusters_from_graph(self):
        visited = set()
        clusters = []
        for node in self.graph.keys():
            if node not in visited:
                cluster = self._bfs_component(node, visited)
                clusters.append(cluster)

        # Assign cluster IDs
        cluster_map = {}
        for cluster_id, cluster in enumerate(clusters):
            for node in cluster:
                cluster_map[node] = cluster_id
        self.clusters = cluster_map

    def _bfs_component(self, start_node, visited):
        from collections import deque

        queue = deque([start_node])
        component = set()
        while queue:
            node = queue.popleft()
            if node not in visited:
                visited.add(node)
                component.add(node)
                neighbors = self.graph[node]
                for nb in neighbors:
                    if nb not in visited:
                        queue.append(nb)
        return component

    def _assign_cluster_labels(self):
        """
        Add a 'cluster_label' to self.df, with the ID for each connected component.
        """
        self.df["cluster_label"] = self.df.index.map(self.clusters)
        # For any rows not in self.graph, assign unique cluster IDs
        unclustered = self.df["cluster_label"].isna()
        if unclustered.any():
            start = (max(self.clusters.values()) + 1) if self.clusters else 0
            self.df.loc[unclustered, "cluster_label"] = range(
                start, start + unclustered.sum()
            )

    def cluster_data(self, old_label_map=None):
        # 1) If old_label_map is provided, inject must-link edges among them
        if old_label_map is not None:
            from itertools import combinations

            label_groups = defaultdict(list)
            for node, lbl in old_label_map.items():
                label_groups[lbl].append(node)
            for lbl, node_list in label_groups.items():
                for a, b in combinations(node_list, 2):
                    if a > b:
                        a, b = b, a
                    self.must_links.add((a, b))

        # 2) Compute all edges from the nested conditions
        final_edges = self._compute_edges_for_condition(self.conditions)

        # 3) Force-add must_link edges and remove cannot_link edges
        for a, b in self.must_links:
            if a > b:
                a, b = b, a
            final_edges.add((a, b))
        for a, b in self.cannot_links:
            if a > b:
                a, b = b, a
            if (a, b) in final_edges:
                final_edges.remove((a, b))

        # 4) Build the graph from final_edges
        for id1, id2 in final_edges:
            self.graph[id1].add(id2)
            self.graph[id2].add(id1)

        # 5) Find clusters using BFS
        self._find_clusters_from_graph()

        # 6) Final label assignment: if an old_label_map is provided, derive final labels to preserve verified clusters;
        #    otherwise, assign labels normally.
        if old_label_map is not None:
            node_to_label = self._derive_final_labels_with_old(
                self.clusters, old_label_map
            )
            self.df["cluster_label"] = self.df.index.map(node_to_label)
        else:
            self._assign_cluster_labels()

        return self.df

    def _derive_final_labels_with_old(self, clusters, old_label_map):
        """
        clusters: dict node -> component_id
        old_label_map: dict node -> old_label (like 'master_1')
        """
        comp_to_nodes = defaultdict(list)
        for node, comp_id in clusters.items():
            comp_to_nodes[comp_id].append(node)

        final_label_map = {}
        for comp_id, node_list in comp_to_nodes.items():
            old_labels_in_this_comp = set()
            for n in node_list:
                if n in old_label_map:
                    old_labels_in_this_comp.add(old_label_map[n])
            if len(old_labels_in_this_comp) == 1:
                final_label_map[comp_id] = list(old_labels_in_this_comp)[0]
            elif len(old_labels_in_this_comp) > 1:
                # multiple old clusters got merged
                merged_label = "_".join(sorted(old_labels_in_this_comp))
                final_label_map[comp_id] = merged_label
            else:
                # brand new
                final_label_map[comp_id] = f"new_{comp_id}"

        node_to_label = {}
        for node, comp_id in clusters.items():
            node_to_label[node] = final_label_map[comp_id]
        return node_to_label

    def _compute_edges_for_condition(self, condition):
        """
        Recursively compute a set of edges that match the given condition structure.

        The condition can be:
          - A dict with key "and" -> list of sub-conditions
          - A dict with key "or"  -> list of sub-conditions
          - A dict representing one or more column thresholds (leaf condition)

        Returns:
            set of (id1, id2) pairs
        """
        if "and" in condition:
            # condition["and"] is a list of sub-conditions
            sub_conditions = condition["and"]
            # We'll compute edges for each sub-condition and intersect them
            edges_list = []
            for sub_cond in sub_conditions:
                edges_list.append(self._compute_edges_for_condition(sub_cond))
            # Intersection
            return set.intersection(*edges_list) if edges_list else set()

        elif "or" in condition:
            # condition["or"] is a list of sub-conditions
            sub_conditions = condition["or"]
            # We'll compute edges for each sub-condition and union them
            edges_list = []
            for sub_cond in sub_conditions:
                edges_list.append(self._compute_edges_for_condition(sub_cond))
            # Union
            return set.union(*edges_list) if edges_list else set()

        else:
            # We assume it's a "leaf" dictionary specifying one or more columns
            # Example:
            # {
            #   "BorrowerName": { ...params... },
            #   "BorrowerAddress": { ...params... }
            # }
            # We can either interpret multiple columns in the same dictionary as AND logic,
            # or treat each key as a separate sub-condition. Typically, you'd do AND here:
            edges_list = []
            for col_name, params in condition.items():
                edges_for_col = self._compute_edges_for_single_column(col_name, params)
                edges_list.append(edges_for_col)

            # Combine them (AND) or (OR). Let's do AND by default here:
            return set.intersection(*edges_list) if edges_list else set()

    def _compute_edges_for_single_column(self, column, params):
        """
        Compute all pairs (id1, id2) that meet the threshold for one column + similarity method + blocking
        """
        # Extract parameters
        threshold = params.get("threshold", 0.8)
        similarity_method = params.get("similarity_method", "tfidf")
        if similarity_method == "exact":
            threshold = 1.0
        # blocking_criteria = params.get("blocking_criteria", [])
        # blocking_columns = params.get("blocking_column", None)

        # 1) Break dataframe into groups according to blocking
        groups = list(
            self.data_grouper.group_dataframe(params, column)
        )  # see DataGrouper below

        all_pairs = set()
        for group in tqdm(groups, desc=f"Processing groups for column '{column}'"):
            if len(group) <= 1:
                continue

            # 2) Vectorize / compute similarity within this group
            group_data = self.similarity_calculator.initialize_vectorizer(
                similarity_method, group[column]
            )

            # 3) Build adjacency list from threshold
            similarity_arr = self.similarity_calculator.create_similarity_matrix(
                group_data, group.index, column, threshold, similarity_method
            )
            # pdb.set_trace()
            # similarity_arr is Nx3: [id1, id2, sim]
            for id1, id2, sim in similarity_arr:
                # Build pairs in canonical order
                if id1 > id2:
                    id1, id2 = id2, id1
                all_pairs.add((id1, id2))

        return all_pairs


class SimilarityCalculator:
    def initialize_vectorizer(self, similarity_method, data_column):
        if similarity_method == "numeric":
            vectorizer = TfidfVectorizer(
                tokenizer=lambda x: re.findall(r"\d+", x),
                preprocessor=None,
                lowercase=False,
            )
            return vectorizer.fit_transform(data_column.values)
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
            return vectorizer.fit_transform(data_column.values)
        elif similarity_method == "exact":
            return data_column
        else:
            raise ValueError(f"Unsupported similarity method: {similarity_method}")

    def create_similarity_matrix(
        self, group_tfidf, group_ids, column_name, threshold, similarity_method
    ):
        if similarity_method == "exact":
            return self._create_exact_match_matrix(group_tfidf, group_ids)
        else:
            return self._create_cosine_similarity_matrix(
                group_tfidf, group_ids, column_name, threshold
            )

    def _create_exact_match_matrix(self, group_data, group_ids):
        """
        Build a similarity matrix for exact matching.
        For each value in group_data, all rows that share that exact value
        form pairs with similarity=1.0.

        Args:
            group_data (pd.Series): The raw text or values for 'exact' matching
            group_ids (pd.Index or list): The row indices corresponding to group_data
            column_name (str): Not strictly needed here, but included for consistency

        Returns:
            np.ndarray of shape (N, 3), each row = [id1, id2, 1.0]
        """

        # Map each unique exact value -> list of row indices
        value_to_indices = defaultdict(list)
        for idx, val in zip(group_ids, group_data):
            value_to_indices[val].append(idx)

        pairs = []
        # For each exact value, form all unique pairs of indices
        for val, idx_list in value_to_indices.items():
            n = len(idx_list)
            if n > 1:
                # Generate pairwise combos among these indices
                # (skip self-pairs, only do combinations i < j)
                for i in range(n):
                    for j in range(i + 1, n):
                        id1 = idx_list[i]
                        id2 = idx_list[j]
                        pairs.append((id1, id2, 1.0))

        if len(pairs) == 0:
            return np.zeros((0, 3), dtype=float)

        return np.array(pairs, dtype=float)

    def _create_cosine_similarity_matrix(
        self, group_tfidf, group_ids, column_name, threshold
    ):
        n_samples = group_tfidf.shape[0]
        chunk_size = 2000
        large_group_threshold = 500

        if n_samples > large_group_threshold:
            top_n = 100
            cos_sim_sparse = sp_matmul_topn(
                group_tfidf, group_tfidf, top_n=top_n, threshold=threshold, n_threads=-1
            )
        else:
            cos_sim_sparse = lil_matrix((n_samples, n_samples), dtype=np.float32)
            for start_idx in range(0, n_samples, chunk_size):
                end_idx = min(start_idx + chunk_size, n_samples)
                chunk_matrix = cosine_similarity(
                    group_tfidf[start_idx:end_idx], group_tfidf
                )
                # Filter by threshold
                mask = chunk_matrix >= threshold
                chunk_matrix = np.where(mask, chunk_matrix, 0)
                cos_sim_sparse[start_idx:end_idx, :] = chunk_matrix

        return self._finalize_similarity_matrix(cos_sim_sparse, group_ids, column_name)

    def _finalize_similarity_matrix(self, cos_sim_sparse, group_ids, column_name):
        coo_ = coo_matrix(cos_sim_sparse)
        rows, cols, vals = coo_.row, coo_.col, coo_.data

        mask = rows != cols
        rows = rows[mask]
        cols = cols[mask]
        vals = vals[mask]

        group_ids_arr = np.array(group_ids)
        similarities = np.vstack((group_ids_arr[rows], group_ids_arr[cols], vals)).T
        return similarities


class DataGrouper:
    def __init__(self, df):
        self.df = df

    def group_dataframe(self, params, column):
        df_filtered = self._preprocess_column(column)
        blocking_criteria = params.get("blocking_criteria", None)

        if not blocking_criteria:
            # Return the entire DF as one group
            return [df_filtered]

        # Otherwise, apply each criterion in sequence
        grouped_data = [df_filtered]
        for criterion in blocking_criteria:
            grouped_data = self._apply_blocking_criterion(
                grouped_data, criterion, params, column
            )

        # grouped_data is a list of (group_key, sub_df) or just sub_dfs:
        # If we want only sub_dfs, we can strip out the group_key.
        final_groups = []
        for item in grouped_data:
            if isinstance(item, tuple):
                # item is (group_key, sub_df)
                if len(item[1]) > 1:
                    final_groups.append(item[1])
            else:
                # item is sub_df
                if len(item) > 1:
                    final_groups.append(item)
        return final_groups

    def _preprocess_column(self, column):
        return self.df[
            (self.df[column].notnull())
            & (self.df[column] != "")
            & ~(self.df[column].str.lower().isin(["unknown", "nan", "none"]))
        ]

    def _apply_blocking_criterion(self, grouped_data, criterion, params, column):
        new_groups = []
        for group in grouped_data:
            if isinstance(group, tuple):
                # (key, df) pattern
                group = group[1]
            if group.empty:
                continue

            if criterion == "first_letter":
                # group by first letter of the column
                new_groups.extend(list(group.groupby(group[column].str[0], sort=False)))
            elif criterion == "blocking_column":
                blocking_cols = params.get("blocking_column")
                if not blocking_cols:
                    new_groups.append((None, group))
                    continue
                if isinstance(blocking_cols, str):
                    new_groups.extend(list(group.groupby(blocking_cols, sort=False)))
                elif isinstance(blocking_cols, list):
                    new_groups.extend(list(group.groupby(blocking_cols, sort=False)))
                else:
                    raise ValueError("Invalid blocking_column type.")
            else:
                raise ValueError(f"Unsupported blocking criterion: {criterion}")

        # Filter out single-row groups
        filtered_groups = []
        for grp in new_groups:
            key, subdf = grp
            if len(subdf) > 1:
                filtered_groups.append(grp)
        return filtered_groups


class TwoDFMatcher:
    """
    A helper class to match rows between two DataFrames using
    column-level thresholds, blocking criteria, and a TF-IDF similarity method.
    """

    # def __init__(self, tfidf_min_value=0.0):
    #     """
    #     Args:
    #         tfidf_min_value (float): If > 0, will prune TF-IDF entries below this value.
    #     """
    #     self.tfidf_min_value = tfidf_min_value

    def match_two_dataframes_blocking_conditions(
        self,
        df1,
        df2,
        conditions,  # <-- This replaces the old 'column_thresholds'
        top_n=1,
        global_threshold=0.0,
    ):
        """
        Main entry point for nested AND/OR logic.

        Args:
            df1 (pd.DataFrame)
            df2 (pd.DataFrame)
            conditions (dict): A nested dict specifying AND/OR logic, for example:
                {
                    "or": [
                        {
                            "and": [
                                {
                                    "BorrowerName": {
                                        "threshold": 0.6,
                                        "similarity_method": "tfidf",
                                        "blocking_column": ["BorrowerCity"],
                                        "blocking_criteria": ["blocking_column"]
                                    }
                                },
                                {
                                    "BorrowerAddress": {
                                        "threshold": 0.9,
                                        "similarity_method": "tfidf",
                                        "blocking_column": ["BorrowerCity"],
                                        "blocking_criteria": ["blocking_column"]
                                    }
                                }
                            ]
                        },
                        {
                            # Another sub-condition (leaf)
                            "BorrowerName": {
                                "threshold": 0.95,
                                "similarity_method": "tfidf",
                                "blocking_criteria": []
                            }
                        }
                    ]
                }
            top_n (int): how many top matches to keep for each row in each block
            global_threshold (float): final filter for similarity scores if needed

        Returns:
            pd.DataFrame with columns [df1_index, df2_index, max_similarity_score]
        """
        # final_dict = { (df1_idx, df2_idx) -> { "BorrowerName": sim, "BorrowerAddress": sim, ... } }
        final_dict = self._compute_matches_for_condition(conditions, df1, df2, top_n)

        # Convert that to a DataFrame with one column per matched column:
        rows = []
        for (idx1, idx2), col_scores in final_dict.items():
            row_data = {
                "df1_index": idx1,
                "df2_index": idx2,
            }

            for col, score in col_scores.items():
                row_data[col + "_sim"] = score
            rows.append(row_data)

        df_result = pd.DataFrame(rows)
        return df_result

    def _compute_matches_for_condition(self, condition, df1, df2, top_n):
        """
        Return a dict: (idx1, idx2) -> { "columnA": simA, "columnB": simB, ... }
        """
        if "and" in condition:
            dicts = []
            for sub_cond in condition["and"]:
                sub_dict = self._compute_matches_for_condition(
                    sub_cond, df1, df2, top_n
                )
                dicts.append(sub_dict)
            if not dicts:
                return {}
            # Intersect keys, merge column-similarity
            return self._intersect_match_dicts(dicts)

        elif "or" in condition:
            dicts = []
            for sub_cond in condition["or"]:
                sub_dict = self._compute_matches_for_condition(
                    sub_cond, df1, df2, top_n
                )
                dicts.append(sub_dict)
            if not dicts:
                return {}
            # Union keys, merge column-similarity
            return self._union_match_dicts(dicts)

        else:
            return self._compute_matches_for_leaf(condition, df1, df2, top_n)

    def _compute_matches_for_leaf(self, leaf_dict, df1, df2, top_n):
        """
        If the leaf has multiple columns, treat them with an AND by default,
        meaning a pair must pass all column thresholds.
        """
        all_columns_dicts = []
        for col_name, params in leaf_dict.items():
            col_dict = self._compute_matches_for_single_column(
                df1, df2, col_name, params, top_n
            )
            # col_dict is: (idx1, idx2) -> { col_name: similarity }
            all_columns_dicts.append(col_dict)

        if not all_columns_dicts:
            return {}
        # Intersect them, because a leaf with multiple columns typically means "AND"
        return self._intersect_match_dicts(all_columns_dicts)

    def _compute_matches_for_single_column(self, df1, df2, column, params, top_n):
        """
        Return a dict { (df1_idx, df2_idx): { column: similarity } }
        """
        threshold = params.get("threshold", 0.8)
        sim_method = params.get("similarity_method", "tfidf")
        blocking_criteria = params.get("blocking_criteria", [])

        # 1) blocking
        block_pairs = list(
            self._generate_block_pairs(df1, df2, column, blocking_criteria, params)
        )
        out = {}
        # Wrap block_pairs in a tqdm loop
        for sub_df1, sub_df2 in tqdm(block_pairs, desc=f"Block pairs for {column}"):
            if sub_df1.empty or sub_df2.empty:
                continue
            block_result_df = self._match_subdataframes(
                sub_df1, sub_df2, column, sim_method, threshold, top_n
            )
            # block_result_df: df1_index, df2_index, similarity_score
            for row in block_result_df.itertuples():
                pair = (row.df1_index, row.df2_index)
                out[pair] = {column: row.similarity_score}
        return out

    def _intersect_match_dicts(self, dict_list):
        """
        Intersect the keys across all dicts. Then merge column-sim values.
        """
        if not dict_list:
            return {}
        # Start from the first dict
        common_keys = set(dict_list[0].keys())
        for d in dict_list[1:]:
            common_keys &= set(d.keys())
        # For each key in common_keys, we merge the column-sim maps
        out = {}
        for k in common_keys:
            merged_cols = {}
            for d in dict_list:
                merged_cols.update(d[k])  # merges the {col:sim} from each dict
            out[k] = merged_cols
        return out

    def _union_match_dicts(self, dict_list):
        """
        Union the keys across all dicts, merging col-sim maps for pairs that appear in multiple.
        """
        out = {}
        for d in dict_list:
            for k, colmap in d.items():
                if k not in out:
                    out[k] = dict(colmap)  # copy
                else:
                    # merge
                    out[k].update(colmap)
        return out

    def _generate_block_pairs(self, df1, df2, col_name, blocking_criteria, params):
        # as before ...
        if not blocking_criteria:
            yield (df1, df2)
            return

        new_groups_1 = [df1]
        new_groups_2 = [df2]

        for criterion in blocking_criteria:
            tmp_groups_1 = []
            for gdf1 in new_groups_1:
                tmp_groups_1.extend(
                    self._apply_blocking_criterion(gdf1, criterion, params, col_name)
                )

            tmp_groups_2 = []
            for gdf2 in new_groups_2:
                tmp_groups_2.extend(
                    self._apply_blocking_criterion(gdf2, criterion, params, col_name)
                )

            new_groups_1 = tmp_groups_1
            new_groups_2 = tmp_groups_2

        # Now match blocks by key
        dict1 = self._grouped_to_dict(new_groups_1)
        dict2 = self._grouped_to_dict(new_groups_2)

        for block_key, sub1 in dict1.items():
            if block_key in dict2:
                sub2 = dict2[block_key]
                yield (sub1, sub2)

    def _grouped_to_dict(self, group_list):
        # group_list is a list of (key, dataframe)
        # convert to dict
        out = {}
        for key, df_ in group_list:
            out[key] = df_
        return out

    def _apply_blocking_criterion(self, df, criterion, params, col_name):
        # as before ...
        df = df.copy()
        if criterion == "first_letter":
            grouped = df.groupby(df[col_name].astype(str).str[0], dropna=False)
            return list(grouped)
        elif criterion == "blocking_column":
            blocking_columns = params.get("blocking_column")
            if isinstance(blocking_columns, list):
                for bc in blocking_columns:
                    df[bc] = df[bc].astype(str).fillna("NA")
                grouped = df.groupby(blocking_columns, dropna=False)
                return list(grouped)
            elif isinstance(blocking_columns, str):
                grouped = df.groupby(blocking_columns, dropna=False)
                return list(grouped)
            else:
                # Previously, maybe you had "pass" or no return statement
                # This leads to returning None, causing the error.
                # We must return SOMETHING (like an empty list or a single group).
                return [(None, df)]
        else:
            raise ValueError(f"Unsupported blocking criterion: {criterion}")

    def _match_subdataframes(
        self, sub_df1, sub_df2, col_name, similarity_method, threshold, top_n
    ):
        # same logic you already have:
        vectorizer = self._create_vectorizer(similarity_method)
        combined_text = pd.concat([sub_df1[col_name], sub_df2[col_name]]).astype(str)
        vectorizer.fit(combined_text)

        tfidf_a = vectorizer.transform(sub_df1[col_name].astype(str))
        tfidf_b = vectorizer.transform(sub_df2[col_name].astype(str))

        results_sparse = sp_matmul_topn(
            tfidf_a, tfidf_b.T, top_n=top_n, threshold=threshold, n_threads=-1
        )
        return self._sparse_results_to_df(results_sparse, sub_df1.index, sub_df2.index)

    def _create_vectorizer(self, similarity_method):
        # same as your code
        if similarity_method == "numeric":
            return TfidfVectorizer(
                tokenizer=lambda x: re.findall(r"\d+", x),
                preprocessor=None,
                lowercase=False,
            )
        elif similarity_method == "tfidf":
            return TfidfVectorizer(
                analyzer="char_wb",
                preprocessor=None,
                lowercase=True,
                ngram_range=(2, 3),
                norm="l2",
                smooth_idf=True,
                use_idf=True,
            )
        else:
            raise ValueError(f"Unsupported similarity method: {similarity_method}")

    def _sparse_results_to_df(self, sparse_matrix, index_a, index_b):
        coo = sparse_matrix.tocoo()
        rows, cols, data = coo.row, coo.col, coo.data

        df1_indices = index_a.to_numpy()[rows]
        df2_indices = index_b.to_numpy()[cols]

        result_df = pd.DataFrame(
            {
                "df1_index": df1_indices,
                "df2_index": df2_indices,
                "similarity_score": data,
            }
        )
        return result_df


if __name__ == "__main__":
    df = pd.read_csv("test_data/100k.csv", nrows=100_000, encoding="latin-1")
    # pdb.set_trace()
    df = df[["BorrowerName", "BorrowerAddress", "BorrowerCity"]]
    df.reset_index(inplace=True)

    old_label_map = {28600: "master_1", 28871: "master_1"}
    df["cluster_label"] = None
    df.loc[[28600, 28871], "cluster_label"] = [old_label_map[i] for i in [28600, 28871]]
    # pdb.set_trace()
    my_conditions = {
        "or": [
            {
                "and": [
                    {
                        "BorrowerName": {
                            "threshold": 0.3,
                            "blocking_column": ["BorrowerCity"],
                            "blocking_criteria": ["blocking_column"],
                            "similarity_method": "tfidf",
                        }
                    },
                    {
                        "BorrowerAddress": {
                            "threshold": 0.9,
                            "blocking_column": ["BorrowerCity"],
                            "blocking_criteria": ["blocking_column"],
                            "similarity_method": "tfidf",
                        }
                    },
                ]
            },
            {
                "BorrowerName": {
                    "blocking_column": ["BorrowerCity"],
                    "blocking_criteria": ["blocking_column"],
                    "similarity_method": "exact",
                }
            },
        ]
    }

    smg = SimilarityMatrixGenerator(df, my_conditions)
    df = smg.cluster_data(old_label_map=old_label_map)
    pdb.set_trace()

    # df1 = df.iloc[:100_000]
    # df2 = df.iloc[100_000:1_000_000]

    # matcher = TwoDFMatcher()
    # pdb.set_trace()
    # Let's keep top 3 matches per row, apply a final global threshold if needed
    # results = matcher.match_two_dataframes_blocking_conditions(
    #     df1=df1,
    #     df2=df2,
    #     conditions=my_conditions,
    #     top_n=3,
    #     global_threshold=0.0,
    # )
    # pdb.set_trace()

    # merged = results.merge(
    #     df1, left_on="df1_index", right_index=True, how="left"
    # ).merge(
    #     df2,
    #     left_on="df2_index",
    #     right_index=True,
    #     how="left",
    #     suffixes=("_df1", "_df2"),
    # )
    # pdb.set_trace()
    # column_thresholds = {
    #     "BorrowerName": {
    #         "threshold": 0.6,
    #         "blocking_column": ["BorrowerCity"],
    #         "blocking_criteria": ["blocking_column"],
    #         "similarity_method": "tfidf",
    #     },
    #     "BorrowerAddress": {
    #         "threshold": 0.9,
    #         "blocking_column": ["BorrowerCity"],
    #         "blocking_criteria": ["blocking_column"],
    #         "similarity_method": "tfidf",
    #     },
    # }
    # pdb.set_trace()
    EF = SimilarityMatrixGenerator(df, column_thresholds, combine_method="AND")
    clustered_df = EF.cluster_data()
