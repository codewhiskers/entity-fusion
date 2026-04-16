from __future__ import annotations

import pdb
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, Union, Set
from pathlib import Path
import math
import json
from datetime import datetime

import polars as pl
import numpy as np
import networkx as nx
from pyvis.network import Network  # For interactive visualization
from tqdm.auto import tqdm

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


RecordFrame = Union[pl.DataFrame, "pd.DataFrame"]


@dataclass
class IncrementalSignalLinker:
    """
    Enhanced SignalLinker with incremental clustering, match persistence, and network analysis.

    New features:
    - Load/save predefined matches
    - Incremental clustering with merge history tracking
    - Network graph creation for cluster analysis
    - Cluster lineage tracking
    """

    record_id_col: str
    blocking_plan: Dict[str, Dict[str, Any]]
    include_types: Optional[List[str]] = None
    k_required: int = 1
    select_threshold: Optional[float] = None
    select_quantile: float = 0.5

    # Incremental clustering settings
    enable_incremental: bool = True
    matches_file: Optional[str] = None  # Path to save/load matches
    cluster_history_file: Optional[str] = None  # Path to save cluster evolution

    # Network analysis settings
    create_network: bool = True
    network_output_dir: str = "./network_outputs"

    # Pair generation safeguards
    block_cap_action: str = "skip"
    global_df_cap: Optional[int] = None

    # Diagnostics
    return_intermediates: bool = False
    show_progress: bool = False

    # Internal state
    _specs: Dict[str, Dict[str, Any]] = field(default_factory=dict, init=False)
    _match_history: List[Dict] = field(default_factory=list, init=False)
    _cluster_graph: nx.Graph = field(default_factory=nx.Graph, init=False)
    _clusters: Dict[str, Set[str]] = field(default_factory=dict, init=False)
    _cluster_id_counter: int = field(default=0, init=False)
    _last_final_pairs: Optional[pl.DataFrame] = field(default=None, init=False)
    _last_records: Optional[pl.DataFrame] = field(default=None, init=False)

    _alias_store: pl.DataFrame = field(
        default_factory=lambda: pl.DataFrame(
            schema={
                "owner_id": pl.Utf8,
                "alias_type": pl.Utf8,
                "alias_key": pl.Utf8,
                "block_key": pl.Utf8,
                "sim_value": pl.Utf8,
                "batch_id": pl.Utf8,
            }
        ),
        init=False,
    )

    def __post_init__(self):
        """Initialize the linker and load existing matches if available."""
        if self.matches_file and Path(self.matches_file).exists():
            self.load_matches(self.matches_file)
        if self.cluster_history_file and Path(self.cluster_history_file).exists():
            self.load_cluster_history(self.cluster_history_file)

        Path(self.network_output_dir).mkdir(parents=True, exist_ok=True)

    # ------------------------ Enhanced Public API ------------------------

    def link_incremental(
        self, records: RecordFrame, batch_id: Optional[str] = None
    ) -> Dict[str, Any]:
        if batch_id is None:
            batch_id = datetime.now().isoformat()

        df = self._to_polars(records)
        self._specs = self._compile_alias_specs(self.blocking_plan)
        include = self.include_types or list(self._specs.keys())

        # Build alias rows for THIS batch
        alias_curr = self._build_alias_value(df, include).with_columns(
            pl.lit(batch_id).alias("batch_id")
        )

        # Build IDF index on history+current to keep scoring stable across time
        if self._alias_store.height:
            alias_index_basis = pl.concat(
                [
                    self._alias_store.select("alias_key", "alias_type", "owner_id"),
                    alias_curr.select("alias_key", "alias_type", "owner_id"),
                ],
                how="vertical",
            )
            n_records_total = alias_index_basis.select(
                pl.col("owner_id").n_unique()
            ).item()
            alias_index = self._build_alias_index(alias_index_basis, n_records_total)
        else:
            n_records_total = df.select(pl.col(self.record_id_col).n_unique()).item()
            alias_index = self._build_alias_index(alias_curr, n_records_total)

        pairs_parts: List[pl.DataFrame] = []

        # 0) If you keep predefined/existing pairs
        if self.enable_incremental and self._match_history:
            existing_pairs = self._get_existing_pairs_df()
            if existing_pairs.height > 0:
                pairs_parts.append(existing_pairs)

        # 1) Within-batch pairs (fast)
        for t in include:
            p_intra = self._pairs_from_alias_type(alias_curr, t)
            if p_intra.height:
                pairs_parts.append(p_intra)

        # 2) Cross-batch pairs (history vs current)
        if self._alias_store.height:
            for t in include:
                p_cross = self._pairs_cross_from_alias_type(
                    self._alias_store, alias_curr, t
                )
                if p_cross.height:
                    pairs_parts.append(p_cross)

        # Candidate filtering: AND/OR tree takes precedence over K-of-N
        if self._is_condition_tree(self.blocking_plan):
            pairs_long = self._candidates_k_of_n(pairs_parts, 1)
            pairs_long = self._apply_condition_filter(pairs_long, self.blocking_plan)
        else:
            pairs_long = self._candidates_k_of_n(pairs_parts, self.k_required)
        pairs_scored = self._score_pairs(pairs_long, alias_index)
        final_pairs = self._select_pairs(
            pairs_scored, threshold=self.select_threshold, quantile=self.select_quantile
        )

        # Update clusters + history
        if final_pairs.height > 0 and self.enable_incremental:
            self._update_clusters(final_pairs, batch_id)

        # Persist alias signatures
        self._alias_store = (
            pl.concat([self._alias_store, alias_curr], how="vertical").unique()
            if self._alias_store.height
            else alias_curr
        )

        out = {
            "pairs_long": pairs_long,
            "pairs_scored": pairs_scored,
            "final_pairs": final_pairs,
        }

        if self.create_network:
            out["network"] = self._build_network_graph(out)
        out["clusters"] = self.get_clusters()
        out["cluster_stats"] = self.get_cluster_stats()
        out["labeled_records"] = self._label_records(df)

        self._last_final_pairs = out["final_pairs"]
        self._last_records = out["labeled_records"]

        if self.matches_file:
            self.save_matches(self.matches_file)
        if self.cluster_history_file:
            self.save_cluster_history(self.cluster_history_file)

        if self.return_intermediates:
            out.update({"alias_store": self._alias_store})

        return out

    def link(self, records: RecordFrame) -> Dict[str, pl.DataFrame]:
        """
        Original link method with minor modifications for tracking.
        """
        df = self._to_polars(records)
        self._specs = self._compile_alias_specs(self.blocking_plan)

        include = self.include_types or list(self._specs.keys())

        alias_value = self._build_alias_value(df, include)
        n_records = df.select(pl.col(self.record_id_col).n_unique()).item()
        alias_index = self._build_alias_index(alias_value, n_records)

        # Include existing matches if incremental mode
        pairs_parts: List[pl.DataFrame] = []

        if self.enable_incremental and self._match_history:
            existing_pairs = self._get_existing_pairs_df()
            if existing_pairs.height > 0:
                pairs_parts.append(existing_pairs)

        for t in include:
            p = self._pairs_from_alias_type(alias_value, t)
            if p.height:
                pairs_parts.append(p)

        if self._is_condition_tree(self.blocking_plan):
            pairs_long = self._candidates_k_of_n(pairs_parts, 1)
            pairs_long = self._apply_condition_filter(pairs_long, self.blocking_plan)
        else:
            pairs_long = self._candidates_k_of_n(pairs_parts, self.k_required)
        pairs_scored = self._score_pairs(pairs_long, alias_index)
        final_pairs = self._select_pairs(
            pairs_scored, threshold=self.select_threshold, quantile=self.select_quantile
        )

        out = {
            "pairs_long": pairs_long,
            "pairs_scored": pairs_scored,
            "final_pairs": final_pairs,
            "labeled_records": self._label_records(df),
        }

        self._last_final_pairs = out["final_pairs"]
        self._last_records = out["labeled_records"]

        if self.return_intermediates:
            out.update({"alias_value": alias_value, "alias_index": alias_index})
        return out

    def _pairs_cross_from_alias_type(
        self, av_hist: pl.DataFrame, av_curr: pl.DataFrame, alias_type: str
    ) -> pl.DataFrame:
        """
        Generate candidate pairs between historical alias rows and current batch alias rows
        for a single alias_type. Handles both exact and cosine similarity.
        """
        cfg = self._specs[alias_type]
        sim = cfg["similarity"]
        sim_type = sim["type"]
        df_cap = cfg.get("df_cap")
        threshold = sim.get("threshold", 0.8)
        analyzer = sim.get("analyzer", "char_wb")
        ngram_range = sim.get("ngram_range", (3, 5))
        min_df = sim.get("min_df", 1)

        # Filter to this alias_type
        H = av_hist.filter(pl.col("alias_type") == alias_type).select(
            "alias_key", "alias_type", "owner_id", "block_key", "sim_value"
        )
        C = av_curr.filter(pl.col("alias_type") == alias_type).select(
            "alias_key", "alias_type", "owner_id", "block_key", "sim_value"
        )

        if H.height == 0 or C.height == 0:
            return pl.DataFrame(
                schema={
                    "a": pl.Utf8,
                    "b": pl.Utf8,
                    "signal": pl.Utf8,
                    "alias_key": pl.Utf8,
                    "alias_type": pl.Utf8,
                    "sim": pl.Float64,
                }
            )

        if sim_type == "exact":
            # Join on alias_key (or block_key if you prefer broader windows)
            joined = H.join(C, on=["alias_key", "alias_type"], how="inner", suffix="_c")
            if joined.height == 0:
                return pl.DataFrame(
                    schema={
                        "a": pl.Utf8,
                        "b": pl.Utf8,
                        "signal": pl.Utf8,
                        "alias_key": pl.Utf8,
                        "alias_type": pl.Utf8,
                        "sim": pl.Float64,
                    }
                )

            # Optionally cap big blocks
            if df_cap is not None:
                blk_sizes = joined.group_by(["alias_key"]).len().rename({"len": "n"})
                joined = joined.join(blk_sizes, on="alias_key", how="left").filter(
                    pl.col("n") <= df_cap
                )

            # Build pairs hist.owner_id vs curr.owner_id
            a_list = joined.get_column("owner_id").to_list()
            b_list = joined.get_column("owner_id_c").to_list()
            # de-dup and order
            pairs = {
                (min(a, b), max(a, b), joined["alias_key"][i])
                for i, (a, b) in enumerate(zip(a_list, b_list))
                if a != b
            }
            if not pairs:
                return pl.DataFrame(
                    schema={
                        "a": pl.Utf8,
                        "b": pl.Utf8,
                        "signal": pl.Utf8,
                        "alias_key": pl.Utf8,
                        "alias_type": pl.Utf8,
                        "sim": pl.Float64,
                    }
                )
            a, b, k = zip(*pairs)
            return pl.DataFrame(
                {
                    "a": list(a),
                    "b": list(b),
                    "signal": [alias_type] * len(a),
                    "alias_key": list(k),
                    "alias_type": [alias_type] * len(a),
                    "sim": [1.0] * len(a),
                }
            )

        # COSINE PATH: join on block_key (compare within same blocks)
        # Identify shared blocks
        H_blk = H.select("block_key").unique()
        C_blk = C.select("block_key").unique()
        shared = H_blk.join(C_blk, on="block_key", how="inner")
        if shared.height == 0:
            return pl.DataFrame(
                schema={
                    "a": pl.Utf8,
                    "b": pl.Utf8,
                    "signal": pl.Utf8,
                    "alias_key": pl.Utf8,
                    "alias_type": pl.Utf8,
                    "sim": pl.Float64,
                }
            )

        out_parts: List[pl.DataFrame] = []
        for blk in shared.get_column("block_key").to_list():
            gH = H.filter(pl.col("block_key") == blk)
            gC = C.filter(pl.col("block_key") == blk)
            n = gH.height + gC.height
            if n < 2:
                continue
            if df_cap is not None and n > df_cap and self.block_cap_action == "skip":
                continue

            texts_H = gH.get_column("sim_value").to_list()
            texts_C = gC.get_column("sim_value").to_list()
            owners_H = gH.get_column("owner_id").to_list()
            owners_C = gC.get_column("owner_id").to_list()

            # Fit on combined vocab, transform separately, compute cross-sim
            vect = TfidfVectorizer(
                analyzer=analyzer, ngram_range=ngram_range, min_df=min_df
            )
            vect.fit(texts_H + texts_C)
            XH = vect.transform(texts_H)
            XC = vect.transform(texts_C)
            S = (XH @ XC.T).tocoo()
            if S.nnz == 0:
                continue

            a_list, b_list, s_list = [], [], []
            for i, j, simv in zip(S.row, S.col, S.data):
                if simv >= threshold:
                    a = owners_H[i]
                    b = owners_C[j]
                    A, B = (a, b) if a < b else (b, a)
                    a_list.append(A)
                    b_list.append(B)
                    s_list.append(float(simv))

            if a_list:
                out_parts.append(
                    pl.DataFrame(
                        {
                            "a": a_list,
                            "b": b_list,
                            "signal": [alias_type] * len(a_list),
                            "alias_key": [blk] * len(a_list),
                            "alias_type": [alias_type] * len(a_list),
                            "sim": s_list,
                        }
                    )
                )

        if out_parts:
            # Deduplicate a/b at the end
            res = pl.concat(out_parts, how="vertical")
            return res.unique(subset=["a", "b", "alias_type"])
        return pl.DataFrame(
            schema={
                "a": pl.Utf8,
                "b": pl.Utf8,
                "signal": pl.Utf8,
                "alias_key": pl.Utf8,
                "alias_type": pl.Utf8,
                "sim": pl.Float64,
            }
        )

    # ------------------------ Cluster Management ------------------------

    def _update_clusters(self, new_pairs: pl.DataFrame, batch_id: str):
        """Update clusters with new pairs and track history."""
        for row in new_pairs.iter_rows(named=True):
            a, b = row["a"], row["b"]
            score = row.get("score", 1.0)
            signals = row.get("signals", [])

            # Add to graph
            self._cluster_graph.add_edge(
                a,
                b,
                weight=score,
                signals=signals,
                batch_id=batch_id,
                timestamp=datetime.now().isoformat(),
            )

            # Update cluster membership
            cluster_a = self._find_cluster(a)
            cluster_b = self._find_cluster(b)

            if cluster_a is None and cluster_b is None:
                # Create new cluster
                cluster_id = f"C{self._cluster_id_counter}"
                self._cluster_id_counter += 1
                self._clusters[cluster_id] = {a, b}
                self._record_cluster_event("create", cluster_id, {a, b}, batch_id)

            elif cluster_a is not None and cluster_b is None:
                # Add b to a's cluster
                self._clusters[cluster_a].add(b)
                self._record_cluster_event("add", cluster_a, {b}, batch_id)

            elif cluster_a is None and cluster_b is not None:
                # Add a to b's cluster
                self._clusters[cluster_b].add(a)
                self._record_cluster_event("add", cluster_b, {a}, batch_id)

            elif cluster_a != cluster_b:
                # Merge clusters
                merged_cluster = self._clusters[cluster_a] | self._clusters[cluster_b]
                del self._clusters[cluster_b]
                self._clusters[cluster_a] = merged_cluster
                self._record_cluster_event(
                    "merge",
                    cluster_a,
                    merged_cluster,
                    batch_id,
                    metadata={"merged_from": cluster_b},
                )

            # Record match
            self._match_history.append(
                {
                    "a": a,
                    "b": b,
                    "score": score,
                    "signals": signals,
                    "batch_id": batch_id,
                    "timestamp": datetime.now().isoformat(),
                }
            )

    def _find_cluster(self, record_id: str) -> Optional[str]:
        """Find which cluster a record belongs to."""
        for cluster_id, members in self._clusters.items():
            if record_id in members:
                return cluster_id
        return None

    def _record_cluster_event(
        self,
        event_type: str,
        cluster_id: str,
        records: Set[str],
        batch_id: str,
        metadata: Dict = None,
    ):
        """Record cluster evolution events."""
        event = {
            "type": event_type,
            "cluster_id": cluster_id,
            "records": list(records),
            "batch_id": batch_id,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {},
        }
        # This would be saved to cluster_history_file

    # ------------------------ Network Analysis ------------------------

    def _build_network_graph(self, link_result: Dict) -> Dict[str, Any]:
        """Build and analyze the network graph from pairs."""
        final_pairs = link_result.get("final_pairs", pl.DataFrame())

        if final_pairs.height == 0:
            return {"nodes": 0, "edges": 0, "components": 0}

        # Create graph from pairs
        G = nx.Graph()

        for row in final_pairs.iter_rows(named=True):
            a, b = row["a"], row["b"]
            score = row.get("score", 1.0)
            signals = row.get("signals", [])

            G.add_edge(a, b, weight=score, signals=signals)

        # Compute network statistics
        stats = {
            "nodes": G.number_of_nodes(),
            "edges": G.number_of_edges(),
            "components": nx.number_connected_components(G),
            "density": nx.density(G) if G.number_of_nodes() > 0 else 0,
        }

        # Identify key nodes
        if G.number_of_nodes() > 0:
            stats["central_nodes"] = self._identify_central_nodes(G)
            stats["bridges"] = list(nx.bridges(G)) if G.number_of_edges() > 0 else []

        # Save interactive visualization
        if self.create_network:
            viz_path = self._create_interactive_viz(G)
            stats["visualization_path"] = viz_path

        return stats

    def _identify_central_nodes(self, G: nx.Graph, top_k: int = 5) -> Dict[str, List]:
        """Identify the most central nodes in the network."""
        centrality_measures = {}

        if G.number_of_nodes() > 0:
            # Degree centrality
            degree_cent = nx.degree_centrality(G)
            centrality_measures["degree"] = sorted(
                degree_cent.items(), key=lambda x: x[1], reverse=True
            )[:top_k]

            # Betweenness centrality (nodes that bridge communities)
            if G.number_of_edges() > 0:
                between_cent = nx.betweenness_centrality(G, weight="weight")
                centrality_measures["betweenness"] = sorted(
                    between_cent.items(), key=lambda x: x[1], reverse=True
                )[:top_k]

        return centrality_measures

    def _create_interactive_viz(self, G: nx.Graph) -> str:
        """Create an interactive network visualization using pyvis."""
        net = Network(
            height="750px",
            width="100%",
            bgcolor="#222222",
            font_color="white",
            notebook=False,
        )

        # Add nodes with cluster coloring
        node_colors = {}
        for cluster_id, members in self._clusters.items():
            # Generate a color for this cluster
            color = f"#{hash(cluster_id) % 0xFFFFFF:06x}"
            for member in members:
                if member in G.nodes():
                    node_colors[member] = color

        for node in G.nodes():
            color = node_colors.get(node, "#97c2fc")
            net.add_node(node, label=node, color=color, title=f"Record: {node}")

        # Add edges with weight as thickness
        for edge in G.edges(data=True):
            weight = edge[2].get("weight", 1.0)
            signals = edge[2].get("signals", [])
            title = f"Score: {weight:.2f}, Signals: {signals}"
            net.add_edge(edge[0], edge[1], value=weight, title=title)

        # Configure physics
        net.barnes_hut(gravity=-80000, central_gravity=0.3, spring_length=250)

        # Save to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"{self.network_output_dir}/network_{timestamp}.html"
        net.save_graph(output_path)

        return output_path

    def visualize_cluster_evolution(self, cluster_id: str) -> str:
        """Create a visualization showing how a specific cluster evolved."""
        if cluster_id not in self._clusters:
            raise ValueError(f"Cluster {cluster_id} not found")

        # Create subgraph for this cluster
        cluster_members = self._clusters[cluster_id]
        subgraph = self._cluster_graph.subgraph(cluster_members).copy()

        # Create temporal network showing edge addition order
        net = Network(
            height="750px",
            width="100%",
            bgcolor="#222222",
            font_color="white",
            notebook=False,
        )

        # Add nodes
        for node in subgraph.nodes():
            net.add_node(node, label=node, color="#97c2fc")

        # Add edges with temporal information
        edges_with_time = []
        for edge in subgraph.edges(data=True):
            timestamp = edge[2].get("timestamp", "")
            batch_id = edge[2].get("batch_id", "")
            edges_with_time.append((edge, timestamp, batch_id))

        # Sort by timestamp
        edges_with_time.sort(key=lambda x: x[1])

        # Add edges with increasing opacity based on time
        for i, (edge, timestamp, batch_id) in enumerate(edges_with_time):
            opacity = (
                0.3 + (0.7 * i / len(edges_with_time))
                if len(edges_with_time) > 1
                else 1.0
            )
            weight = edge[2].get("weight", 1.0)
            signals = edge[2].get("signals", [])

            title = f"Batch: {batch_id}, Time: {timestamp}, Score: {weight:.2f}, Signals: {signals}"
            net.add_edge(
                edge[0], edge[1], value=weight, title=title, color={"opacity": opacity}
            )

        # Save
        output_path = f"{self.network_output_dir}/cluster_{cluster_id}_evolution.html"
        net.save_graph(output_path)

        return output_path

    def visualize_records_with_fields(
        self,
        records: RecordFrame,
        final_pairs: pl.DataFrame,
        fields: List[str],
        cluster_label: Optional[str] = None,
        output_path: Optional[str] = None,
        label_field: Optional[str] = None,
        max_label_length: int = 32,
    ) -> str:
        """
        Create an interactive network using record fields for node labels/tooltips.

        Parameters
        ----------
        records:
            DataFrame containing at least the record id column and the requested fields.
            If a `cluster_label` column is present, it can be used to filter to one cluster.
        final_pairs:
            The accepted pair output from `link()` or `link_incremental()`.
        fields:
            Record fields to display in the node tooltip.
        cluster_label:
            Optional cluster id / cluster_label value to visualize only one cluster.
        output_path:
            Optional path for the generated HTML file.
        label_field:
            Optional field to use as the visible node label. Defaults to `record_id_col`.
        """
        df = self._to_polars(records)
        if final_pairs.height == 0:
            raise ValueError("final_pairs is empty; nothing to visualize")

        if cluster_label is not None:
            if "cluster_label" not in df.columns:
                raise ValueError(
                    "records must include a 'cluster_label' column to filter by cluster"
                )
            df = df.filter(pl.col("cluster_label") == cluster_label)
            if df.height == 0:
                raise ValueError(f"No records found for cluster_label={cluster_label!r}")

        node_ids = set(df.get_column(self.record_id_col).cast(pl.Utf8).to_list())
        pairs = final_pairs.with_columns(
            a=pl.col("a").cast(pl.Utf8),
            b=pl.col("b").cast(pl.Utf8),
        )
        pairs = pairs.filter(
            pl.col("a").is_in(node_ids) & pl.col("b").is_in(node_ids)
        )
        if pairs.height == 0:
            raise ValueError("No accepted edges exist for the requested records/cluster")

        display_fields = [f for f in fields if f in df.columns]
        if not display_fields:
            raise ValueError("None of the requested fields were found in records")

        label_col = (
            label_field
            if label_field and label_field in df.columns
            else self.record_id_col
        )

        node_records = {}
        for row in df.iter_rows(named=True):
            record_id = str(row[self.record_id_col])
            node_records[record_id] = row

        net = Network(
            height="850px",
            width="100%",
            bgcolor="#222222",
            font_color="white",
            notebook=False,
        )

        for record_id, row in node_records.items():
            label_value = str(row.get(label_col, record_id))
            if len(label_value) > max_label_length:
                label_value = label_value[: max_label_length - 3] + "..."

            tooltip_lines = [f"<b>{self.record_id_col}</b>: {record_id}"]
            if "cluster_label" in row:
                tooltip_lines.append(f"<b>cluster_label</b>: {row['cluster_label']}")
            for field in display_fields:
                tooltip_lines.append(f"<b>{field}</b>: {row.get(field, '')}")

            node_color = "#97c2fc"
            cluster_value = row.get("cluster_label")
            if cluster_value and str(cluster_value).startswith("C"):
                node_color = f"#{hash(str(cluster_value)) % 0xFFFFFF:06x}"

            net.add_node(
                record_id,
                label=label_value,
                color=node_color,
                title="<br>".join(tooltip_lines),
            )

        for edge in pairs.iter_rows(named=True):
            signals = edge.get("signals", [])
            if isinstance(signals, list):
                signal_text = ", ".join(signals)
            else:
                signal_text = str(signals)
            score = float(edge.get("score", 1.0))
            title = f"score: {score:.3f}<br>signals: {signal_text}"
            net.add_edge(edge["a"], edge["b"], value=score, title=title)

        net.barnes_hut(gravity=-80000, central_gravity=0.25, spring_length=220)

        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            suffix = cluster_label if cluster_label is not None else "all"
            output_path = (
                f"{self.network_output_dir}/field_network_{suffix}_{timestamp}.html"
            )

        net.save_graph(output_path)
        return output_path

    def generate_dashboard(
        self,
        fields: List[str],
        output_path: str,
        *,
        records: Optional[RecordFrame] = None,
        final_pairs: Optional[pl.DataFrame] = None,
        top_n_clusters: int = 50,
        min_cluster_size: int = 2,
        label_field: Optional[str] = None,
    ) -> str:
        """
        Generate a self-contained interactive HTML dashboard.

        The dashboard shows a vis.js network of matched records on the left and a
        reactive record table on the right.  Clicking a node (or a cluster in the
        sidebar list) highlights that cluster in the graph and populates the table;
        clicking a table row focuses and selects the corresponding node.

        Parameters
        ----------
        fields:
            Record fields to show in node tooltips and the record table.
        output_path:
            Destination path for the HTML file.
        records:
            DataFrame with at least `record_id_col` and the requested `fields`.
            Defaults to the labeled records from the most recent link() call.
        final_pairs:
            Accepted pairs from link() or link_incremental().
            Defaults to the final pairs from the most recent link() call.
        top_n_clusters:
            Only render the N largest clusters.
        min_cluster_size:
            Exclude clusters smaller than this.
        label_field:
            Field to use as the visible node label.  Defaults to record_id_col.

        Returns
        -------
        The output_path that was written.
        """
        import colorsys
        import json as _json

        if records is None:
            if self._last_records is None:
                raise ValueError(
                    "No records available. Either call link() / link_incremental() first, "
                    "or pass records= explicitly."
                )
            records = self._last_records
        if final_pairs is None:
            if self._last_final_pairs is None:
                raise ValueError(
                    "No final_pairs available. Either call link() / link_incremental() first, "
                    "or pass final_pairs= explicitly."
                )
            final_pairs = self._last_final_pairs

        df = self._to_polars(records)

        # ── 1. Choose clusters to show ────────────────────────────────────────
        cluster_sizes = sorted(
            [
                (cid, len(members))
                for cid, members in self._clusters.items()
                if len(members) >= min_cluster_size
            ],
            key=lambda x: -x[1],
        )[:top_n_clusters]

        if not cluster_sizes:
            raise ValueError(
                "No clusters meet the criteria (min_cluster_size=%d). "
                "Run link() or link_incremental() first." % min_cluster_size
            )

        shown_cluster_ids = {cid for cid, _ in cluster_sizes}

        # ── 2. record_id → cluster_id ─────────────────────────────────────────
        id_to_cluster: Dict[str, str] = {}
        for cid, members in self._clusters.items():
            if cid in shown_cluster_ids:
                for member in members:
                    id_to_cluster[str(member)] = cid
        shown_record_ids = set(id_to_cluster.keys())

        # ── 3. Filter records; resolve available fields ───────────────────────
        available_fields = [f for f in fields if f in df.columns]
        label_col = (
            label_field
            if label_field and label_field in df.columns
            else self.record_id_col
        )
        shown_df = df.filter(
            pl.col(self.record_id_col).cast(pl.Utf8).is_in(list(shown_record_ids))
        )

        # ── 4. Colour palette: golden-ratio HSV → perceptually distinct hues ──
        def _cluster_color(i: int) -> tuple:
            hue = (i * 0.618033988749895) % 1.0
            r, g, b = colorsys.hsv_to_rgb(hue, 0.55, 0.90)
            bg = f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"
            r2, g2, b2 = colorsys.hsv_to_rgb(hue, 0.75, 0.68)
            border = f"#{int(r2*255):02x}{int(g2*255):02x}{int(b2*255):02x}"
            return bg, border

        color_map = {
            cid: _cluster_color(i) for i, (cid, _) in enumerate(cluster_sizes)
        }

        # ── 5. Build vis.js nodes + table data ───────────────────────────────
        vis_nodes: List[Dict] = []
        table_data: Dict[str, List[Dict]] = {}

        for row in shown_df.iter_rows(named=True):
            rid = str(row[self.record_id_col])
            cid = id_to_cluster.get(rid)
            if cid is None:
                continue

            bg, border = color_map.get(cid, ("#97c2fc", "#4a7fc2"))
            raw_label = str(row.get(label_col, rid))
            label = raw_label[:32] + ("…" if len(raw_label) > 32 else "")

            tooltip_parts = [f"<b>id:</b> {rid}", f"<b>cluster:</b> {cid}"]
            for f in available_fields:
                v = row.get(f)
                if v is not None:
                    tooltip_parts.append(f"<b>{f}:</b> {v}")

            vis_nodes.append({
                "id": rid,
                "label": label,
                "title": "<br>".join(tooltip_parts),
                "cluster_id": cid,
                "color": {
                    "background": bg,
                    "border": border,
                    "highlight": {"background": "#fde68a", "border": "#d97706"},
                    "hover": {"background": "#fef3c7", "border": "#d97706"},
                },
            })

            row_dict = {self.record_id_col: rid}
            for f in available_fields:
                v = row.get(f)
                row_dict[f] = "" if v is None else str(v)
            table_data.setdefault(cid, []).append(row_dict)

        # ── 6. Build vis.js edges (within shown clusters only) ────────────────
        vis_edges: List[Dict] = []
        if final_pairs.height > 0:
            for row in final_pairs.iter_rows(named=True):
                a, b = str(row["a"]), str(row["b"])
                if a not in shown_record_ids or b not in shown_record_ids:
                    continue
                score = round(float(row.get("score", 1.0)), 3)
                signals = row.get("signals") or []
                if isinstance(signals, str):
                    signals = [signals]
                tooltip = f"score: {score:.3f}<br>signals: {', '.join(signals)}"
                edge_color = color_map.get(id_to_cluster.get(a, ""), ("#97c2fc", "#4a7fc2"))[1]
                vis_edges.append({
                    "from": a,
                    "to": b,
                    "title": tooltip,
                    "value": score,
                    "color": {"color": edge_color, "highlight": "#f59e0b", "hover": "#f59e0b"},
                })

        # ── 7. Summary stats + cluster list for sidebar ───────────────────────
        stats = {
            "clusters_shown": len(cluster_sizes),
            "clusters_total": len(self._clusters),
            "records_shown": len(vis_nodes),
            "edges_shown": len(vis_edges),
            "largest": cluster_sizes[0][1] if cluster_sizes else 0,
        }
        cluster_list = [
            {"id": cid, "size": sz, "color": color_map[cid][0]}
            for cid, sz in cluster_sizes
        ]

        # ── 8. Inject data into template and write ────────────────────────────
        html = _DASHBOARD_HTML
        html = html.replace("__NODES__",        _json.dumps(vis_nodes))
        html = html.replace("__EDGES__",        _json.dumps(vis_edges))
        html = html.replace("__TABLE_DATA__",   _json.dumps(table_data))
        html = html.replace("__FIELDS__",       _json.dumps([self.record_id_col] + available_fields))
        html = html.replace("__STATS__",        _json.dumps(stats))
        html = html.replace("__CLUSTER_LIST__", _json.dumps(cluster_list))

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as fh:
            fh.write(html)
        return output_path

    # ------------------------ Persistence ------------------------

    def save_matches(self, filepath: str):
        """Save match history to JSON file."""
        with open(filepath, "w") as f:
            json.dump(
                {
                    "matches": self._match_history,
                    "clusters": {k: list(v) for k, v in self._clusters.items()},
                    "cluster_id_counter": self._cluster_id_counter,
                },
                f,
                indent=2,
            )

    def load_matches(self, filepath: str):
        """Load match history from JSON file."""
        with open(filepath, "r") as f:
            data = json.load(f)
            self._match_history = data.get("matches", [])
            self._clusters = {k: set(v) for k, v in data.get("clusters", {}).items()}
            self._cluster_id_counter = data.get("cluster_id_counter", 0)

            # Rebuild graph from matches
            for match in self._match_history:
                self._cluster_graph.add_edge(
                    match["a"],
                    match["b"],
                    weight=match.get("score", 1.0),
                    signals=match.get("signals", []),
                    batch_id=match.get("batch_id", "unknown"),
                    timestamp=match.get("timestamp", ""),
                )

    def save_cluster_history(self, filepath: str):
        """Save cluster evolution history."""
        # Implementation would save the cluster events
        pass

    def load_cluster_history(self, filepath: str):
        """Load cluster evolution history."""
        # Implementation would load the cluster events
        pass

    def _get_existing_pairs_df(self) -> pl.DataFrame:
        """Convert existing matches to DataFrame format for reuse."""
        if not self._match_history:
            return pl.DataFrame()

        # Convert match history to pairs format
        pairs_data = []
        for match in self._match_history:
            for signal in match.get("signals", ["PREDEFINED"]):
                pairs_data.append(
                    {
                        "a": match["a"],
                        "b": match["b"],
                        "signal": signal,
                        "alias_key": f"PREDEFINED:{match['a']}:{match['b']}",
                        "alias_type": "PREDEFINED",
                        "sim": 1.0,
                    }
                )

        if pairs_data:
            return pl.DataFrame(pairs_data)
        return pl.DataFrame()

    # ------------------------ Analysis Methods ------------------------

    def _label_records(self, records: pl.DataFrame) -> pl.DataFrame:
        """
        Return records with a 'cluster_label' column attached.

        Records that belong to a cluster get that cluster's ID.
        Unmatched (singleton) records get a label of the form
        'singleton_<record_id>' so every row always has a label.
        """
        id_to_cluster: Dict[str, str] = {}
        for cluster_id, members in self._clusters.items():
            for member in members:
                id_to_cluster[str(member)] = cluster_id

        labels = [
            id_to_cluster.get(str(rid), f"singleton_{rid}")
            for rid in records[self.record_id_col].to_list()
        ]
        return records.with_columns(
            pl.Series("cluster_label", labels, dtype=pl.Utf8)
        )

    def get_clusters(self) -> Dict[str, List[str]]:
        """Get current clusters."""
        return {k: list(v) for k, v in self._clusters.items()}

    def get_cluster_stats(self) -> Dict[str, Any]:
        """Get statistics about current clusters."""
        if not self._clusters:
            return {"total_clusters": 0, "total_records": 0}

        cluster_sizes = [len(members) for members in self._clusters.values()]
        return {
            "total_clusters": len(self._clusters),
            "total_records": sum(cluster_sizes),
            "avg_cluster_size": np.mean(cluster_sizes),
            "max_cluster_size": max(cluster_sizes),
            "min_cluster_size": min(cluster_sizes),
            "size_distribution": dict(
                zip(*np.unique(cluster_sizes, return_counts=True))
            ),
        }

    def get_cluster_graph(self, cluster_id: str) -> nx.Graph:
        """Get the subgraph for a specific cluster."""
        if cluster_id not in self._clusters:
            raise ValueError(f"Cluster {cluster_id} not found")

        members = self._clusters[cluster_id]
        return self._cluster_graph.subgraph(members).copy()

    def analyze_cluster_connectivity(self, cluster_id: str) -> Dict[str, Any]:
        """Analyze how records in a cluster are connected."""
        subgraph = self.get_cluster_graph(cluster_id)

        analysis = {
            "cluster_id": cluster_id,
            "size": subgraph.number_of_nodes(),
            "edges": subgraph.number_of_edges(),
            "density": nx.density(subgraph) if subgraph.number_of_nodes() > 0 else 0,
            "is_connected": nx.is_connected(subgraph),
        }

        if subgraph.number_of_nodes() > 0:
            # Find minimum spanning tree (shows essential connections)
            if nx.is_connected(subgraph):
                mst = nx.minimum_spanning_tree(
                    subgraph, weight=lambda u, v, d: 1 / d.get("weight", 1)
                )
                analysis["mst_edges"] = list(mst.edges(data=True))

            # Find bridges (edges whose removal would disconnect the cluster)
            analysis["bridges"] = (
                list(nx.bridges(subgraph)) if subgraph.number_of_edges() > 0 else []
            )

            # Identify hub nodes
            degree_dict = dict(subgraph.degree())
            analysis["hub_nodes"] = sorted(
                degree_dict.items(), key=lambda x: x[1], reverse=True
            )[:5]

            # Signal distribution
            signal_counts = {}
            for _, _, data in subgraph.edges(data=True):
                for signal in data.get("signals", []):
                    signal_counts[signal] = signal_counts.get(signal, 0) + 1
            analysis["signal_distribution"] = signal_counts

        return analysis

    # ------------------------ Original Methods (kept from base class) ------------------------

    @staticmethod
    def _is_condition_tree(plan: Dict) -> bool:
        """Return True if the blocking plan uses AND/OR condition tree syntax."""
        return "and" in plan or "or" in plan

    def _extract_leaf_specs(self, condition: Dict) -> Dict[str, Dict]:
        """Recursively extract flat signal specs from an AND/OR condition tree."""
        if "and" in condition:
            specs: Dict[str, Dict] = {}
            for sub in condition["and"]:
                specs.update(self._extract_leaf_specs(sub))
            return specs
        if "or" in condition:
            specs = {}
            for sub in condition["or"]:
                specs.update(self._extract_leaf_specs(sub))
            return specs
        # Leaf node — keys are signal type names
        return dict(condition)

    def _compile_alias_specs(
        self, plan: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """Compile alias specifications from blocking plan (flat or AND/OR tree)."""
        leaf_plan = self._extract_leaf_specs(plan) if self._is_condition_tree(plan) else plan
        specs: Dict[str, Dict[str, Any]] = {}
        for alias_type, cfg in leaf_plan.items():
            fields = cfg["fields"]
            compose = cfg.get("compose")
            block_on = cfg.get("block_on")
            weight = float(cfg.get("weight", 1.0))
            df_cap = cfg.get("df_cap", self.global_df_cap)
            bonus = float(cfg.get("bonus", 0.0))
            similarity = cfg.get("similarity", {"type": "exact"})
            stype = similarity.get("type", "exact").lower()
            if stype not in ("cosine", "exact"):
                stype = "cosine"
            sim = {
                "type": stype,
                "field": similarity.get("field", fields[0]),
                "threshold": float(similarity.get("threshold", 0.8)),
                "analyzer": similarity.get("analyzer", "char_wb"),
                "ngram_range": tuple(similarity.get("ngram_range", (3, 5))),
                "min_df": int(similarity.get("min_df", 1)),
            }
            specs[alias_type] = dict(
                fields=fields,
                compose=compose,
                block_on=block_on,
                weight=weight,
                df_cap=df_cap,
                bonus=bonus,
                similarity=sim,
                alias_type=alias_type,
            )
        return specs

    def _build_alias_value(self, df: pl.DataFrame, include: List[str]) -> pl.DataFrame:
        """Build alias values for matching."""
        parts: List[pl.DataFrame] = []
        rid = self.record_id_col

        for atype in include:
            cfg = self._specs[atype]
            fields: List[str] = cfg["fields"]
            compose: Optional[str] = cfg.get("compose")
            block_on: Optional[List[str]] = cfg.get("block_on")
            sim_field: str = cfg["similarity"]["field"]

            if compose:
                import re

                fmt = compose
                names = re.findall(r"{([^{}]+)}", fmt)
                kwargs = {
                    name: pl.col(name).cast(pl.Utf8).fill_null("") for name in names
                }
                alias_val_expr = pl.format(fmt, **kwargs)
            else:
                alias_val_expr = pl.concat_str(
                    [pl.col(c).cast(pl.Utf8).fill_null("") for c in fields],
                    separator="|",
                )

            if block_on:
                block_expr = pl.concat_str(
                    [pl.col(c).cast(pl.Utf8).fill_null("") for c in block_on],
                    separator="|",
                )
            else:
                block_expr = alias_val_expr

            part = (
                df.select(
                    owner_id=pl.col(rid).alias("owner_id"),
                    alias_type=pl.lit(atype),
                    alias_value=alias_val_expr.alias("alias_value"),
                    block_value=block_expr.alias("block_value"),
                    sim_value=pl.col(sim_field).cast(pl.Utf8).fill_null(""),
                )
                .with_columns(
                    alias_key=pl.concat_str(
                        [pl.lit(atype), pl.lit(":"), pl.col("alias_value")]
                    ),
                    block_key=pl.concat_str(
                        [pl.lit(atype), pl.lit(":"), pl.col("block_value")]
                    ),
                )
                .select("owner_id", "alias_type", "alias_key", "block_key", "sim_value")
            )

            parts.append(part)

        alias_value = pl.concat(parts, how="vertical").unique()
        return alias_value

    def _build_alias_index(
        self, alias_value: pl.DataFrame, n_records: int
    ) -> pl.DataFrame:
        k1 = (
            alias_value.select("alias_key", "alias_type", "owner_id")
            .group_by(["alias_key", "alias_type"])
            .agg(df_count=pl.col("owner_id").n_unique())
        )
        parts = [k1]

        if "block_key" in alias_value.columns:
            k2 = (
                alias_value.select(
                    alias_key=pl.col("block_key"),
                    alias_type=pl.col("alias_type"),
                    owner_id=pl.col("owner_id"),
                )
                .group_by(["alias_key", "alias_type"])
                .agg(df_count=pl.col("owner_id").n_unique())
            )
            parts.append(k2)

        idx = pl.concat(parts, how="vertical")
        idx = (
            idx.group_by(["alias_key", "alias_type"])
            .agg(df_count=pl.col("df_count").max())
            .with_columns(
                idf=pl.col("df_count").map_elements(
                    lambda df: math.log((n_records + 1) / (df + 1)) + 1.0,
                    return_dtype=pl.Float64,
                )
            )
        )
        return idx

    def _pairs_from_alias_type(
        self, alias_value: pl.DataFrame, alias_type: str
    ) -> pl.DataFrame:
        """Generate candidate pairs for a single alias_type."""
        cfg = self._specs[alias_type]
        sim = cfg["similarity"]
        sim_type = sim["type"]
        df_cap = cfg.get("df_cap")

        av = alias_value.filter(pl.col("alias_type") == alias_type).select(
            "alias_key", "alias_type", "owner_id", "block_key", "sim_value"
        )

        if av.height == 0:
            return pl.DataFrame(
                schema={
                    "a": pl.Utf8,
                    "b": pl.Utf8,
                    "signal": pl.Utf8,
                    "alias_key": pl.Utf8,
                    "alias_type": pl.Utf8,
                    "sim": pl.Float64,
                }
            )

        av = av.with_columns(
            block_key=pl.when(pl.col("block_key").is_null())
            .then(pl.col("alias_key"))
            .otherwise(pl.col("block_key"))
        )

        if sim_type == "exact":
            pairs_rows: List[pl.DataFrame] = []
            block_sizes = (
                av.select("block_key").group_by("block_key").len().rename({"len": "n"})
            )
            av2 = av.join(block_sizes, on="block_key", how="left")
            if df_cap is not None:
                av2 = av2.filter(pl.col("n") <= df_cap)

            for _, g in av2.group_by("alias_key", maintain_order=False):
                owners = g.get_column("owner_id").to_list()
                if len(owners) < 2:
                    continue
                owners = sorted(owners)
                a_list, b_list = [], []
                for i in range(len(owners) - 1):
                    ai = owners[i]
                    for j in range(i + 1, len(owners)):
                        a_list.append(ai)
                        b_list.append(owners[j])
                if a_list:
                    pairs_rows.append(
                        pl.DataFrame(
                            {
                                "a": a_list,
                                "b": b_list,
                                "signal": [alias_type] * len(a_list),
                                "alias_key": [g["alias_key"][0]] * len(a_list),
                                "alias_type": [alias_type] * len(a_list),
                                "sim": [1.0] * len(a_list),
                            }
                        )
                    )
            return (
                pl.concat(pairs_rows, how="vertical")
                if pairs_rows
                else pl.DataFrame(
                    schema={
                        "a": pl.Utf8,
                        "b": pl.Utf8,
                        "signal": pl.Utf8,
                        "alias_key": pl.Utf8,
                        "alias_type": pl.Utf8,
                        "sim": pl.Float64,
                    }
                )
            )

        # Cosine similarity path
        analyzer = sim["analyzer"]
        ngram_range = sim["ngram_range"]
        min_df = sim["min_df"]
        threshold = sim["threshold"]

        blocks = []
        for blk, g in av.group_by("block_key", maintain_order=False):
            n = g.height
            if df_cap is not None and n > df_cap:
                if self.block_cap_action == "skip":
                    continue
            blocks.append((blk, g))

        out_parts: List[pl.DataFrame] = []
        block_iter = blocks
        if self.show_progress:
            block_iter = tqdm(
                blocks,
                desc=f"{alias_type} blocks",
                unit="block",
            )
        for blk, g in block_iter:
            texts = g.get_column("sim_value").to_list()
            if len(texts) < 2:
                continue
            # Skip blocks that have no usable text for the configured analyzer.
            cleaned_texts = [str(text).strip() for text in texts]
            if sum(bool(text) for text in cleaned_texts) < 2:
                continue
            vect = TfidfVectorizer(
                analyzer=analyzer, ngram_range=ngram_range, min_df=min_df
            )
            try:
                X = vect.fit_transform(cleaned_texts)
            except ValueError:
                # scikit-learn raises "empty vocabulary" when all values in a
                # block are blank or too short to produce any n-grams.
                continue
            S = cosine_similarity(X, dense_output=False)
            S = S.tocoo()
            if S.nnz == 0:
                continue
            owner_ids = g.get_column("owner_id").to_list()
            a_list, b_list, s_list = [], [], []
            for i, j, simv in zip(S.row, S.col, S.data):
                if j <= i:
                    continue
                if simv >= threshold:
                    a, b = sorted((owner_ids[i], owner_ids[j]))
                    a_list.append(a)
                    b_list.append(b)
                    s_list.append(float(simv))
            if a_list:
                block_key_str = blk[0] if isinstance(blk, tuple) else str(blk)
                out_parts.append(
                    pl.DataFrame(
                        {
                            "a": a_list,
                            "b": b_list,
                            "signal": [alias_type] * len(a_list),
                            "alias_key": [block_key_str] * len(a_list),
                            "alias_type": [alias_type] * len(a_list),
                            "sim": s_list,
                        }
                    )
                )

        if out_parts:
            return pl.concat(out_parts, how="vertical")
        else:
            return pl.DataFrame(
                schema={
                    "a": pl.Utf8,
                    "b": pl.Utf8,
                    "signal": pl.Utf8,
                    "alias_key": pl.Utf8,
                    "alias_type": pl.Utf8,
                    "sim": pl.Float64,
                }
            )

    def _candidates_k_of_n(
        self, pairs_by_signal: List[pl.DataFrame], k_required: int
    ) -> pl.DataFrame:
        """Apply K-of-N signal requirement."""
        parts = [p for p in pairs_by_signal if p is not None and p.height]
        if not parts:
            return pl.DataFrame(
                schema={
                    "a": pl.Utf8,
                    "b": pl.Utf8,
                    "signal": pl.Utf8,
                    "alias_key": pl.Utf8,
                    "alias_type": pl.Utf8,
                    "sim": pl.Float64,
                }
            )

        temp = pl.concat(parts, how="vertical")
        keep = (
            temp.select("a", "b", "signal")
            .unique()
            .group_by(["a", "b"])
            .agg(n_signals=pl.len())
            .filter(pl.col("n_signals") >= k_required)
            .select("a", "b")
        )
        return temp.join(keep, on=["a", "b"], how="inner")

    def _apply_condition_filter(
        self, pairs_long: pl.DataFrame, condition: Dict
    ) -> pl.DataFrame:
        """
        Filter pairs_long according to an AND/OR condition tree.

        - AND: a pair must satisfy every sub-condition
        - OR:  a pair must satisfy at least one sub-condition
        - Leaf: a pair satisfies the condition if it has a match on any
                signal type named in that leaf dict

        Returns the filtered pairs_long (same schema, subset of rows).
        """
        if pairs_long.height == 0:
            return pairs_long

        passing = self._condition_tree_pairs(pairs_long, condition)

        if not passing:
            return pairs_long.clear()

        a_list, b_list = zip(*passing)
        keep = pl.DataFrame({"a": list(a_list), "b": list(b_list)})
        return pairs_long.join(keep, on=["a", "b"], how="inner")

    def _condition_tree_pairs(
        self, pairs_long: pl.DataFrame, condition: Dict
    ) -> Set[Tuple[str, str]]:
        """Recursively evaluate an AND/OR condition tree against pairs_long."""
        if "and" in condition:
            sets = [
                self._condition_tree_pairs(pairs_long, sub)
                for sub in condition["and"]
            ]
            return set.intersection(*sets) if sets else set()
        if "or" in condition:
            sets = [
                self._condition_tree_pairs(pairs_long, sub)
                for sub in condition["or"]
            ]
            return set.union(*sets) if sets else set()
        # Leaf: match if any signal in this leaf contributed a pair.
        # Persisted matches retain the original signal but use
        # alias_type="PREDEFINED" when rehydrated.
        signal_types = list(condition.keys())
        matching = pairs_long.filter(pl.col("signal").is_in(signal_types))
        if matching.height == 0:
            return set()
        return set(zip(matching["a"].to_list(), matching["b"].to_list()))

    def _score_pairs(
        self, pairs_long: pl.DataFrame, alias_index: pl.DataFrame
    ) -> pl.DataFrame:
        """Score pairs based on IDF and weights."""
        if pairs_long.height == 0:
            return pl.DataFrame(
                schema={
                    "a": pl.Utf8,
                    "b": pl.Utf8,
                    "score": pl.Float64,
                    "signals": pl.List(pl.Utf8),
                }
            )

        weight_map = {t: self._specs[t]["weight"] for t in self._specs}
        bonus_map = {t: self._specs[t]["bonus"] for t in self._specs}

        tmp = (
            pairs_long.join(
                alias_index.select("alias_key", "alias_type", "idf"),
                on=["alias_key", "alias_type"],
                how="left",
            )
            .with_columns(
                idf=pl.col("idf").fill_null(0.0),
                w=pl.col("signal").replace_strict(
                    weight_map, default=1.0, return_dtype=pl.Float64
                ),
                sim=pl.when(pl.col("sim").is_null()).then(1.0).otherwise(pl.col("sim")),
            )
            .with_columns(part_score=pl.col("idf") * pl.col("w") * pl.col("sim"))
        )

        agg_score = tmp.group_by(["a", "b"]).agg(
            score=pl.col("part_score").sum(), signals=pl.col("signal").unique().sort()
        )

        pres = pairs_long.select("a", "b", "signal").unique()
        pres = pres.with_columns(
            bonus=pl.col("signal").replace_strict(
                bonus_map, default=0.0, return_dtype=pl.Float64
            )
        )
        bonus = pres.group_by(["a", "b"]).agg(bonus=pl.col("bonus").sum())

        s = (
            agg_score.join(bonus, on=["a", "b"], how="left")
            .with_columns(score=pl.col("score") + pl.col("bonus").fill_null(0.0))
            .drop("bonus")
        )

        return s

    def _select_pairs(
        self, pairs_scored: pl.DataFrame, threshold: Optional[float], quantile: float
    ) -> pl.DataFrame:
        """Select pairs based on threshold or quantile."""
        if pairs_scored.height == 0:
            return pairs_scored
        if threshold is None:
            q = float(pairs_scored.select(pl.col("score").quantile(quantile)).item())
            t = q
        else:
            t = threshold
        return pairs_scored.filter(pl.col("score") >= t)

    @staticmethod
    def _to_polars(df: RecordFrame) -> pl.DataFrame:
        """Convert to Polars DataFrame."""
        try:
            import pandas as pd

            if isinstance(df, pd.DataFrame):
                return pl.from_pandas(df)
        except Exception:
            pass
        if isinstance(df, pl.DataFrame):
            return df
        raise TypeError("records must be a polars.DataFrame or pandas.DataFrame")


# ─────────────────────────────────────────────────────────────────────────────
# Dashboard HTML template
# Data placeholders (__NODES__, __EDGES__, etc.) are replaced at render time.
# ─────────────────────────────────────────────────────────────────────────────
_DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Entity Cluster Dashboard</title>
  <script src="https://unpkg.com/vis-network@9.1.9/standalone/umd/vis-network.min.js"></script>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
           background: #0f1117; color: #e2e8f0; height: 100vh;
           display: flex; flex-direction: column; overflow: hidden; }

    /* ── Header ── */
    #hdr { background: #1a1f2e; border-bottom: 1px solid #2d3748;
           padding: 0 20px; height: 48px; flex-shrink: 0;
           display: flex; align-items: center; justify-content: space-between; }
    #hdr h1 { font-size: 14px; font-weight: 600; color: #e2e8f0; letter-spacing: .3px; }
    .stats  { display: flex; gap: 22px; }
    .stat   { font-size: 12px; color: #718096; }
    .stat strong { color: #7dd3fc; }

    /* ── Body ── */
    #body { flex: 1; display: flex; overflow: hidden; }

    /* ── Graph panel ── */
    #gp { flex: 0 0 52%; position: relative; background: #0f1117; min-width: 20%; }
    #net { width: 100%; height: 100%; }

    /* ── Resize handle ── */
    #rzr { width: 5px; cursor: col-resize; background: #2d3748; flex-shrink: 0;
           transition: background .15s; }
    #rzr:hover, #rzr.drag { background: #4a90d9; }

    /* ── Sidebar ── */
    #sb { flex: 0 0 48%; display: flex; flex-direction: column;
          background: #1a1f2e; border-left: none; overflow: hidden; min-width: 20%; }

    /* info card */
    #ic { padding: 10px 16px; border-bottom: 1px solid #2d3748; flex-shrink: 0; }
    .lbl   { font-size: 10px; text-transform: uppercase; letter-spacing: 1px;
             color: #4a5568; margin-bottom: 5px; }
    #ct    { font-size: 16px; font-weight: 700; color: #4a5568; }
    #cs    { font-size: 12px; color: #4a5568; margin-top: 2px; }
    .dot   { display: inline-block; width: 9px; height: 9px; border-radius: 50%;
             margin-right: 6px; vertical-align: middle; flex-shrink: 0; }

    /* cluster list */
    #cl { border-bottom: 1px solid #2d3748; overflow-y: auto; max-height: 220px; flex-shrink: 0; }
    .ci { display: flex; align-items: center; gap: 8px; padding: 7px 16px;
          cursor: pointer; font-size: 12px; color: #a0aec0;
          border-bottom: 1px solid #ffffff08; }
    .ci:hover  { background: #252d40; }
    .ci.active { background: #1e3358; color: #7dd3fc; }
    .ci-id   { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .ci-sz   { font-size: 11px; color: #4a5568; flex-shrink: 0; }
    .ci.active .ci-sz { color: #4a90d9; }

    /* table */
    #ta { flex: 1; overflow-y: auto; }
    table { width: 100%; border-collapse: collapse; font-size: 12px; }
    thead { background: #131720; position: sticky; top: 0; z-index: 2; }
    th { padding: 8px 12px; text-align: left; font-weight: 600; color: #4a5568;
         font-size: 10px; text-transform: uppercase; letter-spacing: .6px;
         border-bottom: 1px solid #2d3748; white-space: nowrap; }
    td { padding: 7px 12px; border-bottom: 1px solid #1a2035; color: #cbd5e1;
         max-width: 200px; overflow: hidden; text-overflow: ellipsis;
         white-space: nowrap; cursor: pointer; }
    tbody tr:hover td { background: #202940; }
    tbody tr.sel td   { background: #1b3460; color: #93c5fd; }

    .hint { display: flex; flex-direction: column; align-items: center;
            justify-content: center; height: 100%; padding: 32px;
            text-align: center; color: #2d3748; pointer-events: none; }
    .hint p { font-size: 13px; line-height: 1.7; }
    .hint p strong { display: block; font-size: 14px; margin-bottom: 6px; color: #3d4f6b; }
  </style>
</head>
<body>
  <div id="hdr">
    <h1>&#11041; Entity Cluster Dashboard</h1>
    <div class="stats" id="statsbar"></div>
  </div>
  <div id="body">
    <div id="gp"><div id="net"></div></div>
    <div id="rzr"></div>
    <div id="sb">
      <div id="ic">
        <div class="lbl">Selected Cluster</div>
        <div id="ct">None</div>
        <div id="cs">Click a node or choose a cluster below</div>
      </div>
      <div id="cl"></div>
      <div id="ta">
        <div class="hint">
          <p><strong>No cluster selected</strong>
          Click any node in the graph, or pick a cluster from the list above.</p>
        </div>
      </div>
    </div>
  </div>

  <script>
    // ── injected data ──────────────────────────────────────────────────────────
    const ND = __NODES__;
    const ED = __EDGES__;
    const TD = __TABLE_DATA__;
    const FL = __FIELDS__;
    const ST = __STATS__;
    const CL = __CLUSTER_LIST__;

    // ── stats bar ──────────────────────────────────────────────────────────────
    const extra = ST.clusters_total > ST.clusters_shown
      ? ` <span style="color:#4a5568">of ${ST.clusters_total}</span>` : "";
    document.getElementById("statsbar").innerHTML = [
      ["Clusters", ST.clusters_shown + extra],
      ["Records",  ST.records_shown],
      ["Edges",    ST.edges_shown],
      ["Largest",  ST.largest + " records"],
    ].map(([k,v]) => `<div class="stat">${k}: <strong>${v}</strong></div>`).join("");

    // ── vis.js ─────────────────────────────────────────────────────────────────
    const nodeIdx = Object.fromEntries(ND.map(n => [n.id, n]));

    // Pre-index nodes and edges by cluster so we can swap them in instantly
    const nodesByCluster = {};
    ND.forEach(n => {
      (nodesByCluster[n.cluster_id] = nodesByCluster[n.cluster_id] || []).push(n);
    });
    const edgesByCluster = {};
    ED.forEach(e => {
      const cid = nodeIdx[e.from]?.cluster_id;
      if (!cid) return;
      (edgesByCluster[cid] = edgesByCluster[cid] || []).push(e);
    });

    // Start with an empty graph — clusters are loaded on demand
    const nodeset = new vis.DataSet([]);
    const edgeset = new vis.DataSet([]);
    const network = new vis.Network(document.getElementById("net"),
      { nodes: nodeset, edges: edgeset },
      {
        physics: {
          enabled: false,
          stabilization: { enabled: true, iterations: 150, updateInterval: 20 },
          forceAtlas2Based: {
            gravitationalConstant: -52, centralGravity: 0.004,
            springLength: 100, springConstant: 0.07, damping: 0.42, avoidOverlap: 0.4,
          },
          solver: "forceAtlas2Based",
        },
        nodes: {
          shape: "dot", size: 11,
          font: { size: 11, face: "-apple-system, Arial", color: "#e2e8f0" },
          borderWidth: 2,
        },
        edges: {
          smooth: { type: "continuous", roundness: 0.15 },
          scaling:  { min: 1, max: 5 },
          color:    { inherit: false },
          hoverWidth: 2,
        },
        interaction: {
          hover: true, tooltipDelay: 80,
          hideEdgesOnDrag: true, zoomView: true,
        },
      }
    );

    // Disable physics after each layout run
    network.on("stabilizationIterationsDone", () =>
      network.setOptions({ physics: { enabled: false } }));

    // ── state ──────────────────────────────────────────────────────────────────
    let activeCid = null;

    // ── helpers ────────────────────────────────────────────────────────────────
    function esc(s) {
      return String(s)
        .replace(/&/g,"&amp;").replace(/</g,"&lt;")
        .replace(/>/g,"&gt;").replace(/"/g,"&quot;");
    }

    function renderTable(cid, focusNodeId) {
      const rows = TD[cid] || [];
      const area = document.getElementById("ta");
      if (!rows.length) {
        area.innerHTML = "<div class='hint'><p>No record data for this cluster.</p></div>";
        return;
      }
      let h = "<table><thead><tr>"
        + FL.map(f => `<th>${esc(f)}</th>`).join("")
        + "</tr></thead><tbody>";
      rows.forEach((row, i) => {
        h += `<tr data-i="${i}">`;
        FL.forEach(f => {
          const v = row[f] != null ? String(row[f]) : "";
          const s = v.length > 40 ? v.slice(0,40) + "…" : v;
          h += `<td title="${esc(v)}">${esc(s)}</td>`;
        });
        h += "</tr>";
      });
      h += "</tbody></table>";
      area.innerHTML = h;

      // row click → focus node
      area.querySelectorAll("tbody tr").forEach(tr => {
        tr.addEventListener("click", () => {
          area.querySelectorAll("tbody tr.sel").forEach(r => r.classList.remove("sel"));
          tr.classList.add("sel");
          const rid = rows[+tr.dataset.i][FL[0]];
          network.selectNodes([rid]);
          network.focus(rid, { scale: 1.6,
            animation: { duration: 350, easingFunction: "easeInOutQuad" } });
        });
      });

      // node click → highlight matching row
      if (focusNodeId != null) {
        const idx = rows.findIndex(r => String(r[FL[0]]) === String(focusNodeId));
        if (idx >= 0) {
          const tr = area.querySelectorAll("tbody tr")[idx];
          if (tr) {
            tr.classList.add("sel");
            tr.scrollIntoView({ block: "nearest" });
          }
        }
      }
    }

    function selectCluster(cid, color, focusNodeId) {
      const switching = cid !== activeCid;
      activeCid = cid;

      if (switching) {
        // Swap in only this cluster's nodes + edges
        nodeset.clear();
        edgeset.clear();
        nodeset.add(nodesByCluster[cid] || []);
        edgeset.add(edgesByCluster[cid] || []);
        network.setOptions({ physics: { enabled: true } });
        network.stabilize(150);
      }

      // Keep / restore node selection in the graph
      if (focusNodeId != null) {
        network.selectNodes([String(focusNodeId)]);
        if (!switching) {
          network.focus(String(focusNodeId), {
            scale: 1.4, animation: { duration: 300, easingFunction: "easeInOutQuad" }
          });
        }
      }

      if (switching) {
        // info card
        const ct = document.getElementById("ct");
        ct.innerHTML = `<span class="dot" style="background:${color}"></span>${esc(cid)}`;
        ct.style.color = "";
        const rows = TD[cid] || [];
        const cs = document.getElementById("cs");
        cs.textContent = `${rows.length} record${rows.length !== 1 ? "s" : ""}`;
        cs.style.color = "";

        // cluster list highlight
        document.querySelectorAll(".ci").forEach(el =>
          el.classList.toggle("active", el.dataset.cid === cid));
      }

      renderTable(cid, focusNodeId);
    }

    function clearSelection() {
      activeCid = null;
      nodeset.clear();
      edgeset.clear();
      const ct = document.getElementById("ct");
      ct.textContent = "None"; ct.style.color = "#4a5568";
      const cs = document.getElementById("cs");
      cs.textContent = "Click a node or choose a cluster below"; cs.style.color = "#4a5568";
      document.querySelectorAll(".ci").forEach(el => el.classList.remove("active"));
      document.getElementById("ta").innerHTML =
        "<div class='hint'><p><strong>No cluster selected</strong>" +
        "Click any node in the graph, or pick a cluster from the list above.</p></div>";
    }

    // ── network events ─────────────────────────────────────────────────────────
    network.on("click", params => {
      if (params.nodes.length) {
        const node = nodeIdx[params.nodes[0]];
        if (node) {
          const entry = CL.find(c => c.id === node.cluster_id);
          selectCluster(node.cluster_id, entry ? entry.color : "#97c2fc", node.id);
        }
      } else {
        clearSelection();
      }
    });

    // ── cluster list ───────────────────────────────────────────────────────────
    const clEl = document.getElementById("cl");
    clEl.innerHTML = CL.map(c =>
      `<div class="ci" data-cid="${esc(c.id)}">
         <span class="dot" style="background:${c.color}"></span>
         <span class="ci-id">${esc(c.id)}</span>
         <span class="ci-sz">${c.size} records</span>
       </div>`
    ).join("");

    clEl.querySelectorAll(".ci").forEach(el => {
      el.addEventListener("click", () => {
        const cid = el.dataset.cid;
        const entry = CL.find(c => c.id === cid);
        selectCluster(cid, entry ? entry.color : "#97c2fc");
      });
    });

    // ── resizable split ────────────────────────────────────────────────────────
    const rzr = document.getElementById("rzr");
    const gp  = document.getElementById("gp");
    const sb  = document.getElementById("sb");
    const bod = document.getElementById("body");
    let resizing = false;

    rzr.addEventListener("mousedown", e => {
      resizing = true;
      rzr.classList.add("drag");
      document.body.style.cursor = "col-resize";
      document.body.style.userSelect = "none";
      e.preventDefault();
    });

    document.addEventListener("mousemove", e => {
      if (!resizing) return;
      const rect  = bod.getBoundingClientRect();
      const pct   = Math.min(Math.max((e.clientX - rect.left) / rect.width * 100, 20), 80);
      gp.style.flex = `0 0 ${pct}%`;
      sb.style.flex = `0 0 ${100 - pct}%`;
      network.redraw();
    });

    document.addEventListener("mouseup", () => {
      if (!resizing) return;
      resizing = false;
      rzr.classList.remove("drag");
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
      network.redraw();
    });
  </script>
</body>
</html>"""


# Example usage
if __name__ == "__main__":
    import pandas as pd

    # First batch of records
    records_batch1 = pd.DataFrame(
        [
            {
                "record_id": "R1",
                "name_key": "ACMSPPLY",
                "zip5": "10011",
                "phone": "+12125550101",
                "geo": "NY|100",
            },
            {
                "record_id": "R2",
                "name_key": "ACMSPPLY",
                "zip5": "10011",
                "phone": "+12125550101",
                "geo": "NY|100",
            },
            {
                "record_id": "R3",
                "name_key": "ACMSPPLY",
                "zip5": "10012",
                "phone": "+12125550101",
                "geo": "NY|100",
            },
        ]
    )

    # Second batch with overlapping records
    records_batch2 = pd.DataFrame(
        [
            {
                "record_id": "R4",
                "name_key": "GLOBEXLLC",
                "zip5": "94107",
                "phone": "+14155550123",
                "geo": "CA|941",
            },
            {
                "record_id": "R5",
                "name_key": "GLOBEXLTD",
                "zip5": "94107",
                "phone": "+14155550123",
                "geo": "CA|941",
            },
            {
                "record_id": "R6",  # Same as batch 1
                "name_key": "ACMESUPPLY",
                "zip5": "10012",
                "phone": "+12125550101",
                "geo": "NY|100",
            },
        ]
    )

    blocking_plan = {
        "PHONE": {
            "fields": ["phone"],
            "weight": 1.0,
            "bonus": 0.5,
            "df_cap": 50000,
        },
        "NAME_KEY_COSINE": {
            "fields": ["name_key"],
            "block_on": ["zip5"],
            "weight": 0.95,
            "df_cap": 20000,
            "similarity": {
                "type": "cosine",
                "field": "name_key",
                "threshold": 0.25,
                "analyzer": "char_wb",
                "ngram_range": (2, 3),
                "min_df": 1,
            },
        },
    }

    # Initialize incremental linker
    linker = IncrementalSignalLinker(
        record_id_col="record_id",
        blocking_plan=blocking_plan,
        include_types=["NAME_KEY_COSINE", "PHONE"],
        k_required=1,
        select_threshold=None,
        select_quantile=0.5,
        enable_incremental=True,
        matches_file="matches.json",
        create_network=True,
        return_intermediates=True,
    )

    # Process first batch
    print("Processing Batch 1...")
    result1 = linker.link_incremental(records_batch1, batch_id="batch1")
    print(f"Clusters after batch 1: {result1['clusters']}")
    print(f"Stats: {result1['cluster_stats']}")

    # Process second batch
    print("\nProcessing Batch 2...")
    result2 = linker.link_incremental(records_batch2, batch_id="batch2")
    print(f"Clusters after batch 2: {result2['clusters']}")
    print(f"Stats: {result2['cluster_stats']}")

    # all_records = pd.concat([records_batch1, records_batch2])

    # Process as single batch
    # result = linker.link_incremental(all_records, batch_id="combined")

    # Analyze cluster connectivity
    for cluster_id in linker._clusters.keys():
        analysis = linker.analyze_cluster_connectivity(cluster_id)
        print(f"\nCluster {cluster_id} analysis:")
        print(f"  Size: {analysis['size']}, Edges: {analysis['edges']}")
        print(f"  Density: {analysis['density']:.3f}")
        print(f"  Signal distribution: {analysis['signal_distribution']}")

        # Create evolution visualization
        viz_path = linker.visualize_cluster_evolution(cluster_id)
        print(f"  Visualization saved to: {viz_path}")
    pdb.set_trace()
