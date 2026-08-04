# entity_fusion_spark.py
from __future__ import annotations

from dataclasses import dataclass, field
from functools import reduce
from typing import Any, Dict, List, Optional, Tuple, Union

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.storagelevel import StorageLevel


_NULL_SENTINEL = "~~NULL~~"

_MANUAL_DECISION_ALIASES = {
    "must_link": "must_link",
    "must-link": "must_link",
    "link": "must_link",
    "same": "must_link",
    "same_entity": "must_link",
    "true_positive": "must_link",
    "true-positive": "must_link",
    "tp": "must_link",
    "cannot_link": "cannot_link",
    "cannot-link": "cannot_link",
    "must_not_link": "cannot_link",
    "must-not-link": "cannot_link",
    "do_not_link": "cannot_link",
    "do-not-link": "cannot_link",
    "unlink": "cannot_link",
    "different": "cannot_link",
    "different_entity": "cannot_link",
    "false_positive": "cannot_link",
    "false-positive": "cannot_link",
    "fp": "cannot_link",
}


@dataclass
class EntityFusionSpark:
    """
    Exact-match entity clustering on PySpark 3 DataFrames.

    This is the Spark sibling of EntityFusionPolars: one-shot, exact-match only,
    no fuzzy scoring and no incremental persistence. It builds candidate edges from
    shared identifier values, enforces K-of-N agreement, applies persistent manual
    review inputs, runs connected components, and returns Spark DataFrames.

    `manual_identifier_clusters` is the durable review layer for identifier aliases:
    rows sharing a `manual_cluster_id` are treated as equivalent everywhere they
    appear, and the same value becomes the output `stable_cluster_id`.
    """

    spark: SparkSession
    record_id_col: str
    match_columns: List[Union[str, Dict[str, Any]]]
    k_required: int = 1
    block_cap: Optional[int] = None
    max_cc_iters: int = 30
    small_graph_collect_limit: int = 10000

    _columns: List[str] = field(default_factory=list, init=False)
    _block_on: Dict[str, List[str]] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        for entry in self.match_columns:
            if isinstance(entry, str):
                self._columns.append(entry)
            elif isinstance(entry, dict):
                col = entry.get("column")
                if not col:
                    raise ValueError(f"match_columns dict entry missing 'column': {entry}")
                self._columns.append(col)
                block_on = entry.get("block_on")
                if block_on:
                    if isinstance(block_on, str):
                        raise ValueError(
                            f"block_on for {col!r} must be a list of column names, not a string"
                        )
                    self._block_on[col] = list(block_on)
            else:
                raise ValueError(
                    f"match_columns entries must be str or dict, got {type(entry)}: {entry}"
                )

        if not self._columns:
            raise ValueError("match_columns must contain at least one column")
        if len(set(self._columns)) != len(self._columns):
            raise ValueError(f"match_columns has duplicate columns: {self._columns}")
        if self.k_required < 1:
            raise ValueError("k_required must be >= 1")
        if self.k_required > len(self._columns):
            raise ValueError(
                f"k_required ({self.k_required}) cannot exceed the number of "
                f"match_columns ({len(self._columns)})"
            )

    # ----------------------------- Public API -----------------------------

    def cluster(
        self,
        df: DataFrame,
        manual_links: Optional[DataFrame] = None,
        manual_decisions: Optional[DataFrame] = None,
        manual_identifier_clusters: Optional[DataFrame] = None,
    ) -> Dict[str, DataFrame]:
        """Cluster `df` and return Spark DataFrames.

        Returns:
          - records: original columns plus hash_id, cluster_id, stable_cluster_id
          - cluster_stats
          - edge_signals
          - manual_edges
          - manual_cannot_link_edges
          - manual_decisions
          - manual_identifier_edges
          - manual_identifier_clusters
          - stable_cluster_conflicts
          - manual_conflicts
        """
        self._validate_input(df)

        fingerprints = self._fingerprint(df).persist(StorageLevel.MEMORY_AND_DISK)
        nodes = fingerprints.select("hash_id").dropDuplicates()

        blocked_values = self._blocked_values(fingerprints)
        block_sizes = self._block_sizes(blocked_values)
        candidate_pairs = self._candidate_pairs(blocked_values, block_sizes)
        _, edge_signals = self._k_of_n_filter(candidate_pairs)

        manual_edges, manual_cannot_link_edges, manual_decisions_resolved = (
            self._resolve_manual_decisions(manual_decisions, manual_links, fingerprints)
        )
        manual_identifier_edges, manual_identifier_clusters_resolved = (
            self._resolve_manual_identifier_clusters(
                df, fingerprints, manual_identifier_clusters
            )
        )

        manual_edges = self._union_all([manual_edges, manual_identifier_edges]).dropDuplicates()
        edge_signals = self._drop_cannot_link_edges(edge_signals, manual_cannot_link_edges)

        all_edges = self._union_all(
            [edge_signals.select("a", "b"), manual_edges.select("a", "b")]
        ).dropDuplicates()
        cluster_map = self._connected_components(nodes, all_edges)
        cluster_map, stable_cluster_conflicts = self._stable_cluster_ids(
            cluster_map, manual_identifier_clusters_resolved
        )

        records = (
            df.join(
                fingerprints.select(self.record_id_col, "hash_id"),
                on=self.record_id_col,
                how="left",
            )
            .join(cluster_map, on="hash_id", how="left")
        )

        stats = self._cluster_stats(
            records, cluster_map, edge_signals, manual_edges, fingerprints
        )
        manual_conflicts = self._manual_conflicts(manual_decisions_resolved, records)

        return {
            "records": records,
            "cluster_stats": stats,
            "edge_signals": edge_signals,
            "manual_edges": manual_edges,
            "manual_cannot_link_edges": manual_cannot_link_edges,
            "manual_decisions": manual_decisions_resolved,
            "manual_identifier_edges": manual_identifier_edges,
            "manual_identifier_clusters": manual_identifier_clusters_resolved,
            "stable_cluster_conflicts": stable_cluster_conflicts,
            "manual_conflicts": manual_conflicts,
        }

    def preview_blocks(self, df: DataFrame) -> DataFrame:
        """Return per-identifier block sizes before pairing."""
        self._validate_input(df)
        fingerprints = self._fingerprint(df)
        return self._block_sizes(self._blocked_values(fingerprints))

    # ----------------------------- Schemas / helpers -----------------------------

    @staticmethod
    def _df_is_empty(df: DataFrame) -> bool:
        return df.limit(1).count() == 0

    def _empty_df(self, schema: T.StructType) -> DataFrame:
        return self.spark.createDataFrame([], schema)

    def _empty_edge_frame(self) -> DataFrame:
        return self._empty_df(
            T.StructType(
                [
                    T.StructField("a", T.StringType(), False),
                    T.StructField("b", T.StringType(), False),
                ]
            )
        )

    def _empty_pairs_frame(self) -> DataFrame:
        return self._empty_df(
            T.StructType(
                [
                    T.StructField("a", T.StringType(), False),
                    T.StructField("b", T.StringType(), False),
                    T.StructField("signal", T.StringType(), False),
                ]
            )
        )

    def _empty_edge_signals_frame(self) -> DataFrame:
        return self._empty_df(
            T.StructType(
                [
                    T.StructField("a", T.StringType(), False),
                    T.StructField("b", T.StringType(), False),
                    T.StructField("n_signals", T.LongType(), False),
                ]
            )
        )

    def _empty_manual_decisions_frame(self) -> DataFrame:
        return self._empty_df(
            T.StructType(
                [
                    T.StructField("record_id_a", T.StringType(), True),
                    T.StructField("record_id_b", T.StringType(), True),
                    T.StructField("decision", T.StringType(), True),
                    T.StructField("hash_id_a", T.StringType(), True),
                    T.StructField("hash_id_b", T.StringType(), True),
                    T.StructField("a", T.StringType(), True),
                    T.StructField("b", T.StringType(), True),
                ]
            )
        )

    def _empty_manual_identifier_clusters_frame(self) -> DataFrame:
        return self._empty_df(
            T.StructType(
                [
                    T.StructField("manual_cluster_id", T.StringType(), True),
                    T.StructField("field", T.StringType(), True),
                    T.StructField("value", T.StringType(), True),
                    T.StructField("hash_id", T.StringType(), True),
                ]
            )
        )

    def _empty_manual_conflicts_frame(self) -> DataFrame:
        return self._empty_df(
            T.StructType(
                [
                    T.StructField("record_id_a", T.StringType(), True),
                    T.StructField("record_id_b", T.StringType(), True),
                    T.StructField("decision", T.StringType(), True),
                    T.StructField("hash_id_a", T.StringType(), True),
                    T.StructField("hash_id_b", T.StringType(), True),
                    T.StructField("a", T.StringType(), True),
                    T.StructField("b", T.StringType(), True),
                    T.StructField("cluster_id", T.StringType(), True),
                    T.StructField("conflict_type", T.StringType(), True),
                ]
            )
        )

    def _empty_stable_cluster_conflicts_frame(self) -> DataFrame:
        return self._empty_df(
            T.StructType(
                [
                    T.StructField("cluster_id", T.StringType(), True),
                    T.StructField("stable_cluster_id", T.StringType(), True),
                    T.StructField("auto_stable_cluster_id", T.StringType(), True),
                    T.StructField("n_hash_ids", T.LongType(), True),
                    T.StructField("n_manual_cluster_ids", T.LongType(), True),
                    T.StructField(
                        "manual_cluster_ids", T.ArrayType(T.StringType()), True
                    ),
                ]
            )
        )

    @staticmethod
    def _union_all(frames: List[DataFrame]) -> DataFrame:
        if not frames:
            raise ValueError("_union_all requires at least one dataframe")
        return reduce(lambda a, b: a.unionByName(b, allowMissingColumns=True), frames)

    @staticmethod
    def _alias_map_expr() -> Any:
        pairs: List[Any] = []
        for key, value in _MANUAL_DECISION_ALIASES.items():
            pairs.extend([F.lit(key), F.lit(value)])
        return F.create_map(*pairs)

    @staticmethod
    def _manual_decision_key(col: Any) -> Any:
        return F.lower(F.regexp_replace(F.trim(col.cast("string")), " +", "_"))

    @staticmethod
    def _sorted_values(values: List[Any]) -> List[Any]:
        return sorted(values, key=lambda value: (type(value).__name__, str(value)))

    def _extra_block_on_columns(self) -> List[str]:
        seen = set(self._columns)
        extra: List[str] = []
        for cols in self._block_on.values():
            for col in cols:
                if col not in seen:
                    seen.add(col)
                    extra.append(col)
        return extra

    # ----------------------------- Input / fingerprinting -----------------------------

    def _validate_input(self, df: DataFrame) -> None:
        required = [self.record_id_col, *self._columns, *self._extra_block_on_columns()]
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"df is missing required columns: {missing}")

        if df.where(F.col(self.record_id_col).isNull()).limit(1).count() > 0:
            raise ValueError(f"{self.record_id_col!r} must be a non-null primary key")

        duplicate_ids = (
            df.groupBy(self.record_id_col)
            .count()
            .where(F.col("count") > 1)
            .limit(10)
            .collect()
        )
        if duplicate_ids:
            examples = [row[self.record_id_col] for row in duplicate_ids]
            raise ValueError(
                f"{self.record_id_col!r} must be unique; duplicate values found: "
                f"{self._sorted_values(examples)}"
            )

    def _fingerprint(self, df: DataFrame) -> DataFrame:
        hash_fields = [
            F.coalesce(F.col(col).cast("string"), F.lit(_NULL_SENTINEL)).alias(col)
            for col in self._columns
        ]
        hash_expr = F.sha2(F.to_json(F.struct(*hash_fields)), 256)
        return df.select(
            self.record_id_col,
            *self._columns,
            *self._extra_block_on_columns(),
            hash_expr.alias("hash_id"),
        )

    # ----------------------------- Candidate generation -----------------------------

    def _blocked_values(self, fingerprints: DataFrame) -> Dict[str, DataFrame]:
        tables: Dict[str, DataFrame] = {}
        for col in self._columns:
            block_cols = self._block_on.get(col, [])
            if block_cols:
                block_expr = F.concat_ws(
                    "|",
                    *[
                        F.coalesce(F.col(c).cast("string"), F.lit(_NULL_SENTINEL))
                        for c in block_cols
                    ],
                )
            else:
                block_expr = F.lit("")

            tables[col] = (
                fingerprints.select(
                    F.col("hash_id"),
                    F.col(col).cast("string").alias("value"),
                    block_expr.alias("block"),
                )
                .where(F.col("value").isNotNull())
                .where(F.length(F.trim(F.col("value"))) > 0)
                .dropDuplicates()
            )
        return tables

    def _block_sizes(self, blocked_values: Dict[str, DataFrame]) -> DataFrame:
        parts: List[DataFrame] = []
        for col, table in blocked_values.items():
            parts.append(
                table.groupBy("block", "value")
                .agg(F.countDistinct("hash_id").alias("n_hash_ids"))
                .withColumn("match_column", F.lit(col))
                .select("match_column", "block", "value", "n_hash_ids")
            )

        if not parts:
            return self._empty_df(
                T.StructType(
                    [
                        T.StructField("match_column", T.StringType(), True),
                        T.StructField("block", T.StringType(), True),
                        T.StructField("value", T.StringType(), True),
                        T.StructField("n_hash_ids", T.LongType(), True),
                    ]
                )
            )

        return self._union_all(parts).orderBy(F.col("n_hash_ids").desc())

    def _candidate_pairs(
        self, blocked_values: Dict[str, DataFrame], block_sizes: DataFrame
    ) -> DataFrame:
        parts: List[DataFrame] = []
        for col, table in blocked_values.items():
            tt = table
            if self.block_cap is not None:
                oversized = (
                    block_sizes.where(
                        (F.col("match_column") == F.lit(col))
                        & (F.col("n_hash_ids") > F.lit(int(self.block_cap)))
                    )
                    .select("block", "value")
                    .dropDuplicates()
                )
                tt = tt.join(oversized, on=["block", "value"], how="left_anti")

            left = tt.select(
                F.col("block"),
                F.col("value"),
                F.col("hash_id").alias("a"),
            )
            right = tt.select(
                F.col("block"),
                F.col("value"),
                F.col("hash_id").alias("b"),
            )
            parts.append(
                left.join(right, on=["block", "value"], how="inner")
                .where(F.col("a") < F.col("b"))
                .select("a", "b", F.lit(col).alias("signal"))
                .dropDuplicates()
            )

        if not parts:
            return self._empty_pairs_frame()
        return self._union_all(parts)

    def _k_of_n_filter(self, pairs: DataFrame) -> Tuple[DataFrame, DataFrame]:
        if self._df_is_empty(pairs):
            return pairs, self._empty_edge_signals_frame()

        edge_signals = (
            pairs.groupBy("a", "b")
            .agg(F.countDistinct("signal").alias("n_signals"))
            .where(F.col("n_signals") >= F.lit(int(self.k_required)))
        )
        filtered_pairs = pairs.join(edge_signals.select("a", "b"), on=["a", "b"], how="inner")
        return filtered_pairs, edge_signals

    # ----------------------------- Manual review inputs -----------------------------

    def _resolve_manual_links(
        self, manual_links: Optional[DataFrame], fingerprints: DataFrame
    ) -> DataFrame:
        must_link_edges, _, _ = self._resolve_manual_decisions(None, manual_links, fingerprints)
        return must_link_edges

    def _manual_decision_rows(
        self,
        manual_decisions: Optional[DataFrame],
        manual_links: Optional[DataFrame],
    ) -> DataFrame:
        frames: List[DataFrame] = []

        if manual_decisions is not None:
            missing_cols = {"record_id_a", "record_id_b", "decision"} - set(
                manual_decisions.columns
            )
            if missing_cols:
                raise ValueError(
                    f"manual_decisions is missing columns: {sorted(missing_cols)}"
                )
            active = manual_decisions
            if "active" in active.columns:
                active = active.where(F.col("active") == F.lit(True))
            frames.append(
                active.select(
                    F.col("record_id_a").cast("string").alias("record_id_a"),
                    F.col("record_id_b").cast("string").alias("record_id_b"),
                    F.col("decision").cast("string").alias("decision_raw"),
                )
            )

        if manual_links is not None:
            missing_cols = {"record_id_a", "record_id_b"} - set(manual_links.columns)
            if missing_cols:
                raise ValueError(f"manual_links is missing columns: {sorted(missing_cols)}")
            frames.append(
                manual_links.select(
                    F.col("record_id_a").cast("string").alias("record_id_a"),
                    F.col("record_id_b").cast("string").alias("record_id_b"),
                    F.lit("must_link").alias("decision_raw"),
                )
            )

        if not frames:
            return self._empty_df(
                T.StructType(
                    [
                        T.StructField("record_id_a", T.StringType(), True),
                        T.StructField("record_id_b", T.StringType(), True),
                        T.StructField("decision_raw", T.StringType(), True),
                    ]
                )
            )
        return self._union_all(frames)

    def _resolve_manual_decisions(
        self,
        manual_decisions: Optional[DataFrame],
        manual_links: Optional[DataFrame],
        fingerprints: DataFrame,
    ) -> Tuple[DataFrame, DataFrame, DataFrame]:
        decisions_raw = self._manual_decision_rows(manual_decisions, manual_links)
        if self._df_is_empty(decisions_raw):
            return (
                self._empty_edge_frame(),
                self._empty_edge_frame(),
                self._empty_manual_decisions_frame(),
            )

        decisions = decisions_raw.withColumn(
            "decision",
            F.element_at(
                self._alias_map_expr(), self._manual_decision_key(F.col("decision_raw"))
            ),
        )

        null_refs = decisions.where(
            F.col("record_id_a").isNull() | F.col("record_id_b").isNull()
        ).limit(10).collect()
        if null_refs:
            raise ValueError("manual decisions cannot reference null record_ids")

        self_refs = decisions.where(F.col("record_id_a") == F.col("record_id_b")).limit(10).collect()
        if self_refs:
            examples = [(row["record_id_a"], row["record_id_b"]) for row in self_refs]
            raise ValueError(f"manual decisions cannot reference the same record twice: {examples}")

        invalid = decisions.where(F.col("decision").isNull()).limit(10).collect()
        if invalid:
            examples = [row["decision_raw"] for row in invalid]
            raise ValueError(
                "Unsupported manual decision values: "
                f"{examples}; expected must_link/cannot_link or aliases"
            )

        id_to_hash = fingerprints.select(
            F.col(self.record_id_col).cast("string").alias("record_id"),
            F.col("hash_id"),
        )

        missing_a = (
            decisions.select(F.col("record_id_a").alias("record_id"))
            .join(id_to_hash, on="record_id", how="left_anti")
            .limit(10)
            .collect()
        )
        missing_b = (
            decisions.select(F.col("record_id_b").alias("record_id"))
            .join(id_to_hash, on="record_id", how="left_anti")
            .limit(10)
            .collect()
        )
        if missing_a or missing_b:
            missing = {row["record_id"] for row in [*missing_a, *missing_b]}
            raise ValueError(
                "manual decisions reference record_ids not in df: "
                f"{self._sorted_values(list(missing))}"
            )

        resolved = (
            decisions.select("record_id_a", "record_id_b", "decision")
            .join(
                id_to_hash.select(
                    F.col("record_id").alias("record_id_a"),
                    F.col("hash_id").alias("hash_id_a"),
                ),
                on="record_id_a",
                how="left",
            )
            .join(
                id_to_hash.select(
                    F.col("record_id").alias("record_id_b"),
                    F.col("hash_id").alias("hash_id_b"),
                ),
                on="record_id_b",
                how="left",
            )
            .withColumn("a", F.least("hash_id_a", "hash_id_b"))
            .withColumn("b", F.greatest("hash_id_a", "hash_id_b"))
            .dropDuplicates()
        )

        conflicting_decisions = (
            resolved.groupBy("a", "b")
            .agg(F.countDistinct("decision").alias("n_decisions"))
            .where(F.col("n_decisions") > 1)
            .limit(10)
            .collect()
        )
        if conflicting_decisions:
            examples = [(row["a"], row["b"]) for row in conflicting_decisions]
            raise ValueError(
                "manual decisions contain both must_link and cannot_link for the "
                f"same hash_id pair: {examples}"
            )

        must_link_edges = (
            resolved.where((F.col("decision") == "must_link") & (F.col("a") != F.col("b")))
            .select("a", "b")
            .dropDuplicates()
        )
        cannot_link_edges = (
            resolved.where(F.col("decision") == "cannot_link")
            .select("a", "b")
            .dropDuplicates()
        )

        return must_link_edges, cannot_link_edges, resolved

    def _drop_cannot_link_edges(
        self, edge_signals: DataFrame, cannot_link_edges: DataFrame
    ) -> DataFrame:
        if self._df_is_empty(edge_signals) or self._df_is_empty(cannot_link_edges):
            return edge_signals
        return edge_signals.join(cannot_link_edges, on=["a", "b"], how="left_anti")

    def _resolve_manual_identifier_clusters(
        self,
        df: DataFrame,
        fingerprints: DataFrame,
        manual_identifier_clusters: Optional[DataFrame],
    ) -> Tuple[DataFrame, DataFrame]:
        if manual_identifier_clusters is None:
            return self._empty_edge_frame(), self._empty_manual_identifier_clusters_frame()

        required_cols = {"manual_cluster_id", "field", "value"}
        missing_cols = required_cols - set(manual_identifier_clusters.columns)
        if missing_cols:
            raise ValueError(
                "manual_identifier_clusters is missing columns: "
                f"{sorted(missing_cols)}"
            )

        active = manual_identifier_clusters
        if "active" in active.columns:
            active = active.where(F.col("active") == F.lit(True))

        clusters = active.select(
            F.col("manual_cluster_id").cast("string").alias("manual_cluster_id"),
            F.col("field").cast("string").alias("field"),
            F.col("value").cast("string").alias("value"),
        ).dropDuplicates()

        if self._df_is_empty(clusters):
            return self._empty_edge_frame(), self._empty_manual_identifier_clusters_frame()

        bad_required = clusters.where(
            F.col("manual_cluster_id").isNull()
            | F.col("field").isNull()
            | F.col("value").isNull()
            | (F.length(F.trim(F.col("manual_cluster_id"))) == 0)
            | (F.length(F.trim(F.col("field"))) == 0)
            | (F.length(F.trim(F.col("value"))) == 0)
        ).limit(10).collect()
        if bad_required:
            examples = [
                (row["manual_cluster_id"], row["field"], row["value"])
                for row in bad_required
            ]
            raise ValueError(
                "manual_identifier_clusters has null/blank required values: "
                f"{examples}"
            )

        fields = [row["field"] for row in clusters.select("field").distinct().collect()]
        missing_fields = set(fields) - set(df.columns)
        if missing_fields:
            raise ValueError(
                "manual_identifier_clusters references fields not in df: "
                f"{self._sorted_values(list(missing_fields))}"
            )

        df_hash = df.join(
            fingerprints.select(self.record_id_col, "hash_id"),
            on=self.record_id_col,
            how="left",
        )

        resolved_parts: List[DataFrame] = []
        for field in fields:
            cluster_values = clusters.where(F.col("field") == F.lit(field))
            field_values = (
                df_hash.select(
                    F.col("hash_id"),
                    F.col(field).cast("string").alias("value"),
                )
                .where(F.col("value").isNotNull())
                .where(F.length(F.trim(F.col("value"))) > 0)
                .dropDuplicates()
            )
            resolved_parts.append(
                cluster_values.join(field_values, on="value", how="inner")
                .select("manual_cluster_id", "field", "value", "hash_id")
                .dropDuplicates()
            )

        resolved = self._union_all(resolved_parts) if resolved_parts else self._empty_manual_identifier_clusters_frame()
        if self._df_is_empty(resolved):
            return self._empty_edge_frame(), resolved

        manual_nodes = resolved.select("manual_cluster_id", "hash_id").dropDuplicates()
        root_window = Window.partitionBy("manual_cluster_id")
        manual_edges = (
            manual_nodes.withColumn("root", F.min("hash_id").over(root_window))
            .where(F.col("hash_id") != F.col("root"))
            .select(
                F.least("hash_id", "root").alias("a"),
                F.greatest("hash_id", "root").alias("b"),
            )
            .dropDuplicates()
        )
        return manual_edges, resolved

    # ----------------------------- Connected components / stable IDs -----------------------------

    def _connected_components(self, nodes: DataFrame, edges: DataFrame) -> DataFrame:
        node_ids = nodes.select(F.col("hash_id").alias("id")).dropDuplicates()
        if self._df_is_empty(node_ids):
            return self._empty_df(
                T.StructType(
                    [
                        T.StructField("hash_id", T.StringType(), True),
                        T.StructField("cluster_id", T.StringType(), True),
                    ]
                )
            )

        if self._df_is_empty(edges):
            return node_ids.select(
                F.col("id").alias("hash_id"),
                F.concat(F.lit("UN_"), F.col("id")).alias("cluster_id"),
            )

        n_nodes = node_ids.count()
        if n_nodes <= self.small_graph_collect_limit:
            return self._connected_components_small(node_ids, edges)

        edge_df = (
            edges.select(F.col("a").alias("src"), F.col("b").alias("dst"))
            .dropDuplicates()
            .persist(StorageLevel.MEMORY_AND_DISK)
        )
        _ = edge_df.count()

        labels = node_ids.withColumn("component", F.col("id")).persist(
            StorageLevel.MEMORY_AND_DISK
        )
        _ = labels.count()

        for _ in range(self.max_cc_iters):
            prop1 = edge_df.join(
                labels.select(F.col("id").alias("nbr"), F.col("component").alias("nbr_comp")),
                F.col("src") == F.col("nbr"),
                "left",
            ).select(F.col("dst").alias("id"), F.col("nbr_comp").alias("candidate"))
            prop2 = edge_df.join(
                labels.select(F.col("id").alias("nbr"), F.col("component").alias("nbr_comp")),
                F.col("dst") == F.col("nbr"),
                "left",
            ).select(F.col("src").alias("id"), F.col("nbr_comp").alias("candidate"))

            next_labels = (
                prop1.unionByName(prop2)
                .groupBy("id")
                .agg(F.min("candidate").alias("min_candidate"))
                .join(labels, on="id", how="right")
                .withColumn(
                    "component",
                    F.least(
                        F.col("component"),
                        F.coalesce(F.col("min_candidate"), F.col("component")),
                    ),
                )
                .select("id", "component")
                .persist(StorageLevel.MEMORY_AND_DISK)
            )
            _ = next_labels.count()

            changed = (
                next_labels.alias("next")
                .join(labels.alias("prev"), on="id", how="inner")
                .where(F.col("next.component") != F.col("prev.component"))
                .limit(1)
                .count()
            )
            labels.unpersist()
            labels = next_labels
            if changed == 0:
                break

        return self._labels_to_cluster_map(labels.select("id", "component"))

    def _connected_components_small(self, node_ids: DataFrame, edges: DataFrame) -> DataFrame:
        nodes = [row["id"] for row in node_ids.select("id").collect()]
        edge_rows = [(row["a"], row["b"]) for row in edges.select("a", "b").collect()]
        parent: Dict[str, str] = {}

        def find(x: str) -> str:
            parent.setdefault(x, x)
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(a: str, b: str) -> None:
            root_a, root_b = find(a), find(b)
            if root_a != root_b:
                if root_a < root_b:
                    parent[root_b] = root_a
                else:
                    parent[root_a] = root_b

        for node in nodes:
            parent.setdefault(node, node)
        for a, b in edge_rows:
            union(a, b)

        labels = [(node, find(node)) for node in nodes]
        labels_df = self.spark.createDataFrame(
            labels,
            schema=T.StructType(
                [
                    T.StructField("id", T.StringType(), False),
                    T.StructField("component", T.StringType(), False),
                ]
            ),
        )
        return self._labels_to_cluster_map(labels_df)

    def _labels_to_cluster_map(self, labels: DataFrame) -> DataFrame:
        summary = labels.groupBy("component").agg(
            F.min("id").alias("min_hash_id"),
            F.countDistinct("id").alias("n_hash_ids"),
        )

        matched = (
            summary.where(F.col("n_hash_ids") > 1)
            .withColumn(
                "cluster_id",
                F.dense_rank().over(Window.orderBy("min_hash_id")).cast("string"),
            )
            .select("component", "cluster_id")
        )
        singles = summary.where(F.col("n_hash_ids") == 1).select(
            "component",
            F.concat(F.lit("UN_"), F.col("min_hash_id")).alias("cluster_id"),
        )
        component_ids = matched.unionByName(singles)
        return labels.join(component_ids, on="component", how="inner").select(
            F.col("id").alias("hash_id"), "cluster_id"
        )

    def _stable_cluster_ids(
        self,
        cluster_map: DataFrame,
        manual_identifier_clusters: DataFrame,
    ) -> Tuple[DataFrame, DataFrame]:
        summary = cluster_map.groupBy("cluster_id").agg(
            F.min("hash_id").alias("min_hash_id"),
            F.countDistinct("hash_id").alias("n_hash_ids"),
        )
        summary = summary.withColumn(
            "auto_stable_cluster_id",
            F.when(
                F.col("n_hash_ids") == 1,
                F.concat(F.lit("UN_"), F.col("min_hash_id")),
            ).otherwise(F.concat(F.lit("AUTO_"), F.col("min_hash_id"))),
        )

        if not self._df_is_empty(manual_identifier_clusters):
            manual_by_cluster = (
                manual_identifier_clusters.select("manual_cluster_id", "hash_id")
                .dropDuplicates()
                .join(cluster_map, on="hash_id", how="inner")
                .groupBy("cluster_id")
                .agg(
                    F.min("manual_cluster_id").alias("manual_cluster_id"),
                    F.sort_array(F.collect_set("manual_cluster_id")).alias(
                        "manual_cluster_ids"
                    ),
                    F.countDistinct("manual_cluster_id").alias("n_manual_cluster_ids"),
                )
            )
            summary = summary.join(manual_by_cluster, on="cluster_id", how="left")
        else:
            summary = (
                summary.withColumn("manual_cluster_id", F.lit(None).cast("string"))
                .withColumn(
                    "manual_cluster_ids",
                    F.lit(None).cast(T.ArrayType(T.StringType())),
                )
                .withColumn("n_manual_cluster_ids", F.lit(0).cast("long"))
            )

        summary = summary.fillna({"n_manual_cluster_ids": 0}).withColumn(
            "stable_cluster_id",
            F.when(F.col("n_manual_cluster_ids") == 1, F.col("manual_cluster_id"))
            .when(
                F.col("n_manual_cluster_ids") > 1,
                F.concat(F.lit("MANUAL_CONFLICT_"), F.col("min_hash_id")),
            )
            .otherwise(F.col("auto_stable_cluster_id")),
        )

        stable_map = cluster_map.join(
            summary.select("cluster_id", "stable_cluster_id"),
            on="cluster_id",
            how="left",
        )
        conflicts = summary.where(F.col("n_manual_cluster_ids") > 1).select(
            "cluster_id",
            "stable_cluster_id",
            "auto_stable_cluster_id",
            "n_hash_ids",
            "n_manual_cluster_ids",
            "manual_cluster_ids",
        )
        if self._df_is_empty(conflicts):
            conflicts = self._empty_stable_cluster_conflicts_frame()
        return stable_map, conflicts

    # ----------------------------- Conflicts / stats -----------------------------

    def _manual_conflicts(self, manual_decisions: DataFrame, records: DataFrame) -> DataFrame:
        if self._df_is_empty(manual_decisions):
            return self._empty_manual_conflicts_frame()

        cannot_links = manual_decisions.where(F.col("decision") == "cannot_link")
        if self._df_is_empty(cannot_links):
            return self._empty_manual_conflicts_frame()

        cluster_lookup = records.select("hash_id", "cluster_id").dropDuplicates()
        conflicts = (
            cannot_links.join(
                cluster_lookup.select(
                    F.col("hash_id").alias("a"),
                    F.col("cluster_id").alias("cluster_id_a"),
                ),
                on="a",
                how="left",
            )
            .join(
                cluster_lookup.select(
                    F.col("hash_id").alias("b"),
                    F.col("cluster_id").alias("cluster_id_b"),
                ),
                on="b",
                how="left",
            )
            .where(F.col("cluster_id_a") == F.col("cluster_id_b"))
            .withColumn("cluster_id", F.col("cluster_id_a"))
            .withColumn(
                "conflict_type",
                F.when(F.col("a") == F.col("b"), F.lit("same_hash_id")).otherwise(
                    F.lit("still_connected")
                ),
            )
            .drop("cluster_id_a", "cluster_id_b")
        )
        if self._df_is_empty(conflicts):
            return self._empty_manual_conflicts_frame()
        return conflicts

    def _cluster_stats(
        self,
        records: DataFrame,
        cluster_map: DataFrame,
        edge_signals: DataFrame,
        manual_edges: DataFrame,
        fingerprints: DataFrame,
    ) -> DataFrame:
        base = records.select(
            self.record_id_col, "hash_id", "cluster_id", "stable_cluster_id"
        )
        stats = base.groupBy("cluster_id").agg(
            F.first("stable_cluster_id", ignorenulls=True).alias("stable_cluster_id"),
            F.countDistinct(self.record_id_col).alias("n_primary_ids"),
            F.countDistinct("hash_id").alias("n_hash_ids"),
        )

        cluster_lookup = cluster_map.select("hash_id", "cluster_id").dropDuplicates()

        def same_cluster(edges: DataFrame) -> DataFrame:
            return (
                edges.join(
                    cluster_lookup.select(
                        F.col("hash_id").alias("a"),
                        F.col("cluster_id").alias("cluster_id_a"),
                    ),
                    on="a",
                    how="left",
                )
                .join(
                    cluster_lookup.select(
                        F.col("hash_id").alias("b"),
                        F.col("cluster_id").alias("cluster_id_b"),
                    ),
                    on="b",
                    how="left",
                )
                .where(F.col("cluster_id_a") == F.col("cluster_id_b"))
            )

        algo_internal = same_cluster(edge_signals)
        manual_internal = same_cluster(manual_edges)
        all_internal = same_cluster(
            edge_signals.select("a", "b")
            .unionByName(manual_edges.select("a", "b"))
            .dropDuplicates()
        )

        edge_counts = all_internal.groupBy(F.col("cluster_id_a").alias("cluster_id")).agg(
            F.count(F.lit(1)).alias("n_edges")
        )
        thin_counts = algo_internal.groupBy(F.col("cluster_id_a").alias("cluster_id")).agg(
            F.sum(
                F.when(F.col("n_signals") == F.lit(int(self.k_required)), F.lit(1)).otherwise(
                    F.lit(0)
                )
            ).alias("thin_edges")
        )
        manual_counts = manual_internal.groupBy(
            F.col("cluster_id_a").alias("cluster_id")
        ).agg(F.count(F.lit(1)).alias("n_manual_edges"))

        stats = (
            stats.join(edge_counts, on="cluster_id", how="left")
            .join(thin_counts, on="cluster_id", how="left")
            .join(manual_counts, on="cluster_id", how="left")
            .fillna({"n_edges": 0, "thin_edges": 0, "n_manual_edges": 0})
            .withColumn(
                "dedup_ratio",
                F.col("n_primary_ids").cast("double") / F.col("n_hash_ids").cast("double"),
            )
            .withColumn(
                "edge_density",
                F.when(
                    F.col("n_edges") > 0,
                    F.col("n_hash_ids").cast("double") / F.col("n_edges").cast("double"),
                ),
            )
            .withColumn(
                "single_edge_cluster",
                (F.col("n_hash_ids") >= 2) & (F.col("n_edges") == 1),
            )
        )

        for col in self._columns:
            per_col = (
                fingerprints.select(self.record_id_col, "hash_id", col)
                .join(base, on=[self.record_id_col, "hash_id"], how="inner")
                .where(F.col(col).isNotNull())
                .groupBy("cluster_id")
                .agg(F.countDistinct(col).alias(f"n_distinct_{col}"))
            )
            stats = stats.join(per_col, on="cluster_id", how="left")

        return stats.orderBy(F.col("n_primary_ids").desc())


if __name__ == "__main__":
    spark = (
        SparkSession.builder.master("local[2]")
        .appName("EntityFusionSpark-demo")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    demo = spark.createDataFrame(
        [
            {"record_id": "E1A", "ein": "12-3456789", "phone": "P1"},
            {"record_id": "E1B", "ein": "12-3456789", "phone": "P2"},
            {"record_id": "E2A", "ein": "98-7654321", "phone": "P3"},
            {"record_id": "E2B", "ein": "98-7654321", "phone": "P4"},
            {"record_id": "E3A", "ein": "11-1111111", "phone": "P5"},
        ]
    )
    linker = EntityFusionSpark(
        spark=spark,
        record_id_col="record_id",
        match_columns=["ein", "phone"],
        k_required=1,
    )
    manual_identifier_clusters = spark.createDataFrame(
        [
            {
                "manual_cluster_id": "EIN_ALIAS_001",
                "field": "ein",
                "value": "12-3456789",
                "active": True,
            },
            {
                "manual_cluster_id": "EIN_ALIAS_001",
                "field": "ein",
                "value": "98-7654321",
                "active": True,
            },
        ]
    )

    out = linker.cluster(demo, manual_identifier_clusters=manual_identifier_clusters)
    out["records"].select(
        "record_id", "ein", "hash_id", "cluster_id", "stable_cluster_id"
    ).orderBy("record_id").show(truncate=False)
    out["cluster_stats"].show(truncate=False)
    spark.stop()
