# entity_fusion_polars.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import polars as pl
import networkx as nx


# Distinguishes "value is null" from "value is an empty string" when hashing,
# so two rows can't accidentally fingerprint the same just because one field is missing.
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
class EntityFusionPolars:
    """
    Exact-match entity clustering on Polars.

    Given a dataframe and a set of columns that act as exact-match identifiers,
    clusters rows that are transitively connected through shared identifier
    values, requiring agreement across `k_required` distinct identifier
    columns before two rows are considered linked.

    One-shot / in-memory only: no incremental persistence, no fuzzy matching.
    Rows are deduplicated on a fingerprint of their identifier values before
    any graph work runs, since rows with identical identifier values are
    interchangeable for clustering purposes.

    Each entry in `match_columns` is either a plain column name, or a dict
    `{"column": ..., "block_on": [...]}` for a column whose candidate
    generation should be scoped to rows that also agree on the given
    `block_on` columns. `block_on` is a pure compute/candidate-pruning
    device -- those columns never count toward `k_required`.
    """

    record_id_col: str
    match_columns: List[Union[str, Dict[str, Any]]]
    k_required: int = 1
    block_cap: Optional[int] = None

    # Normalized from match_columns in __post_init__.
    _columns: List[str] = field(default_factory=list, init=False)
    _block_on: Dict[str, List[str]] = field(default_factory=dict, init=False)

    def __post_init__(self):
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
        df: pl.DataFrame,
        manual_links: Optional[pl.DataFrame] = None,
        manual_decisions: Optional[pl.DataFrame] = None,
        manual_identifier_clusters: Optional[pl.DataFrame] = None,
    ) -> Dict[str, pl.DataFrame]:
        """Cluster `df` on shared match_column values.

        `manual_decisions`, if given, is a persistent source-of-truth dataframe with
        `record_id_a`, `record_id_b`, and `decision` columns. `decision` should be
        either `must_link` (also accepts aliases like `true_positive`) or
        `cannot_link` (also accepts aliases like `false_positive`). Must-link pairs
        bypass k_required and participate in transitive clustering. Cannot-link
        pairs remove the direct algorithmic edge between those records' hash_ids; if
        the pair still lands in the same connected component through other edges, it
        is returned in `manual_conflicts`.

        `manual_links` is kept as a compatibility shortcut for positive-only manual
        links. Rows in `manual_links` are treated as `must_link` decisions.

        `manual_identifier_clusters`, if given, is a persistent source-of-truth table
        with `manual_cluster_id`, `field`, and `value` columns. All current records
        carrying any identifier values in the same `manual_cluster_id` are connected
        before clustering. Use this for durable identifier aliases, e.g. two EINs that
        should be treated as the same entity everywhere they appear.
        """
        self._validate_input(df)
        fingerprints = self._fingerprint(df)
        nodes = fingerprints.select("hash_id").unique()

        blocked_values = self._blocked_values(fingerprints)
        block_sizes = self._block_sizes(blocked_values)

        candidate_pairs = self._candidate_pairs(blocked_values, block_sizes)
        _, edge_signals = self._k_of_n_filter(candidate_pairs)

        (
            manual_edges,
            manual_cannot_link_edges,
            manual_decisions_resolved,
        ) = self._resolve_manual_decisions(manual_decisions, manual_links, fingerprints)
        manual_identifier_edges, manual_identifier_clusters_resolved = (
            self._resolve_manual_identifier_clusters(
                df, fingerprints, manual_identifier_clusters
            )
        )
        manual_edges = pl.concat(
            [manual_edges, manual_identifier_edges], how="vertical"
        ).unique()
        edge_signals = self._drop_cannot_link_edges(edge_signals, manual_cannot_link_edges)

        all_edges = pl.concat(
            [edge_signals.select("a", "b"), manual_edges], how="vertical"
        ).unique()
        cluster_map = self._connected_components(nodes, all_edges)
        cluster_map, stable_cluster_conflicts = self._stable_cluster_ids(
            cluster_map, manual_identifier_clusters_resolved
        )

        records = df.join(
            fingerprints.select(self.record_id_col, "hash_id"),
            on=self.record_id_col,
            how="left",
        ).join(cluster_map, on="hash_id", how="left")

        stats = self._cluster_stats(records, cluster_map, edge_signals, manual_edges, fingerprints)
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

    def _resolve_manual_links(
        self, manual_links: Optional[pl.DataFrame], fingerprints: pl.DataFrame
    ) -> pl.DataFrame:
        must_link_edges, _, _ = self._resolve_manual_decisions(None, manual_links, fingerprints)
        return must_link_edges

    def preview_blocks(self, df: pl.DataFrame) -> pl.DataFrame:
        """Block-size distribution per match_column, computed before any pairing/clustering.

        Run this before `cluster()` to catch a runaway identifier value (e.g. a
        placeholder like "N/A" shared by thousands of rows) before it drives an
        expensive self-join.
        """
        self._validate_input(df)
        fingerprints = self._fingerprint(df)
        blocked_values = self._blocked_values(fingerprints)
        return self._block_sizes(blocked_values)

    # ----------------------------- Manual review overrides -----------------------------
    #
    # cluster() draws its conclusions purely from shared match_column values. These let
    # a human reviewer correct the output afterwards without re-running the pipeline --
    # they only relabel cluster_id on `records`; they never change the underlying match
    # graph. Follow either with recompute_cluster_stats() to get a cluster_stats table
    # that reflects the correction.

    def merge_clusters(
        self, records: pl.DataFrame, cluster_ids: List[str], into: Optional[str] = None
    ) -> pl.DataFrame:
        """Force the given cluster_ids to become one cluster.

        Use when cluster_stats/manual review shows two clusters are actually the same
        entity despite no surviving edge connecting them. `into` picks the surviving
        cluster_id label (defaults to the first one given).
        """
        if len(cluster_ids) < 2:
            raise ValueError("merge_clusters needs at least 2 cluster_ids")
        target = into if into is not None else cluster_ids[0]
        expressions = [
            pl.when(pl.col("cluster_id").is_in(cluster_ids))
            .then(pl.lit(target))
            .otherwise(pl.col("cluster_id"))
            .alias("cluster_id")
        ]
        if "stable_cluster_id" in records.columns:
            target_stable = f"MANUAL_{target}"
            target_rows = records.filter(pl.col("cluster_id") == target)
            if target_rows.height > 0:
                stable_values = target_rows["stable_cluster_id"].unique().to_list()
                if len(stable_values) == 1:
                    target_stable = stable_values[0]
            expressions.append(
                pl.when(pl.col("cluster_id").is_in(cluster_ids))
                .then(pl.lit(target_stable))
                .otherwise(pl.col("stable_cluster_id"))
                .alias("stable_cluster_id")
            )
        return records.with_columns(expressions)

    def split_out(
        self,
        records: pl.DataFrame,
        record_ids: List[str],
        new_cluster_id: Optional[str] = None,
    ) -> pl.DataFrame:
        """Pull the given record_ids out of their current cluster(s).

        Use when a cluster merged records that shouldn't be together (e.g. linked only
        by a shared registered-agent address). Without `new_cluster_id`, the records
        become their own cluster; pass an existing cluster_id instead to move them
        there rather than carving out a new one.

        Records sharing a hash_id (identical match_column values) can't be split apart
        from each other -- they're indistinguishable under the current match_columns,
        so all of them must be included in `record_ids` together.
        """
        rid = self.record_id_col
        record_ids = list(record_ids)

        to_split = records.filter(pl.col(rid).is_in(record_ids))
        found = set(to_split[rid].to_list())
        missing = set(record_ids) - found
        if missing:
            raise ValueError(f"record_ids not found in records: {sorted(missing)}")

        affected_hash_ids = to_split.select("hash_id").unique()
        stray = records.join(affected_hash_ids, on="hash_id", how="inner").filter(
            ~pl.col(rid).is_in(record_ids)
        )
        if stray.height > 0:
            raise ValueError(
                "Cannot split: these record_ids are identical under match_columns "
                f"(same hash_id) to rows not included in the split: "
                f"{sorted(stray[rid].to_list())}. Include them too, or they'll never "
                "be distinguishable under the current match_columns."
            )

        if new_cluster_id is None:
            hash_ids = to_split["hash_id"].unique().to_list()
            if len(hash_ids) != 1:
                raise ValueError(
                    "record_ids span more than one hash_id -- pass an explicit "
                    "new_cluster_id, since there's no single natural singleton label "
                    "for a multi-hash_id group."
                )
            new_cluster_id = f"UN_{hash_ids[0]}"

        expressions = [
            pl.when(pl.col(rid).is_in(record_ids))
            .then(pl.lit(new_cluster_id))
            .otherwise(pl.col("cluster_id"))
            .alias("cluster_id")
        ]
        if "stable_cluster_id" in records.columns:
            expressions.append(
                pl.when(pl.col(rid).is_in(record_ids))
                .then(pl.lit(new_cluster_id))
                .otherwise(pl.col("stable_cluster_id"))
                .alias("stable_cluster_id")
            )
        return records.with_columns(expressions)

    def recompute_cluster_stats(
        self,
        records: pl.DataFrame,
        edge_signals: pl.DataFrame,
        manual_edges: Optional[pl.DataFrame] = None,
    ) -> pl.DataFrame:
        """Rebuild cluster_stats from `records` after a merge_clusters/split_out edit.

        `edge_signals` (and `manual_edges`, if the original cluster() call used
        manual must-link decisions) are the graph edges from that original call --
        overrides never change the underlying match graph, only the cluster_id labels,
        so the same edges are reused. Note that a merged cluster's
        n_edges/edge_density won't show a connecting edge between its two original
        halves, since none exists in the match graph -- that's expected, it's exactly
        what makes it a manual override rather than an algorithmic match.
        """
        if manual_edges is None:
            manual_edges = pl.DataFrame(schema={"a": pl.UInt64, "b": pl.UInt64})
        cluster_map = records.select("hash_id", "cluster_id").unique()
        return self._cluster_stats(records, cluster_map, edge_signals, manual_edges, records)

    # ----------------------------- Internals -----------------------------

    def _validate_input(self, df: pl.DataFrame) -> None:
        required = [self.record_id_col, *self._columns, *self._extra_block_on_columns()]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"df is missing required columns: {missing}")

        null_ids = df.filter(pl.col(self.record_id_col).is_null())
        if null_ids.height > 0:
            raise ValueError(f"{self.record_id_col!r} must be a non-null primary key")

        duplicate_ids = (
            df.group_by(self.record_id_col)
            .agg(pl.len().alias("n"))
            .filter(pl.col("n") > 1)
            .head(10)
        )
        if duplicate_ids.height > 0:
            examples = duplicate_ids.get_column(self.record_id_col).to_list()
            raise ValueError(
                f"{self.record_id_col!r} must be unique; duplicate values found: "
                f"{self._sorted_values(examples)}"
            )

    @staticmethod
    def _empty_edge_frame() -> pl.DataFrame:
        return pl.DataFrame(schema={"a": pl.UInt64, "b": pl.UInt64})

    def _empty_manual_decisions_frame(self, rid_dtype: pl.DataType) -> pl.DataFrame:
        return pl.DataFrame(
            schema={
                "record_id_a": rid_dtype,
                "record_id_b": rid_dtype,
                "decision": pl.Utf8,
                "hash_id_a": pl.UInt64,
                "hash_id_b": pl.UInt64,
                "a": pl.UInt64,
                "b": pl.UInt64,
            }
        )

    def _empty_manual_conflicts_frame(self, rid_dtype: pl.DataType) -> pl.DataFrame:
        return pl.DataFrame(
            schema={
                "record_id_a": rid_dtype,
                "record_id_b": rid_dtype,
                "decision": pl.Utf8,
                "hash_id_a": pl.UInt64,
                "hash_id_b": pl.UInt64,
                "a": pl.UInt64,
                "b": pl.UInt64,
                "cluster_id": pl.Utf8,
                "conflict_type": pl.Utf8,
            }
        )

    def _empty_manual_identifier_clusters_frame(self) -> pl.DataFrame:
        return pl.DataFrame(
            schema={
                "manual_cluster_id": pl.Utf8,
                "field": pl.Utf8,
                "value": pl.Utf8,
                "hash_id": pl.UInt64,
            }
        )

    @staticmethod
    def _empty_stable_cluster_conflicts_frame() -> pl.DataFrame:
        return pl.DataFrame(
            schema={
                "cluster_id": pl.Utf8,
                "stable_cluster_id": pl.Utf8,
                "auto_stable_cluster_id": pl.Utf8,
                "n_hash_ids": pl.UInt32,
                "n_manual_cluster_ids": pl.UInt32,
                "manual_cluster_ids": pl.List(pl.Utf8),
            }
        )

    @staticmethod
    def _sorted_values(values: Any) -> List[Any]:
        return sorted(values, key=lambda value: (type(value).__name__, str(value)))

    @staticmethod
    def _normalize_manual_decision(value: Any) -> str:
        if value is None:
            raise ValueError("manual_decisions decision cannot be null")
        key = str(value).strip().lower().replace(" ", "_")
        decision = _MANUAL_DECISION_ALIASES.get(key)
        if decision is None:
            allowed = ["must_link", "cannot_link", "true_positive", "false_positive"]
            raise ValueError(
                f"Unsupported manual decision {value!r}; expected one of {allowed}"
            )
        return decision

    def _manual_decision_rows(
        self,
        manual_decisions: Optional[pl.DataFrame],
        manual_links: Optional[pl.DataFrame],
    ) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []

        if manual_decisions is not None and manual_decisions.height > 0:
            missing_cols = {"record_id_a", "record_id_b", "decision"} - set(
                manual_decisions.columns
            )
            if missing_cols:
                raise ValueError(
                    f"manual_decisions is missing columns: {sorted(missing_cols)}"
                )

            active_decisions = manual_decisions
            if "active" in active_decisions.columns:
                active_decisions = active_decisions.filter(pl.col("active") == True)

            for row in active_decisions.to_dicts():
                normalized = dict(row)
                normalized["decision"] = self._normalize_manual_decision(row["decision"])
                rows.append(normalized)

        if manual_links is not None and manual_links.height > 0:
            missing_cols = {"record_id_a", "record_id_b"} - set(manual_links.columns)
            if missing_cols:
                raise ValueError(f"manual_links is missing columns: {sorted(missing_cols)}")

            for row in manual_links.to_dicts():
                rows.append(
                    {
                        "record_id_a": row["record_id_a"],
                        "record_id_b": row["record_id_b"],
                        "decision": "must_link",
                    }
                )

        return rows

    def _resolve_manual_decisions(
        self,
        manual_decisions: Optional[pl.DataFrame],
        manual_links: Optional[pl.DataFrame],
        fingerprints: pl.DataFrame,
    ) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        rid = self.record_id_col
        rid_dtype = fingerprints.schema[rid]
        rows = self._manual_decision_rows(manual_decisions, manual_links)
        if not rows:
            return (
                self._empty_edge_frame(),
                self._empty_edge_frame(),
                self._empty_manual_decisions_frame(rid_dtype),
            )

        ordered_keys = [
            "record_id_a",
            "record_id_b",
            "decision",
            *sorted(
                {
                    key
                    for row in rows
                    for key in row.keys()
                    if key not in {"record_id_a", "record_id_b", "decision"}
                }
            ),
        ]
        decisions = pl.DataFrame(
            [{key: row.get(key) for key in ordered_keys} for row in rows]
        )

        null_refs = decisions.filter(
            pl.col("record_id_a").is_null() | pl.col("record_id_b").is_null()
        )
        if null_refs.height > 0:
            raise ValueError("manual decisions cannot reference null record_ids")

        self_refs = decisions.filter(pl.col("record_id_a") == pl.col("record_id_b"))
        if self_refs.height > 0:
            examples = self_refs.select("record_id_a", "record_id_b").head(10).to_dicts()
            raise ValueError(f"manual decisions cannot reference the same record twice: {examples}")

        id_to_hash = fingerprints.select(rid, "hash_id")
        known = set(id_to_hash[rid].to_list())
        referenced = set(decisions["record_id_a"].to_list()) | set(
            decisions["record_id_b"].to_list()
        )
        missing = referenced - known
        if missing:
            raise ValueError(
                "manual decisions reference record_ids not in df: "
                f"{self._sorted_values(missing)}"
            )

        resolved = (
            decisions.join(
                id_to_hash.rename({rid: "record_id_a", "hash_id": "hash_id_a"}),
                on="record_id_a",
                how="left",
            )
            .join(
                id_to_hash.rename({rid: "record_id_b", "hash_id": "hash_id_b"}),
                on="record_id_b",
                how="left",
            )
            .with_columns(
                a=pl.min_horizontal("hash_id_a", "hash_id_b"),
                b=pl.max_horizontal("hash_id_a", "hash_id_b"),
            )
        )

        conflicting_decisions = (
            resolved.group_by(["a", "b"])
            .agg(n_decisions=pl.col("decision").n_unique())
            .filter(pl.col("n_decisions") > 1)
            .head(10)
        )
        if conflicting_decisions.height > 0:
            examples = conflicting_decisions.select("a", "b").to_dicts()
            raise ValueError(
                "manual decisions contain both must_link and cannot_link for the "
                f"same hash_id pair: {examples}"
            )

        must_link_edges = (
            resolved.filter((pl.col("decision") == "must_link") & (pl.col("a") != pl.col("b")))
            .select("a", "b")
            .unique()
        )
        cannot_link_edges = (
            resolved.filter(pl.col("decision") == "cannot_link")
            .select("a", "b")
            .unique()
        )

        return must_link_edges, cannot_link_edges, resolved

    def _drop_cannot_link_edges(
        self, edge_signals: pl.DataFrame, cannot_link_edges: pl.DataFrame
    ) -> pl.DataFrame:
        if edge_signals.height == 0 or cannot_link_edges.height == 0:
            return edge_signals
        return edge_signals.join(cannot_link_edges, on=["a", "b"], how="anti")

    def _resolve_manual_identifier_clusters(
        self,
        df: pl.DataFrame,
        fingerprints: pl.DataFrame,
        manual_identifier_clusters: Optional[pl.DataFrame],
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        if manual_identifier_clusters is None or manual_identifier_clusters.height == 0:
            return self._empty_edge_frame(), self._empty_manual_identifier_clusters_frame()

        required_cols = {"manual_cluster_id", "field", "value"}
        missing_cols = required_cols - set(manual_identifier_clusters.columns)
        if missing_cols:
            raise ValueError(
                "manual_identifier_clusters is missing columns: "
                f"{sorted(missing_cols)}"
            )

        active_clusters = manual_identifier_clusters
        if "active" in active_clusters.columns:
            active_clusters = active_clusters.filter(pl.col("active") == True)

        if active_clusters.height == 0:
            return self._empty_edge_frame(), self._empty_manual_identifier_clusters_frame()

        clusters = active_clusters.with_columns(
            manual_cluster_id=pl.col("manual_cluster_id").cast(pl.Utf8),
            field=pl.col("field").cast(pl.Utf8),
            value=pl.col("value").cast(pl.Utf8),
        )

        blank_required = clusters.filter(
            pl.any_horizontal(
                [
                    pl.col(c).is_null() | (pl.col(c).str.strip_chars().str.len_chars() == 0)
                    for c in ["manual_cluster_id", "field", "value"]
                ]
            )
        )
        if blank_required.height > 0:
            examples = blank_required.select(
                "manual_cluster_id", "field", "value"
            ).head(10).to_dicts()
            raise ValueError(
                "manual_identifier_clusters has null/blank required values: "
                f"{examples}"
            )

        fields = clusters.get_column("field").unique().to_list()
        missing_fields = set(fields) - set(df.columns)
        if missing_fields:
            raise ValueError(
                "manual_identifier_clusters references fields not in df: "
                f"{self._sorted_values(missing_fields)}"
            )

        metadata_cols = clusters.columns
        df_hash = df.join(
            fingerprints.select(self.record_id_col, "hash_id"),
            on=self.record_id_col,
            how="left",
        )

        resolved_parts = []
        for field in fields:
            cluster_values = clusters.filter(pl.col("field") == field)
            field_values = (
                df_hash.select(
                    hash_id=pl.col("hash_id"),
                    value=pl.col(field).cast(pl.Utf8),
                )
                .filter(pl.col("value").is_not_null())
                .filter(pl.col("value").str.strip_chars().str.len_chars() > 0)
                .unique()
            )
            resolved_parts.append(
                cluster_values.join(field_values, on="value", how="inner")
                .select(*metadata_cols, "hash_id")
                .unique()
            )

        resolved = (
            pl.concat(resolved_parts, how="vertical")
            if resolved_parts
            else self._empty_manual_identifier_clusters_frame()
        )
        if resolved.height == 0:
            return self._empty_edge_frame(), resolved

        manual_nodes = resolved.select("manual_cluster_id", "hash_id").unique()
        manual_edges = (
            manual_nodes.with_columns(
                root=pl.col("hash_id").min().over("manual_cluster_id")
            )
            .filter(pl.col("hash_id") != pl.col("root"))
            .select(
                a=pl.min_horizontal("hash_id", "root"),
                b=pl.max_horizontal("hash_id", "root"),
            )
            .unique()
        )
        return manual_edges, resolved

    def _stable_cluster_ids(
        self,
        cluster_map: pl.DataFrame,
        manual_identifier_clusters: pl.DataFrame,
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        component_summary = cluster_map.group_by("cluster_id").agg(
            min_hash_id=pl.col("hash_id").min(),
            n_hash_ids=pl.col("hash_id").n_unique(),
        )
        component_summary = component_summary.with_columns(
            auto_stable_cluster_id=pl.when(pl.col("n_hash_ids") == 1)
            .then(pl.concat_str([pl.lit("UN_"), pl.col("min_hash_id").cast(pl.Utf8)]))
            .otherwise(
                pl.concat_str([pl.lit("AUTO_"), pl.col("min_hash_id").cast(pl.Utf8)])
            )
        )

        if manual_identifier_clusters.height > 0:
            manual_by_cluster = (
                manual_identifier_clusters.select("manual_cluster_id", "hash_id")
                .unique()
                .join(cluster_map, on="hash_id", how="inner")
                .group_by("cluster_id")
                .agg(
                    manual_cluster_id=pl.col("manual_cluster_id").min(),
                    manual_cluster_ids=pl.col("manual_cluster_id").unique().sort(),
                    n_manual_cluster_ids=pl.col("manual_cluster_id").n_unique(),
                )
            )
            component_summary = component_summary.join(
                manual_by_cluster, on="cluster_id", how="left"
            )
        else:
            component_summary = component_summary.with_columns(
                manual_cluster_id=pl.lit(None, dtype=pl.Utf8),
                manual_cluster_ids=pl.lit(None, dtype=pl.List(pl.Utf8)),
                n_manual_cluster_ids=pl.lit(0, dtype=pl.UInt32),
            )

        component_summary = component_summary.with_columns(
            n_manual_cluster_ids=pl.col("n_manual_cluster_ids").fill_null(0)
        ).with_columns(
            stable_cluster_id=pl.when(pl.col("n_manual_cluster_ids") == 1)
            .then(pl.col("manual_cluster_id"))
            .when(pl.col("n_manual_cluster_ids") > 1)
            .then(
                pl.concat_str(
                    [pl.lit("MANUAL_CONFLICT_"), pl.col("min_hash_id").cast(pl.Utf8)]
                )
            )
            .otherwise(pl.col("auto_stable_cluster_id"))
        )

        stable_map = cluster_map.join(
            component_summary.select("cluster_id", "stable_cluster_id"),
            on="cluster_id",
            how="left",
        )
        conflicts = component_summary.filter(pl.col("n_manual_cluster_ids") > 1).select(
            "cluster_id",
            "stable_cluster_id",
            "auto_stable_cluster_id",
            "n_hash_ids",
            "n_manual_cluster_ids",
            "manual_cluster_ids",
        )
        if conflicts.height == 0:
            conflicts = self._empty_stable_cluster_conflicts_frame()

        return stable_map, conflicts

    def _manual_conflicts(
        self, manual_decisions: pl.DataFrame, records: pl.DataFrame
    ) -> pl.DataFrame:
        rid_dtype = records.schema[self.record_id_col]
        if manual_decisions.height == 0:
            return self._empty_manual_conflicts_frame(rid_dtype)

        cannot_links = manual_decisions.filter(pl.col("decision") == "cannot_link")
        if cannot_links.height == 0:
            return self._empty_manual_conflicts_frame(rid_dtype)

        cluster_lookup = records.select("hash_id", "cluster_id").unique()
        conflicts = (
            cannot_links.join(
                cluster_lookup.rename({"hash_id": "a", "cluster_id": "cluster_id_a"}),
                on="a",
                how="left",
            )
            .join(
                cluster_lookup.rename({"hash_id": "b", "cluster_id": "cluster_id_b"}),
                on="b",
                how="left",
            )
            .filter(pl.col("cluster_id_a") == pl.col("cluster_id_b"))
            .with_columns(
                cluster_id=pl.col("cluster_id_a"),
                conflict_type=pl.when(pl.col("a") == pl.col("b"))
                .then(pl.lit("same_hash_id"))
                .otherwise(pl.lit("still_connected")),
            )
        )
        if conflicts.height == 0:
            return self._empty_manual_conflicts_frame(rid_dtype)
        return conflicts.drop(["cluster_id_a", "cluster_id_b"])

    def _fingerprint(self, df: pl.DataFrame) -> pl.DataFrame:
        hash_expr = pl.struct(
            [
                pl.col(c).cast(pl.Utf8).fill_null(_NULL_SENTINEL)
                for c in self._columns
            ]
        ).hash()
        extra_cols = self._extra_block_on_columns()
        return df.select(
            self.record_id_col,
            *self._columns,
            *extra_cols,
            hash_id=hash_expr,
        )

    def _extra_block_on_columns(self) -> List[str]:
        if not self._block_on:
            return []
        seen = set(self._columns)
        extra = []
        for cols in self._block_on.values():
            for c in cols:
                if c not in seen:
                    seen.add(c)
                    extra.append(c)
        return extra

    def _blocked_values(self, fingerprints: pl.DataFrame) -> Dict[str, pl.DataFrame]:
        # Per match_column: distinct (block, value, hash_id) rows with blanks dropped,
        # so a shared null/empty value can never masquerade as a real identifier match.
        # `block` scopes candidate generation to rows that also agree on the
        # match_column's block_on columns (empty string when no block_on is set).
        tables: Dict[str, pl.DataFrame] = {}
        for col in self._columns:
            block_cols = self._block_on.get(col)
            block_expr = (
                pl.concat_str(
                    [pl.col(c).cast(pl.Utf8).fill_null(_NULL_SENTINEL) for c in block_cols],
                    separator="|",
                )
                if block_cols
                else pl.lit("")
            )
            t = (
                fingerprints.select(
                    hash_id=pl.col("hash_id"),
                    value=pl.col(col).cast(pl.Utf8),
                    block=block_expr,
                )
                .filter(pl.col("value").is_not_null())
                .filter(pl.col("value").str.strip_chars().str.len_chars() > 0)
                .unique()
            )
            tables[col] = t
        return tables

    def _block_sizes(self, blocked_values: Dict[str, pl.DataFrame]) -> pl.DataFrame:
        parts = []
        for col, t in blocked_values.items():
            sizes = (
                t.group_by(["block", "value"])
                .agg(n_hash_ids=pl.col("hash_id").n_unique())
                .with_columns(match_column=pl.lit(col))
                .select("match_column", "block", "value", "n_hash_ids")
            )
            parts.append(sizes)
        if not parts:
            return pl.DataFrame(
                schema={
                    "match_column": pl.Utf8,
                    "block": pl.Utf8,
                    "value": pl.Utf8,
                    "n_hash_ids": pl.UInt32,
                }
            )
        return pl.concat(parts, how="vertical").sort("n_hash_ids", descending=True)

    def _candidate_pairs(
        self, blocked_values: Dict[str, pl.DataFrame], block_sizes: pl.DataFrame
    ) -> pl.DataFrame:
        parts = []
        for col, t in blocked_values.items():
            tt = t
            if self.block_cap is not None:
                oversized = block_sizes.filter(
                    (pl.col("match_column") == col)
                    & (pl.col("n_hash_ids") > self.block_cap)
                ).select("block", "value")
                tt = tt.join(oversized, on=["block", "value"], how="anti")

            a = tt.rename({"hash_id": "a"})
            b = tt.rename({"hash_id": "b"})
            p = (
                a.join(b, on=["block", "value"], how="inner")
                .filter(pl.col("a") < pl.col("b"))
                .select("a", "b", signal=pl.lit(col))
                .unique()
            )
            parts.append(p)
        if not parts:
            return pl.DataFrame(schema={"a": pl.UInt64, "b": pl.UInt64, "signal": pl.Utf8})
        return pl.concat(parts, how="vertical")

    def _k_of_n_filter(self, pairs: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
        edge_schema = {"a": pl.UInt64, "b": pl.UInt64, "n_signals": pl.UInt32}
        if pairs.height == 0:
            return pairs, pl.DataFrame(schema=edge_schema)

        edge_signals = pairs.group_by(["a", "b"]).agg(
            n_signals=pl.col("signal").n_unique()
        )
        edge_signals = edge_signals.filter(pl.col("n_signals") >= self.k_required)
        filtered_pairs = pairs.join(edge_signals.select("a", "b"), on=["a", "b"], how="inner")
        return filtered_pairs, edge_signals

    def _connected_components(self, nodes: pl.DataFrame, pairs: pl.DataFrame) -> pl.DataFrame:
        # networkx is already a project dependency (used elsewhere for cluster graphs);
        # reusing it here avoids hand-rolling union-find.
        G = nx.Graph()
        G.add_nodes_from(nodes.get_column("hash_id").to_list())
        if pairs.height > 0:
            G.add_edges_from(pairs.select("a", "b").unique().iter_rows())

        components = [tuple(sorted(component)) for component in nx.connected_components(G)]
        components.sort(key=lambda component: component[0])

        rows = []
        next_id = 1
        for component in components:
            if len(component) == 1:
                (hash_id,) = component
                cluster_id = f"UN_{hash_id}"
            else:
                cluster_id = str(next_id)
                next_id += 1
            for hash_id in component:
                rows.append({"hash_id": hash_id, "cluster_id": cluster_id})

        return pl.DataFrame(rows, schema={"hash_id": pl.UInt64, "cluster_id": pl.Utf8})

    def _cluster_stats(
        self,
        records: pl.DataFrame,
        cluster_map: pl.DataFrame,
        edge_signals: pl.DataFrame,
        manual_edges: pl.DataFrame,
        fingerprints: pl.DataFrame,
    ) -> pl.DataFrame:
        stable_expr = (
            pl.col("stable_cluster_id")
            if "stable_cluster_id" in records.columns
            else pl.col("cluster_id").alias("stable_cluster_id")
        )
        base = records.select(self.record_id_col, "hash_id", "cluster_id", stable_expr)

        stats = base.group_by("cluster_id").agg(
            stable_cluster_id=pl.col("stable_cluster_id").first(),
            n_primary_ids=pl.col(self.record_id_col).n_unique(),
            n_hash_ids=pl.col("hash_id").n_unique(),
        )

        # Only count an edge toward a cluster if both endpoints are actually in it.
        # In the un-overridden output this is always true (cluster_id comes from
        # connected components of this exact edge set), but split_out() can put a and
        # b in different clusters -- a cut edge shouldn't inflate either side's count.
        def _same_cluster(edges: pl.DataFrame) -> pl.DataFrame:
            cluster_lookup = cluster_map.select("hash_id", "cluster_id")
            return (
                edges.join(
                    cluster_lookup.rename({"hash_id": "a", "cluster_id": "cluster_id_a"}),
                    on="a",
                    how="left",
                )
                .join(
                    cluster_lookup.rename({"hash_id": "b", "cluster_id": "cluster_id_b"}),
                    on="b",
                    how="left",
                )
                .filter(pl.col("cluster_id_a") == pl.col("cluster_id_b"))
            )

        algo_internal = _same_cluster(edge_signals)
        manual_internal = _same_cluster(manual_edges)
        # A pair present in both edge_signals and manual_edges (a reviewer manually
        # linking two records that already matched algorithmically) is deduped here so
        # it doesn't get counted twice toward n_edges.
        all_internal = _same_cluster(
            pl.concat([edge_signals.select("a", "b"), manual_edges.select("a", "b")], how="vertical").unique()
        )

        edge_counts = all_internal.group_by(pl.col("cluster_id_a").alias("cluster_id")).agg(
            n_edges=pl.len()
        )
        thin_counts = algo_internal.group_by(pl.col("cluster_id_a").alias("cluster_id")).agg(
            # Edges backed by exactly k_required signals: no corroboration beyond the
            # minimum bar, unlike edges confirmed by extra signal types.
            thin_edges=(pl.col("n_signals") == self.k_required).sum()
        )
        manual_counts = manual_internal.group_by(pl.col("cluster_id_a").alias("cluster_id")).agg(
            n_manual_edges=pl.len()
        )

        stats = (
            stats.join(edge_counts, on="cluster_id", how="left")
            .join(thin_counts, on="cluster_id", how="left")
            .join(manual_counts, on="cluster_id", how="left")
            .with_columns(
                n_edges=pl.col("n_edges").fill_null(0),
                thin_edges=pl.col("thin_edges").fill_null(0),
                n_manual_edges=pl.col("n_manual_edges").fill_null(0),
            )
            .with_columns(dedup_ratio=pl.col("n_primary_ids") / pl.col("n_hash_ids"))
            .with_columns(
                # A connected graph needs >= n_hash_ids - 1 edges, so this ratio is >= ~1
                # for a bare spanning tree (fragile: one bad edge could split the cluster)
                # and drops toward 0 as redundant edges pile up (robust).
                edge_density=pl.when(pl.col("n_edges") > 0)
                .then(pl.col("n_hash_ids") / pl.col("n_edges"))
                .otherwise(None),
                single_edge_cluster=(pl.col("n_hash_ids") >= 2) & (pl.col("n_edges") == 1),
            )
        )

        for col in self._columns:
            per_col = (
                fingerprints.select(self.record_id_col, "hash_id", col)
                .join(base, on=[self.record_id_col, "hash_id"], how="inner")
                .filter(pl.col(col).is_not_null())
                .group_by("cluster_id")
                .agg(**{f"n_distinct_{col}": pl.col(col).n_unique()})
            )
            stats = stats.join(per_col, on="cluster_id", how="left")

        return stats.sort("n_primary_ids", descending=True)


# ----------------------------- Example -----------------------------
if __name__ == "__main__":
    records = pl.DataFrame(
        [
            # TA group: R1/R2 are exact duplicates (dedup collapses them to one hash_id).
            {"record_id": "R1", "tax_id": "TA", "phone": "PA", "email": None},
            {"record_id": "R2", "tax_id": "TA", "phone": "PA", "email": None},
            {"record_id": "R3", "tax_id": "TA", "phone": "PX", "email": None},
            # TB group.
            {"record_id": "R4", "tax_id": "TB", "phone": "PB", "email": None},
            {"record_id": "R5", "tax_id": "TB", "phone": "PB", "email": None},
            {"record_id": "R6", "tax_id": "TB", "phone": "PY", "email": None},
            # Bridge: connects the TA group and TB group via one record.
            {"record_id": "R7", "tax_id": "TA", "phone": "PB", "email": None},
            # All-null rows: harmless collapse into a single UN_ singleton.
            {"record_id": "R8", "tax_id": None, "phone": None, "email": None},
            {"record_id": "R9", "tax_id": None, "phone": None, "email": None},
            # Single-shared-column pair -> single_edge_cluster flag should fire.
            {"record_id": "R10", "tax_id": None, "phone": None, "email": "Z1"},
            {"record_id": "R11", "tax_id": None, "phone": None, "email": "Z1"},
            # Shares 2 distinct columns (phone + email) -> only links under k_required=2.
            {"record_id": "R12", "tax_id": "TA", "phone": "PQ", "email": "ZQ"},
            {"record_id": "R13", "tax_id": "TD", "phone": "PQ", "email": "ZQ"},
        ]
    )

    linker = EntityFusionPolars(
        record_id_col="record_id",
        match_columns=["tax_id", "phone", "email"],
        k_required=1,
    )

    print("=== Pre-clustering block-size preview ===")
    print(linker.preview_blocks(records))

    print("=== k_required=1 (any shared column links) ===")
    out = linker.cluster(records)
    print(out["records"])
    print(out["cluster_stats"])

    print("=== k_required=2 (needs agreement on 2 distinct columns) ===")
    strict_linker = EntityFusionPolars(
        record_id_col="record_id",
        match_columns=["tax_id", "phone", "email"],
        k_required=2,
    )
    strict_out = strict_linker.cluster(records)
    print(strict_out["records"])
    print(strict_out["cluster_stats"])

    print("=== block_on: scoping tax_id matching to within the same region ===")
    # `name` also has to differ here: block_on only affects whether two *distinct*
    # hash_ids get an edge. If tax_id were the only match_column, N1/N2/N3 would
    # already collapse to one hash_id during fingerprinting (dedup only looks at
    # match_columns, not block_on scope columns) before block_on ever runs.
    region_records = pl.DataFrame(
        [
            {"record_id": "N1", "name": "ACME", "tax_id": "SHARED", "region": "east"},
            {"record_id": "N2", "name": "ACME EAST BRANCH", "tax_id": "SHARED", "region": "east"},
            # Same tax_id, different region -> not linked to N1/N2, thanks to block_on.
            {"record_id": "N3", "name": "ACME WEST BRANCH", "tax_id": "SHARED", "region": "west"},
        ]
    )
    region_linker = EntityFusionPolars(
        record_id_col="record_id",
        match_columns=["name", {"column": "tax_id", "block_on": ["region"]}],
        k_required=1,
    )
    region_out = region_linker.cluster(region_records)
    print(region_out["records"])

    print("=== manual_links: linking records with no shared identifier ===")
    ein_records = pl.DataFrame(
        [
            {"record_id": "M1", "ein": "EIN-AAA", "phone": "P1"},
            # Different EIN and phone -> no algorithmic edge to M1 at all.
            {"record_id": "M2", "ein": "EIN-BBB", "phone": "P2"},
            {"record_id": "M3", "ein": "EIN-CCC", "phone": "P3"},
        ]
    )
    ein_linker = EntityFusionPolars(
        record_id_col="record_id", match_columns=["ein", "phone"], k_required=1
    )
    manual_links = pl.DataFrame({"record_id_a": ["M1"], "record_id_b": ["M2"]})
    ein_out = ein_linker.cluster(ein_records, manual_links=manual_links)
    print(ein_out["records"])
    print(ein_out["cluster_stats"].select("cluster_id", "n_edges", "n_manual_edges"))

    print("=== manual_decisions: persistent true/false-positive review ===")
    review_records = pl.DataFrame(
        [
            {"record_id": "V1", "name": "ACME LLC", "address": "10 MAIN ST"},
            {"record_id": "V2", "name": "ACME LLC", "address": "20 OAK ST"},
            # Same shared office address as V1, but manually reviewed as different.
            {"record_id": "V3", "name": "BETA LLC", "address": "10 MAIN ST"},
            # No shared identifiers with V2, but a reviewer knows it is the same entity.
            {"record_id": "V4", "name": "ACME HOLDINGS", "address": "99 PINE ST"},
        ]
    )
    review_linker = EntityFusionPolars(
        record_id_col="record_id", match_columns=["name", "address"], k_required=1
    )
    manual_decisions = pl.DataFrame(
        {
            "record_id_a": ["V1", "V2"],
            "record_id_b": ["V3", "V4"],
            "decision": ["false_positive", "true_positive"],
            "reviewer": ["alice", "alice"],
            "active": [True, True],
        }
    )
    review_out = review_linker.cluster(
        review_records, manual_decisions=manual_decisions
    )
    print(review_out["records"])
    print(review_out["manual_conflicts"])

    print("=== manual_identifier_clusters: aliasing identifier values everywhere ===")
    alias_records = pl.DataFrame(
        [
            {"record_id": "E1A", "ein": "12-3456789", "phone": "P1"},
            {"record_id": "E1B", "ein": "12-3456789", "phone": "P2"},
            {"record_id": "E2A", "ein": "98-7654321", "phone": "P3"},
            {"record_id": "E2B", "ein": "98-7654321", "phone": "P4"},
            {"record_id": "E3A", "ein": "11-1111111", "phone": "P5"},
        ]
    )
    alias_linker = EntityFusionPolars(
        record_id_col="record_id", match_columns=["ein", "phone"], k_required=1
    )
    manual_identifier_clusters = pl.DataFrame(
        {
            "manual_cluster_id": ["EIN_ALIAS_001", "EIN_ALIAS_001"],
            "field": ["ein", "ein"],
            "value": ["12-3456789", "98-7654321"],
            "reviewer": ["alice", "alice"],
            "reason": ["known EIN rollover", "known EIN rollover"],
            "active": [True, True],
        }
    )
    alias_out = alias_linker.cluster(
        alias_records, manual_identifier_clusters=manual_identifier_clusters
    )
    print(alias_out["records"].select("record_id", "ein", "cluster_id", "stable_cluster_id"))
    print(alias_out["manual_identifier_edges"])
