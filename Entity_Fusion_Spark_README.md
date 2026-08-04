# Entity Fusion — PySpark exact-match edition

`Entity_Fusion_Spark.py` is a PySpark 3-compatible exact-match backend for the same workflow as `Entity_Fusion_Polars.py`.

Use this when the data is already in Spark/Databricks/Delta, or when one-machine Polars memory becomes the limiting factor. For ~1M-row local jobs, the Polars version will usually be faster and simpler.

## Quick Start

```python
from pyspark.sql import SparkSession
from Entity_Fusion_Spark import EntityFusionSpark

spark = SparkSession.builder.getOrCreate()

linker = EntityFusionSpark(
    spark=spark,
    record_id_col="record_id",
    match_columns=[
        "BorrowerAddress",
        "BorrowerZip",
        {"column": "BorrowerName", "block_on": ["BorrowerState"]},
    ],
    k_required=2,
    block_cap=2000,
)

out = linker.cluster(df)

records = out["records"]
stats = out["cluster_stats"]
```

`out["records"]` includes:

- `hash_id`
- `cluster_id`
- `stable_cluster_id`

`out["cluster_stats"]` includes one row per `cluster_id`, with the same review metrics as the Polars version plus `stable_cluster_id`.

## Pre-Cluster Block Preview

```python
blocks = linker.preview_blocks(df)
blocks.orderBy("n_hash_ids", ascending=False).show(25, truncate=False)
```

The output is:

```text
match_column | block | value | n_hash_ids
```

`block` is the extra `block_on` scope. If a match column has no `block_on`, `block`
is an empty string; that still means candidate generation is blocked by
`(match_column, value)`. With `block_on`, the block key is
`(match_column, block, value)`.

## Manual Identifier Clusters

Use this when the reviewed truth is "these identifier values are aliases everywhere they appear":

```python
manual_identifier_clusters = spark.createDataFrame([
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
])

out = linker.cluster(
    df,
    manual_identifier_clusters=manual_identifier_clusters,
)
```

All current rows whose `ein` is either value will be connected, and the resulting component gets:

```text
stable_cluster_id = EIN_ALIAS_001
```

If one connected component inherits multiple manual IDs, the output includes `stable_cluster_conflicts`, and that component receives a `MANUAL_CONFLICT_<min_hash_id>` stable label.

## Manual Pair Decisions

Use this when the reviewed truth is about a specific record pair:

```python
manual_decisions = spark.createDataFrame([
    {
        "record_id_a": "R1",
        "record_id_b": "R2",
        "decision": "must_link",    # aliases: true_positive, same, link
        "active": True,
    },
    {
        "record_id_a": "R3",
        "record_id_b": "R4",
        "decision": "cannot_link",  # aliases: false_positive, different
        "active": True,
    },
])

out = linker.cluster(df, manual_decisions=manual_decisions)
```

`cannot_link` removes the direct algorithmic edge. If the pair still lands in the same connected component through another path, it appears in `manual_conflicts`.

## Connected Components Note

This file does not require GraphFrames. It uses a pure-Spark min-label propagation fallback with:

```python
max_cc_iters=30
```

For very large or high-diameter graphs, raise `max_cc_iters` or use a production graph engine/GraphFrames-backed implementation. The fallback is intentionally dependency-light and PySpark-3-friendly.
