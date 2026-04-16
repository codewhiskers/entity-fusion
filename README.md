# Entity Fusion

Entity Fusion is a record-linkage / entity-resolution project for grouping rows that likely refer to the same real-world entity.

The current core implementation lives in [Incremental_Fusion.py](./Incremental_Fusion.py). It supports:

- exact and fuzzy matching signals
- blocking to avoid all-to-all comparisons
- weighted scoring with optional bonuses
- flat signal plans and nested `AND` / `OR` condition trees
- incremental linking across batches
- cluster construction and persistence
- PyVis / HTML graph visualizations
- cluster dashboards and field-aware graph inspection

## Repo Map

- [Incremental_Fusion.py](./Incremental_Fusion.py)
  Main in-memory linker built on Polars, scikit-learn, NetworkX, and PyVis.
- [Incremental_Fusion_Spark.py](./Incremental_Fusion_Spark.py)
  Spark-oriented version of the linker for larger-scale workflows.
- [run_100k_entity_resolution.py](./run_100k_entity_resolution.py)
  Example runner for the PPP-style `100k.csv` dataset under `test_data/`.
- [er_incremental_debug.py](./er_incremental_debug.py)
  Small self-contained debug harness with inline toy data.
- [CompanyCleaner.py](./CompanyCleaner.py)
  Older name-cleaning utility.
- [test_data/](./test_data)
  Sample data, notebook experiments, and generated debug outputs.

## How It Works

At a high level, the linker does this:

1. Convert each record into one or more alias rows based on a `blocking_plan`.
2. Generate candidate pairs inside each signal type.
3. Optionally enforce `AND` / `OR` logic or K-of-N signal logic.
4. Score each pair using IDF, signal weight, and similarity score.
5. Select the strongest pairs.
6. Turn accepted pairs into clusters.
7. Optionally persist matches and emit visualizations.

The useful mental model is:

- a signal says "these two records look similar under one rule"
- the scorer combines multiple signals into one edge score
- accepted edges form a graph
- connected records become clusters

## Core API

The main class is `IncrementalSignalLinker` in [Incremental_Fusion.py](./Incremental_Fusion.py).

The two primary entry points are:

- `link(records)`
  One-shot linking. Returns matched pairs and labeled records, but does not mutate incremental cluster history.
- `link_incremental(records, batch_id=...)`
  Incremental linking. Reuses prior match state, updates clusters, and returns cluster metadata.

Both methods return a dict containing some combination of:

- `pairs_long`
- `pairs_scored`
- `final_pairs`
- `labeled_records`
- `clusters`
- `cluster_stats`
- `network`

## Blocking Plan Format

The `blocking_plan` is the main configuration object. It defines:

- which fields to compare
- how to block records before comparing
- whether matching is exact or cosine-based
- each signal's weight / bonus

### Flat Plan

Use a flat plan when each signal is independent and you want K-of-N logic.

```python
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
```

### Condition Tree

Use a nested tree when the logical structure matters more than K-of-N.

```python
blocking_plan = {
    "or": [
        {
            "and": [
                {"PHONE": {...}},
                {"NAME_KEY_COSINE": {...}},
            ]
        },
        {"PHONE": {...}},
    ]
}
```

Interpretation:

- `and`: a pair must satisfy every child condition
- `or`: a pair must satisfy at least one child condition
- leaf dicts define actual signal specs

## Important Parameters

Common `IncrementalSignalLinker` parameters:

- `record_id_col`
  Unique record identifier column.
- `blocking_plan`
  Signal configuration.
- `include_types`
  Which signal names to actually run.
- `k_required`
  Minimum number of signals needed when using a flat plan.
- `select_threshold`
  Hard score cutoff. If `None`, the quantile is used instead.
- `select_quantile`
  Quantile-based cutoff for final pair selection.
- `enable_incremental`
  Whether prior match state should be reused and updated.
- `matches_file`
  JSON file used to save/load prior matches.
- `create_network`
  Whether to emit PyVis network HTML outputs.
- `network_output_dir`
  Where those HTML outputs should be written.
- `return_intermediates`
  Include extra debugging artifacts in the output dict.
- `show_progress`
  Show progress bars for expensive cosine-block loops.

## Minimal Example

```python
import pandas as pd
from Incremental_Fusion import IncrementalSignalLinker

records = pd.DataFrame(
    [
        {"record_id": "R1", "name_key": "ACMSPPLY", "zip5": "10011", "phone": "+12125550101"},
        {"record_id": "R2", "name_key": "ACMSPPLY", "zip5": "10011", "phone": "+12125550101"},
        {"record_id": "R3", "name_key": "ACMESUPPLY", "zip5": "10012", "phone": "+12125550101"},
    ]
)

blocking_plan = {
    "PHONE": {
        "fields": ["phone"],
        "weight": 1.0,
    },
    "NAME_KEY_COSINE": {
        "fields": ["name_key"],
        "block_on": ["zip5"],
        "weight": 0.95,
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

linker = IncrementalSignalLinker(
    record_id_col="record_id",
    blocking_plan=blocking_plan,
    include_types=["NAME_KEY_COSINE", "PHONE"],
    k_required=1,
    create_network=False,
)

result = linker.link_incremental(records, batch_id="demo")
print(result["final_pairs"])
print(result["labeled_records"])
```

## Running The Debug Harness

[er_incremental_debug.py](./er_incremental_debug.py) is the easiest way to test the linker on a tiny known dataset.

Examples:

```bash
python er_incremental_debug.py
python er_incremental_debug.py --condition-tree
python er_incremental_debug.py --persist-matches --pdb
python er_incremental_debug.py --mode single-batch --pdb
```

This script uses inline sample batches and writes debug artifacts under:

- `test_data/_debug_outputs/`

## Running On `100k.csv`

[run_100k_entity_resolution.py](./run_100k_entity_resolution.py) is a dataset-specific runner for the PPP-style `100k.csv` file in `test_data/`.

Right now it:

- loads `test_data/100k.csv`
- creates a `record_id` from the row index
- matches on:
  - `BorrowerName`
  - `BorrowerAddress`
- blocks both signals by `BorrowerCity`
- writes labeled output to:
  - `test_data/_debug_outputs/100k_labeled_records.csv`

Run it with:

```bash
python run_100k_entity_resolution.py
```

Generated outputs include:

- `100k_labeled_records.csv`
- `dashboard.html`
- cluster-level field-aware graph HTML files
- `field_network_index.html`

## Visualizations

There are a few visualization layers in the project.

### 1. Basic Network Output

If `create_network=True`, the linker can generate PyVis HTML network outputs for accepted edges.

### 2. Cluster Evolution

`visualize_cluster_evolution(cluster_id)` writes an HTML file showing the time-ordered edges inside one cluster.

### 3. Field-Aware Graphs

`visualize_records_with_fields(...)` lets you build a graph where:

- node labels use a real field like `BorrowerName`
- node tooltips show selected input fields
- edge tooltips show `score` and `signals`

### 4. Dashboard

`generate_dashboard(...)` creates a larger interactive HTML page with:

- a graph on the left
- cluster list / cluster filtering
- a record table on the right

This is the best starting point when you want to inspect how nodes and edges are connected using real record values.

## Output Files

Common output locations:

- [matches.json](./matches.json)
  Saved match history and clusters.
- [network_outputs/](./network_outputs)
  PyVis cluster/network HTML files.
- [test_data/_debug_outputs/](./test_data/_debug_outputs)
  Debug script outputs, dashboards, labeled CSVs, and field-aware graph bundles.

## Notes And Caveats

- The current in-memory linker is best for local experimentation and moderate-sized datasets.
- Large blocks can still be expensive, even with blocking.
- For fuzzy matching, blank or too-short strings are skipped to avoid vectorizer failures.
- `link()` and `link_incremental()` are intentionally different:
  - `link()` is the simpler one-shot API
  - `link_incremental()` updates persistent cluster state
- Some older files in `archive/` reflect prior designs and may not match the current API exactly.

## Recommended Starting Points

If you are new to the codebase:

1. Read [Incremental_Fusion.py](./Incremental_Fusion.py)
2. Run [er_incremental_debug.py](./er_incremental_debug.py)
3. Run [run_100k_entity_resolution.py](./run_100k_entity_resolution.py)
4. Open the generated dashboard / field-network HTML outputs

If you want to tune matching quality:

1. start with thresholds and `block_on`
2. inspect clusters in the dashboard
3. inspect field-aware cluster graphs
4. add normalization / cleaning if the raw strings are noisy

