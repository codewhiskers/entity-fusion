# Polars Transitive Entity Clustering

For exact ID matching without the larger linker's dependencies or configuration,
use [transitive_fusion.py](./transitive_fusion.py). It requires only Polars.

```python
import polars as pl
from transitive_fusion import cluster_entities

records = pl.DataFrame({
    "name": ["Alice", "Alice B", "A. Brown", "Unrelated"],
    "ssn": ["111", "111", "222", None],
    "driver_license": ["NY123", "NY456", "NY456", None],
})

result = cluster_entities(records, ["ssn", "driver_license"], min_matches=1)
print(result)  # cluster_id: [0, 0, 0, 3]

# Require both IDs to agree before creating a direct link:
strict = cluster_entities(records, ["ssn", "driver_license"], min_matches=2)
# cluster_id: [0, 1, 2, 3]

# Optional cluster summary:
summary = result.group_by("cluster_id").len(name="cluster_size")
```

- `min_matches` defaults to 1 and counts distinct selected fields per pair.
  Matching is within each column: an SSN never matches a driver's license.
- Clustering is transitive **after** applying the threshold. For a two-of-three
  rule, A can match B on SSN/license and B can match C on license/phone, putting
  all three in one cluster even when A and C do not meet the threshold.
- Nulls, NaNs, and blank/whitespace-only strings never create links. All rows,
  including duplicates and unmatched records, are preserved in their original
  order. Missing fields do not count even on otherwise identical rows.
- Values compare exactly. Normalize formatting and replace placeholders such as
  `"UNKNOWN"` with null before clustering. Keep IDs as strings to preserve leading
  zeros. Consider combining license number with issuing state before matching
  when license numbers are only unique within a state.
- `cluster_column="entity_id"` changes the output column name. Existing columns
  are never overwritten. Labels are the earliest row position in each cluster;
  they are deterministic for a fixed input order, not durable IDs across batches.
- Grouping avoids expanding all pairs in large matching groups. It makes one
  grouping pass per combination of `min_matches` fields, so use a small set of
  trusted IDs. It runs in memory and supports scalar identifier columns.

For quality review, inspect large clusters and identifiers shared by unusually
many records: a common placeholder or an incorrect ID can bridge separate people.
Use the existing larger linker if you need fuzzy matching or manual overrides.

Run the standalone tests with `python -m unittest discover -s tests`.

## Nested Rules, Blockers, And Cluster Review

Use `match_rule` instead of `match_columns`/`min_matches` for more flexible logic:

```python
# Standalone example with all fields referenced by the rule.
import polars as pl
from transitive_fusion import cluster_entities

records = pl.DataFrame({
    "ssn": ["111", "111", "222", None],
    "driver_license": ["L1", "L2", "L2", None],
    "license_state": ["NY", "NY", "NY", None],
    "phone": ["555-0101", "555-0102", "555-0102", None],
    "country": ["US", "US", "US", None],
    "email": ["a@example.test", "b@example.test", "c@example.test", None],
})

rule = {
    "or": [
        {"field": "ssn"},
        {
            "at_least": 2,
            "conditions": [
                {"field": "driver_license", "block_on": ["license_state"]},
                {"field": "phone", "block_on": ["country"]},
                {"field": "email"},
            ],
        },
    ],
}

result = cluster_entities(records, match_rule=rule, return_details=True)
labeled_records = result.records
evidence = result.match_groups
result.export_html(
    "cluster_review.html",
    top_n=10,
    min_cluster_size=2,
    max_records=100,
    max_groups=200,
    conflict_columns=["ssn"],
    common_value_min_count=5,
    # record_id_column="record_id",  # optional labels for path lookup
)
```

This rule accepts SSN alone, or any two of license, phone, and email. Each
blocker must be nonmissing and equal for its field condition to count. Blockers
do not add votes, and they can change results: an equal phone with different
countries will not satisfy the phone condition. An `and` group requires every
child; an `or` group requires one. Nest any of these groups, with exactly one
operator per group. `at_least` counts immediate child conditions, not all leaves.
Duplicate children are rejected. Conditions may share fields across branches;
design branches carefully so correlated evidence does not imply more confidence
than intended. Conflicting values do not veto otherwise qualifying links.

The default return remains a Polars DataFrame. `return_details=True` instead
returns a `FusionResult` containing `records`, the canonical `rule`, and
`match_groups`. Each evidence group contains a group ID, cluster ID, zero-based
`row_positions`, sufficient matching `conditions`, and JSON `shared_values`
(including blockers). Every pair within a group satisfies the rule; overlapping
groups explain transitivity. Groups store complete evidence without expanding
all record pairs. More than one group may explain the same direct link.

The HTML is self-contained and opens locally, without dependencies or a server.
It includes:

- Largest clusters ranked by full record count, with record tables highlighting
  missing values and conflicting fields. Conflict checks default to all matching
  fields; use `conflict_columns` to focus on IDs expected to be unique.
- Common identifier frequencies computed from complete clusters, not samples.
- Evidence tables showing shared values, satisfied conditions, and blockers.
- A path lookup between two row positions, explaining each direct step even
  when the endpoints share nothing. Optional record IDs are display labels;
  zero-based positions always disambiguate records.
- A complete direct-match graph for clusters of at most `max_records` rows.
  Hover over links for evidence. Red links are true graph bridges: removing one
  separates the cluster. These are review hints, not proof of incorrect matches.

Record and evidence tables show deterministic first-row/group samples at their
limits, with explicit full counts. Larger clusters omit graphs and bridge claims;
their path lookup still uses full evidence. Thus the HTML embeds full membership
and matching values for selected clusters and can grow with large clusters.
`display_columns` limits only the record table, not match evidence. The report
contains the actual identifiers used for review. Nested/object columns are allowed
in display tables, but matching and conflict checks require scalar columns.

Complex rules compile into sufficient combinations (limited to 10,000 per group).
Each combination requires a grouping pass, so keep rule trees modest. Graphs may
have quadratic edge counts, bounded by the chosen `max_records` limit. The
clustering itself and stored evidence avoid quadratic pair materialization.

Try a synthetic overcluster example:

```bash
python examples/transitive_review.py /tmp/transitive_review.html
```

To benchmark the local PPP dataset and export reviews of its largest clusters:

```bash
python examples/benchmark_transitive_fusion.py --mode blocked
python examples/benchmark_transitive_fusion.py --mode unblocked
python examples/benchmark_transitive_fusion.py --mode strict
```

The default input is `test_data/100k.parquet` (968,512 rows despite its name).
`unblocked` matches exact borrower name OR address; `blocked` additionally requires
city and state agreement for each condition; `strict` requires both name AND
address within city/state. No additional normalization is applied. Use `--limit 10000` for a smaller run or `--input path.parquet` for another file with those
columns. Each run writes JSON timings, cluster statistics, and an HTML review to
`test_data/_debug_outputs/transitive_benchmark/`. Run modes in separate processes
for comparable peak memory figures; those figures include input loading,
clustering with evidence, validation, and report generation. Consistency checks
verify row preservation, earliest-member labels, and evidence memberships;
these checks do not establish real-world entity accuracy.

## Function Parameters

| Parameter | Default | Meaning |
|---|---|---|
| `df` | Required | Eager Polars DataFrame. |
| `match_columns` | `None` | Distinct fields for simple K-of-N matching; supply this or `match_rule`. |
| `min_matches` | `1` | Minimum number of matching fields in simple mode. |
| `match_rule` | `None` | Nested rule dictionary; replaces `match_columns`/`min_matches`. |
| `cluster_column` | `"cluster_id"` | Output label column; must not already exist. |
| `return_details` | `False` | Return `FusionResult` rather than just the labeled DataFrame. |

## HTML Review Options

Call `result.export_html(path, ...)` after clustering with `return_details=True`.

| Option | Default | Meaning |
|---|---|---|
| `top_n` | `10` | Largest qualifying clusters to include. |
| `min_cluster_size` | `2` | Minimum full cluster size. |
| `max_records` | `100` | Record table cap and maximum size for complete graphs/bridge detection. |
| `max_groups` | `200` | Evidence table cap. |
| `record_id_column` | `None` | Optional input column used for labels in path explanations. |
| `display_columns` | All columns | Columns in the record table. |
| `conflict_columns` | All matching fields | Fields to check for multiple distinct nonmissing values. |
| `common_value_min_count` | `5` | Minimum within-cluster record count for a common identifier flag. |

The export returns the destination `Path`. The parent directory must already exist.

## Benchmark: 968,512 Records

Measured locally on November 17, 2026, using Polars 1.33.1 and the full
`test_data/100k.parquet` file. The four input columns were `BorrowerName`,
`BorrowerAddress`, `BorrowerCity`, and `BorrowerState`, with no additional
normalization. The CSV is a separate input with 968,525 rows; these results
apply to the Parquet file.

Each rule ran once in a fresh process with `return_details=True`. Clustering
times exclude input loading, consistency validation, and HTML generation.
Memory is the peak for the **entire process**, including those additional steps.
These are observations on this machine, not guaranteed performance figures.

| Rule | Clustering | HTML export | Process peak memory | Clusters | Records in nonsingleton clusters | Largest cluster | HTML size |
|---|---:|---:|---:|---:|---:|---:|---:|
| Name OR address | 3.37 s | 0.95 s | 1,381.7 MiB | 786,906 | 322,189 | 108 | 0.282 MiB |
| Name OR address, blocked by city/state | 3.73 s | 0.92 s | 2,102.7 MiB | 812,230 | 284,188 | 56 | 0.284 MiB |
| Name AND address, blocked by city/state | 0.53 s | 0.44 s | 958.2 MiB | 956,266 | 24,480 | 7 | 0.037 MiB |

The review exports contained the ten largest clusters using default table and
graph limits. The unblocked run's largest cluster (108 records) exceeded the
100-record graph limit, so its report omitted that cluster's graph and bridge
analysis while retaining full evidence for path lookup. All three reports stayed
below 0.3 MiB; this does not imply that reports of much larger clusters will stay
small, because full selected-cluster evidence remains embedded.

All three runs passed checks for unchanged input rows/order, labels equal to the
earliest cluster member, and evidence groups belonging entirely to their assigned
clusters. The 12-test suite also passed, including randomized comparisons against
an independent pairwise reference. These validate implementation consistency;
there is no ground-truth assessment of whether each business cluster is correct.

The more permissive rules produced larger clusters and more linked records.
Neither more links nor smaller clusters establish better quality: inspect shared
addresses, common business names, and transitive bridges before choosing a rule.

To reproduce and inspect the reports, run the benchmark commands above. Artifacts
are written under `test_data/_debug_outputs/transitive_benchmark/`:

| Mode | Review report | Machine-readable metrics |
|---|---|---|
| Unblocked | `unblocked_968512.html` | `unblocked_968512.json` |
| Blocked | `blocked_968512.html` | `blocked_968512.json` |
| Strict | `strict_968512.html` | `strict_968512.json` |

For a quick smoke test, add `--limit 10000`; output filenames use the actual row
count. The benchmark script is [examples/benchmark_transitive_fusion.py](./examples/benchmark_transitive_fusion.py).
