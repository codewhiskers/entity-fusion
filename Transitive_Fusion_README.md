# Polars Transitive Entity Clustering

## Dominant-Value Evaluation: Two DataFrames

Use `fields` in `evaluate_clusters` for record-level review against each cluster's
dominant field values. This mode returns **all input records and all clusters by
default**, including singletons. It does not change membership or propose splits.

```python
from transitive_review import evaluate_clusters

assessment = evaluate_clusters(
    real_fusion,  # FusionResult, or a Polars DataFrame already containing cluster IDs
    record_id_column="demo_record_id",
    fields={
        "BorrowerName": {"method": "tfidf", "threshold": 0.85},
        "BorrowerAddress": {"method": "levenshtein", "threshold": 0.80},
        "BorrowerState": {"method": "exact"},
    },
    # filing_count_column="filing total",  # sum a numeric count, if present
    # filing_id_column="filing_id",         # optional alongside the count column
)
record_review, cluster_review = assessment
```

`assessment.records` and `.clusters` are the same two DataFrames; `.configuration`
records the evaluation settings. For a plain DataFrame with a differently named
cluster column, supply `cluster_id_column="entity_id"`. Record IDs must be unique
and nonmissing; aggregate filing rows by record first or supply unique source-row
IDs. IDs in these tables are standardized to `record_id` and `cluster_id`.

### Record Review

One row per input record, preserving input order:

| Column | Meaning |
|---|---|
| `record_id`, `cluster_id` | Join keys into the original clustered records. |
| `evaluated` | Whether the record's cluster was selected for scoring. |
| `requires_review` | True if this record has any configured issue; null when not evaluated. |
| `review_reasons` | Field-specific reasons such as `BorrowerName:below_threshold` or `ein:dominant_tie`. |
| `<field>__value` | Original input value. |
| `<field>__dominant_value` | Normalized reference value chosen for this cluster. |
| `<field>__similarity` | Similarity to that reference (0–1), not a probability. |
| `<field>__status` | `ok`, `below_threshold`, `missing`, `dominant_tie`, or `not_evaluated`. |
| `<field>__requires_review` | Whether this specific field causes review. |
| `<field>__distance` | Raw edit distance, included for Levenshtein fields only. |

A record is flagged for its own field results, rather than inheriting every other
member's issue. For example, with two identical names and one dissimilar name,
only the dissimilar record fails the name threshold. Ambiguous reference ties
can flag every record because no unique dominant value was established.

```python
annotations = record_review.select(
    "record_id", "requires_review", "review_reasons",
)
enriched = real_fusion.records.join(
    annotations.rename({"record_id": "demo_record_id"}),
    on="demo_record_id", how="left", validate="1:1", maintain_order="left",
)
```

### Cluster Review

One row per cluster, sorted by record count descending and cluster ID ascending:

| Column | Meaning |
|---|---|
| `record_count` | Number of unique input record IDs in the cluster. |
| `total_filings` | Uses summed counts when a count column is configured; otherwise distinct filing IDs. Null when that source is missing. |
| `summed_filing_count`, `distinct_filing_ids` | Separate totals for each configured source; both may be supplied together. |
| `filing_counts_missing_count`, `filing_ids_missing_count` | Separate missing-data counts for the two sources. |
| `filings_missing_count` | Number of records lacking the configured filing information. |
| `review_record_count`, `requires_review`, `review_reasons` | Cluster-level rollup of individual record issues. |
| `<field>__method`, `<field>__threshold` | How the field was evaluated. |
| `<field>__dominant_value`, `__dominant_count`, `__dominant_fraction` | Reference and its count/share among nonmissing records. |
| `<field>__dominant_tied`, `__distinct_values`, `__missing_count` | Reference ambiguity and data consistency. |
| `<field>__compared_count`, `__below_threshold_count`, `__review_count` | Scoring coverage and field-specific issues. |
| `<field>__mean_similarity`, `__min_similarity` | Cohesion against the reference, excluding unscored/missing values. |

Supply a literal source name such as `filing_count_column="filing total"` for a
per-record filing-count column. You may also supply `filing_id_column` in the same
call. The two measures are never added together: summed counts include filings
represented by the counts, while distinct IDs count only IDs actually present
in that column. `total_filings` consistently uses the count source when configured,
without falling back to ID counts for individual missing totals.

Filings are independent of record counts and do **not** weight dominance. Null
filing counts are excluded from sums and reported in `filings_missing_count`;
partial totals should be interpreted with that count. Counts must be finite,
nonnegative integers. Distinct-ID totals exclude null/blank IDs. No filing field
is guessed automatically, and the local four-column PPP Parquet has none.

### Field Policies And Dominance

Each field supports `method`, `threshold`, `normalize`, `missing`, and
`review_on_tie`:

- `exact`: score 1 for equal normalized values, 0 otherwise; default threshold 1.
- `tfidf`: character `char_wb` n-grams (2–4) and cosine similarity to the dominant
  value; default threshold 0.85. The corpus is the **distinct normalized values**
  in evaluated comparisons, unlike the legacy neighbor evaluator's record-weighted
  corpus. Scores can therefore differ between the two modes.
- `levenshtein`: score `1 - distance / max(length(value), length(reference))`, with
  the raw distance also returned; default threshold 0.85. RapidFuzz accelerates
  this when installed; a Python fallback produces the same distance.
- `normalize=True` lowercases text and collapses/strips whitespace. Set it to
  `False` for case-sensitive/raw comparisons. Nulls, NaNs, and whitespace-only
  strings remain missing. Exact comparisons retain numeric/scalar field types.
- `missing="review"` flags missing values by default. `missing="ignore"` retains
  their missing status and null score without generating a missing-value flag.
- `review_on_tie=True` flags tied dominant values. Ties choose the lowest sorted
  normalized value for reproducibility and remain visible even if this flag is
  disabled. A dominant value is the mode, not necessarily a strict majority.

Dominance uses all records in each cluster, excluding missing values. A dominant
value is a comparison reference, not proof that the majority is correct. Review
thresholds and policies need to reflect the field's role and source quality.

`max_clusters` and `min_cluster_size` can limit scoring, but **never drop rows** in
this mode. Unselected rows remain with `evaluated=False`, null review flags and
scores, and `not_evaluated` field statuses. Basic counts/dominant statistics still
exist for every cluster. Without explicit limits, all records are evaluated.

The existing `name_column=...` call still returns the legacy `NameReview` used by
the graph dashboard and split workflow. It is also available explicitly as
`evaluate_name_neighbors`. Do not combine `fields` with neighbor-specific
parameters. The new two-table result is for dominant-value diagnostics; it does
not silently generate neighbor graphs or name-based subgroup proposals.

Try [examples/dominant_review.py](./examples/dominant_review.py):

```bash
python examples/dominant_review.py
python examples/dominant_review.py --input test_data/100k.parquet --limit 100000
```

The notebook includes synthetic filing totals and a real-data two-table example.
Outputs are written as Parquet plus JSON configuration. A local 100,000-row run
with name TF-IDF and address Levenshtein took 3.85 seconds for evaluation and
returned 100,000 record rows and 85,649 cluster rows. This is a different scoring
policy from neighbor review; their flag counts are not directly comparable.

## Interactive Notebook

Open [Test_Transitive_Fusion.ipynb](./Test_Transitive_Fusion.ipynb) and run cells
in order with a kernel containing `polars`, `scikit-learn`, and `sparse_dot_topn`.
It covers simple K-of-N matching, nested rules/blockers, exact match evidence,
name-review tables, HTML exports, and a saved synthetic manual split.

The real-data configuration defaults to the first 10,000 rows of the local
`100k.parquet` and its ten largest clusters. Set `ROW_LIMIT=None` for the full file
and `REVIEW_MAX_CLUSTERS=None` for every nonsingleton cluster. Tables are exported
in full; HTML is limited to `REPORT_TOP_N` clusters. Notebook artifacts go under
`test_data/_debug_outputs/transitive_notebook/`. The real-data experiment does not
apply splits, and its generated record IDs are specific to that extraction.

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

Complex rules compile into sufficient combinations (limited to 10,000 at each
rule-tree group). Compilation also has a shared work limit of 1,000,000 generated
combinations and subset comparisons across the tree, including combinations that
deduplicate away. Rules exceeding either limit raise `ValueError` before execution.
Each sufficient combination requires a grouping pass, so keep rule trees modest. Graphs may
have quadratic edge counts, bounded by the chosen `max_records` limit. The
clustering itself and stored evidence avoid quadratic pair materialization.

Internally, rule handling has three separate steps:

1. `_parse_rule` validates the dictionary and builds immutable field conditions
   and threshold groups. `and` and `or` use the same threshold model while
   retaining their original operator names for report output. Parsing does not
   expand combinations.
2. `_compile_exact_rule` turns that tree into an exact-match plan. It rejects
   logically equivalent sibling conditions, removes redundant combinations, and
   applies the compilation limits. The plan retains all referenced fields for
   input validation, including fields removed from the execution combinations.
3. `_execute_exact_plan` validates input columns and runs Polars grouping and
   union-find using the plan, retaining compact evidence when requested.

These are internal interfaces; the public `cluster_entities` arguments, return
types, rule dictionaries, and HTML export API are unchanged. The separation lets
future comparison methods reuse the parsed rule structure without requiring them
to use exact grouping plans.

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

## Name Review And Explicit Cluster Splits

[transitive_review.py](./transitive_review.py) adds a separate evaluation/refinement
layer. The exact clustering module still requires only Polars. Name evaluation
additionally requires `scikit-learn` and `sparse_dot_topn` (v1 API). Its retrieval
uses [`sp_matmul_topn`](https://github.com/ing-bank/sparse_dot_topn#usage).

```bash
pip install scikit-learn sparse_dot_topn
python examples/name_refinement.py
# Review the ten largest clusters from the full local PPP file:
python examples/name_refinement.py --input test_data/100k.parquet
# Evaluate every nonsingleton cluster; keep HTML limited to the ten largest:
python examples/name_refinement.py --input test_data/100k.parquet --all-clusters \
    --output-dir test_data/_debug_outputs/name_refinement_full
```

These commands write review HTML, Parquet tables, and configuration/summary JSON
under `test_data/_debug_outputs/name_refinement/`. The synthetic example also
applies an explicit illustrative reviewer decision and saves refined records and
crossing evidence. The PPP example makes **no membership changes**. Its generated
row IDs are for that extraction only; supply `--record-id-column` with a unique
source ID when available for durable review decisions.

With `--all-clusters`, complete review tables are written for every cluster with
at least two records. Singletons remain in `ppp_labeled_records.parquet` but are
not name-evaluated. `ppp_match_groups.parquet` retains the original exact evidence.
The HTML includes only the ten largest evaluated clusters. Because TF-IDF is fit
over all evaluated usable names, scores can differ from the ten-cluster preview.

Full-file name review on November 18, 2026 (`threshold=0.85`):

| Measurement | Result |
|---|---:|
| Input rows | 968,512 |
| Nonsingleton clusters evaluated | 127,906 |
| Records evaluated | 284,188 |
| Input loading plus exact clustering | 4.34 s |
| Name evaluation | 44.11 s |
| Clusters flagged for review | 31,648 (24.74%) |
| Records with a name neighbor above threshold | 219,785 |
| Records below the name threshold | 64,402 |
| Records with missing names | 1 |
| Retrieved name pairs | 255,252 |

All evaluated clusters were small enough for complete positive-pair comparisons,
so this run's name component counts were not truncated by top-N retrieval.
Original memberships were preserved; pair endpoints were verified to belong to
their assigned original clusters, and there were no self matches. The 31,648
flags identify multiple name groups, not confirmed errors. No manual split was
applied. Outputs are under `test_data/_debug_outputs/name_refinement_full/`;
the HTML shows ten clusters, while Parquet contains the full evaluation.

### Name Evaluation Performance

The evaluator now computes sparse-vector feature presence once, batches dot
products for clusters with exactly two usable names, and skips neighbor-search
setup for singletons or clusters with fewer than two usable names. Singletons
still receive review rows when `min_cluster_size=1` and remain part of TF-IDF
fitting, preserving the selected evaluation corpus and score interpretation.

Local measurements on November 18, 2026:

| Input | Minimum cluster size | Before | Optimized | Comparison |
|---|---:|---:|---:|---|
| First 100,000 rows | 2 | 3.03 s | 1.05 s | Back-to-back; 2.9× faster |
| First 100,000 rows | 1 | 18.30 s | 3.55 s | Back-to-back; 5.2× faster |
| Full 968,512 rows | 2 | 44.11 s | 12.55 s | Earlier full run versus optimized run |

All four output tables matched the prior implementation on both sample scopes
and the saved full-file evaluation, with a `1e-12` tolerance for numeric scores.
Flags, proposed memberships, coverage, and the full-file 31,648 review count were
unchanged. Timings exclude exact clustering and HTML generation; the sample
comparison used warmed imports. This is an implementation optimization, not a
reduction in evaluation coverage or a new scoring policy. Threshold changes still
recompute vectors and similarities; cross-call caching is not implemented.

If the evaluator is already imported in a running notebook, reload it before
measuring the updated implementation:

```python
import importlib
import transitive_review
importlib.reload(transitive_review)
evaluate_clusters = transitive_review.evaluate_clusters
```

### Evaluate Without Changing Membership

Given an input `df` with a unique, nonmissing stable `record_id`, a string `name`,
and identifier columns:

```python
from transitive_fusion import cluster_entities
from transitive_review import evaluate_clusters, apply_cluster_splits

fusion = cluster_entities(df, ["ein", "phone"], return_details=True)
review = evaluate_clusters(
    fusion,
    name_column="name",
    record_id_column="record_id",
    threshold=0.85,
    top_n=10,
    complete_limit=100,
    min_cluster_size=2,
    max_clusters=10,  # None evaluates all qualifying clusters
)
review.export_html("name_review.html", fusion=fusion)
review.records.write_parquet("record_review.parquet")
review.clusters.write_parquet("cluster_review.parquet")
review.name_pairs.write_parquet("name_pairs.parquet")
review.proposed_splits.write_parquet("proposed_splits.parquet")
```

The tables have distinct purposes:

| Table | Contents |
|---|---|
| `fusion.records` | Original records plus unchanged exact `cluster_id`. |
| `fusion.match_groups` | Exact rule evidence behind the original cluster. |
| `review.records` | Record ID, name, original cluster, best neighbor and score, name status, threshold support, and proposed subgroup. |
| `review.clusters` | Counts, threshold, supported fraction, unsupported names, component counts, retrieval completeness, and review reasons. |
| `review.name_pairs` | Retrieved within-cluster pairs, cosine similarities, and threshold decisions. |
| `review.proposed_splits` | Proposed memberships by record ID; these are not accepted decisions. |

Review tables use fixed `record_id` and `cluster_id` column names, even if the
input uses different names. Names are lowercased and whitespace-normalized for
comparison, but original names are retained in the record table. TF-IDF character
word-boundary n-grams of lengths 2–4 are fitted over usable names in the selected
clusters. Changing the selected corpus can change similarity scores; scores are
not probabilities and 0.85 is an exploratory setting, not a calibrated cutoff.

Comparisons never cross original cluster boundaries. Clusters up to
`complete_limit` retrieve all positive pair scores. Larger clusters retain up to
`top_n` neighbors per record, excluding self; the union of both directions forms
the review graph. Best-neighbor scores below the threshold are retained, too.
The `graph_complete` flag distinguishes complete comparisons from a top-N graph,
whose component count may overstate fragmentation by omitting qualifying links.

`name_component_count` counts threshold-connected groups among usable names;
`proposed_component_count` additionally includes separate proposals for records
with missing/unusable names. Similarity components allow transitive chains:
not every pair in a proposed group must meet the threshold. Missing names,
singletons, and records without another comparable name have separate statuses
and null similarities. A usable name with comparable peers but no shared features
has similarity zero. `unsupported_name_count` counts usable names without
threshold support only when at least two usable names exist.

A cluster can have 100% supported names yet contain two internally consistent,
mutually dissimilar groups; inspect component counts alongside supported fraction.
The offline HTML dashboard has searchable, sortable, paginated tables. Select a
cluster in **Cluster diagnostics** (or enter an exact cluster ID) to filter its
records, name comparisons, and exact evidence together. Search can also find
clusters by member name or review reason. Headers use readable display labels;
Parquet column names are unchanged. Selecting a record focuses the shared-value
and name-comparison tables on its connections.

Pass `fusion=fusion` to `review.export_html(...)` to include source columns,
original matching rules, shared-value groups with member names/IDs, and a path
lookup explaining indirect exact connections. Without `fusion`, the dashboard
still supports name review and explicitly indicates that exact evidence was not
included. Name similarity is kept separate from the original exact-match reasons.
The notebook and example runner supply `fusion` automatically.

The **Why did these records cluster?** section includes an offline force-directed
graph. Circles are records, colored by their proposed name group; diamonds are
exact match groups labeled with their shared values. Links represent membership
in a match group, rather than every possible record pair. Hover nodes for full
names, IDs, matching conditions, and values. Selecting a record highlights its
connections; selecting a diamond filters the tables to that group's members and
evidence. The connection-path tool highlights the corresponding graph path.
Drag nodes to arrange them, drag the background to pan, scroll or use buttons to
zoom, and pause/resume the layout as needed. The graph is embedded in the HTML
and needs no CDN or additional Python package.
For larger graphs, record labels appear on hover, keyboard focus, or selected
paths to reduce clutter; use **Show record labels** to display them all.

Graph previews are bounded at 150 records, 100 match groups, and 1,500 membership
links. Any omissions are explicitly labeled; missing visual links do not imply
that records are disconnected. The tables and path lookup still use the complete
exported evidence. Node colors show name-review proposals, not accepted splits
or the reason for the original exact match.

Every supplied review row is embedded; pagination does not reduce file size.
For large inputs, restrict HTML to selected clusters and use Parquet for fuller
analysis. The example exports the ten largest evaluated clusters; an ID outside
that exported scope will not appear in the dashboard. A separate exact-evidence
report with graphs is still available.

### Accept And Persist A Reviewed Split

An accepted decision table has three columns:

| record_id | refined_cluster_id | reason |
|---|---|---|
| A | FI-NORTH | Reviewer confirmed separate institutions |
| B | FI-NORTH | Reviewer confirmed separate institutions |
| C | FI-DESERT | Reviewer confirmed separate institutions |
| D | FI-DESERT | Reviewer confirmed separate institutions |

Every current record in an affected original cluster must have an assignment.
The groups must actually split that cluster, and refined IDs cannot combine
records from different original clusters. Missing names need explicit reviewer
assignments rather than automatic placement.

```python
# decisions is an explicitly reviewed Polars DataFrame with the schema above.
decisions.write_parquet("accepted_splits.parquet")
split = apply_cluster_splits(fusion, decisions, record_id_column="record_id")
split.records.write_parquet("refined_records.parquet")
split.crossing_evidence.write_parquet("overridden_exact_evidence.parquet")
```

`split.records` preserves every input field and original cluster ID, appending
`refined_cluster_id`. Unreviewed clusters receive `original:<cluster_id>` labels;
accepted groups retain their reviewer-provided labels. These unreviewed fallback
labels are run-specific, not persistent entity IDs. `split.decisions` retains the
accepted decision table. `split.crossing_evidence` contains exact match groups
whose members now span refined groups, including the shared values and refined
labels. This explicitly exposes overrides of strong identifiers; no identifier
is automatically protected or overridden based on its name.

Reapply saved decisions explicitly after each clustering run. Assignments follow
stable record IDs rather than row order. Unknown record IDs, duplicate IDs,
new members without assignments in affected clusters, and cross-cluster merges
raise errors so changed data needs review. Decisions are not a persistent registry
or constraints enforced inside the original clustering algorithm; they are a
separate post-clustering membership assignment. Save each run's original evidence
and accepted decisions for audit history.
