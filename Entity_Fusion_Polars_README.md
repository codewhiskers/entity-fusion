# Entity Fusion — Polars (exact-match) edition

`Entity_Fusion_Polars.py` is a small, standalone entity-clustering utility built on Polars. It's a deliberately narrower sibling of [Incremental_Fusion.py](./Incremental_Fusion.py) and [Incremental_Fusion_Spark.py](./Incremental_Fusion_Spark.py):

|          | `Incremental_Fusion.py` / `..._Spark.py` | `Entity_Fusion_Polars.py`                        |
| -------- | ---------------------------------------- | ------------------------------------------------ |
| Matching | exact + fuzzy (cosine on char n-grams)   | exact only                                       |
| Scoring  | IDF-weighted, thresholds/quantiles       | none — K-of-N pass/fail                          |
| State    | incremental, persisted across batches    | one-shot, in-memory                              |
| Best for | messy text fields, evolving datasets     | clean identifiers, single large batch (~1M rows) |

Use this one when you have a dataframe and a set of columns that act as **exact-match identifiers** (tax IDs, phone numbers, normalized names/addresses, etc.) and want to cluster rows that are transitively connected through shared values — with no fuzzy tolerance, no persisted history, and a review table designed to catch bad merges before you trust them.

The `Test_Entity_Fusion_Polars.ipynb` notebook runs all of this against the ~968k-row PPP dataset in `test_data/100k.csv` end-to-end, with real output — worth reading alongside this doc.

## Install

Needs `polars` and `networkx` (both already used elsewhere in this repo).

## Quick start

```python
import polars as pl
from Entity_Fusion_Polars import EntityFusionPolars

df = pl.DataFrame([
    {"record_id": "R1", "tax_id": "TA", "phone": "PA"},
    {"record_id": "R2", "tax_id": "TA", "phone": "PA"},  # exact duplicate of R1
    {"record_id": "R3", "tax_id": "TA", "phone": "PX"},  # shares tax_id only
    {"record_id": "R4", "tax_id": "TB", "phone": "PB"},
])

linker = EntityFusionPolars(
    record_id_col="record_id",
    match_columns=["tax_id", "phone"],
    k_required=1,
)

out = linker.cluster(df)
out["records"]        # original df + hash_id + cluster_id + stable_cluster_id
out["cluster_stats"]  # one row per cluster_id, with review metrics
```

## How it works

1. **Fingerprint & dedup** — every row is hashed (`hash_id`) from its `match_columns` values (nulls are distinguished from empty strings, so a missing field can't masquerade as a real shared value). Rows with identical values collapse to one `hash_id` — they're provably interchangeable for clustering, so there's no reason to carry duplicates through the graph work.
2. **Block-size preview** — for each `match_column`, count how many distinct `hash_id`s share each value, _before_ doing anything expensive. This is exposed directly via `preview_blocks()`.
3. **Candidate pairs** — for each `match_column`, a vectorized self-join within each shared value (optionally scoped further by `block_on`, see below) produces candidate edges. No Python-level looping over blocks.
4. **K-of-N filter** — a candidate pair only survives if it agrees on at least `k_required` distinct `match_columns`. This is the main guard against one noisy column causing a bad merge.
5. **Manual decisions** — persistent `must_link` record pairs and manual identifier clusters add edges, while `cannot_link` decisions remove direct algorithmic edges before clustering. Any cannot-link pair that still lands in one component is reported in `manual_conflicts`.
6. **Clustering** — surviving edges go into a `networkx.Graph`; `nx.connected_components` gives the clusters. All `hash_id`s are added as nodes (even ones with no edges), so untouched rows show up too.
7. **cluster_id assignment** — components of size 1 (no edges survived) get `cluster_id = "UN_<hash_id>"`. Real clusters (size ≥ 2) get a deterministic sequential integer string for the current graph (`"1"`, `"2"`, ...), sorted by each component's minimum `hash_id`.
8. **stable_cluster_id assignment** — manually aliased identifier clusters use their `manual_cluster_id` as the durable label. Purely algorithmic multi-hash clusters get `AUTO_<min_hash_id>`; untouched single-hash components get `UN_<hash_id>`.
9. **Fan back out & summarize** — `cluster_id` and `stable_cluster_id` are joined back onto the original dataframe (`out["records"]`), and a per-cluster review table is built (`out["cluster_stats"]`).

## Configuration

```python
EntityFusionPolars(
    record_id_col: str,
    match_columns: list[str | dict],
    k_required: int = 1,
    block_cap: int | None = None,
)
```

| Field           | Meaning                                                                                                                                                                                                |
| --------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `record_id_col` | the non-null, unique primary key column in your input dataframe                                                                                                                                         |
| `match_columns` | columns treated as independent exact-match identifiers. Each entry is either a plain column name, or `{"column": ..., "block_on": [...]}` for a column whose matching should be scoped (see below)     |
| `k_required`    | minimum number of distinct `match_columns` that must agree before two rows link. `1` = any single shared identifier links (permissive); higher values require corroboration across multiple columns     |
| `block_cap`     | if set, a (column, value) block bigger than this is excluded from pairing entirely — a safety valve against a runaway value blowing up the self-join. Applied _within_ a column's `block_on` scope if set |

### `block_on`

A `match_columns` entry can scope its candidate generation to rows that also agree on other columns, without those columns counting toward `k_required` — purely a compute/candidate-pruning device, not a matching signal:

```python
match_columns=[
    "BorrowerAddress",
    "BorrowerZip",
    {"column": "BorrowerName", "block_on": ["BorrowerState"]},
]
```

Because `block_on` is nested inside its column's own entry, there's no separate structure that can drift out of sync with `match_columns` — a `{"column": ...}` entry with a bad/missing `"column"` key raises a `ValueError` at construction time, same as a duplicate column name.

The scope columns (the `block_on` list values, e.g. `BorrowerState`) don't need to appear in `match_columns` themselves — they're only used to narrow the self-join, never treated as identifiers.

**Gotcha:** `block_on` only affects whether two *distinct* `hash_id`s get an edge — it can't split rows that were already fingerprinted identically. Fingerprinting only looks at `match_columns` values, not `block_on` scope columns, so if two rows agree on every match column and differ only in a `block_on`-scoped column, they collapse to the same `hash_id` during dedup *before* `block_on` ever runs. `block_on` only does something useful when there's at least one other `match_column` keeping those rows as distinct nodes in the first place.

## Output

### `out["records"]`

The original dataframe, plus:

- `hash_id` — the row's identity fingerprint
- `cluster_id` — `"UN_<hash_id>"` for untouched rows, or a deterministic sequential integer string for matched clusters in the current graph
- `stable_cluster_id` — durable review label. Use this for manual vetting: manual identifier clusters keep their `manual_cluster_id`; algorithmic clusters use `AUTO_<min_hash_id>`; untouched single-hash components use `UN_<hash_id>`

### `out["cluster_stats"]`

One row per `cluster_id`:

| Column                | Meaning                                                                                                                                                                                                                                           |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `n_primary_ids`       | row count in the cluster (post fan-out, including duplicates)                                                                                                                                                                                     |
| `n_hash_ids`          | distinct identity fingerprints in the cluster                                                                                                                                                                                                     |
| `stable_cluster_id`   | durable review label for the cluster                                                                                                                                                                                                              |
| `dedup_ratio`         | `n_primary_ids / n_hash_ids` — how much raw duplication got collapsed                                                                                                                                                                             |
| `n_edges`             | distinct connected pairs holding the cluster together — algorithmic (post K-of-N, after `cannot_link` suppressions) plus any manual `must_link` edges                                                                                            |
| `n_manual_edges`      | of those, how many came from manual `must_link` decisions rather than an algorithmic match                                                                                                                                                       |
| `thin_edges`          | of the algorithmic edges, how many are backed by _exactly_ `k_required` signals — no corroboration beyond the minimum bar. (Manual edges have no `n_signals`, so they're never counted here)                                                    |
| `edge_density`        | `n_hash_ids / n_edges`. A connected graph needs ≥ `n_hash_ids - 1` edges, so this sits near its ~1 floor for a bare spanning tree (**fragile** — one bad edge could split the cluster) and drops toward 0 as redundant edges pile up (**robust**) |
| `single_edge_cluster` | `true` if a cluster of size ≥ 2 is held together by exactly one edge                                                                                                                                                                              |
| `n_distinct_<col>`    | for each `match_column`, how many distinct values appear in the cluster                                                                                                                                                                           |

### Graph evidence outputs

`edge_signals` is the algorithmic match graph edges that survived K-of-N and any manual `cannot_link` suppressions (`a`, `b` — `hash_id`s, `n_signals` — how many distinct columns they agreed on).

`manual_edges` is the full manual `must_link` graph resolved to `hash_id` pairs (`a`, `b`) from both record-pair decisions and identifier clusters. `manual_cannot_link_edges` is the manual `cannot_link` graph resolved the same way. `manual_decisions` is the resolved source-of-truth table with original record ids plus `hash_id` endpoints.

`manual_identifier_edges` is the subset of manual edges produced by `manual_identifier_clusters`. `manual_identifier_clusters` is the resolved identifier source-of-truth table: one row per `(manual_cluster_id, field, value, hash_id)` that matched current data.

`stable_cluster_conflicts` contains any connected component that inherited more than one `manual_cluster_id`. In that case, `stable_cluster_id` is set to `MANUAL_CONFLICT_<min_hash_id>` for the component so it is easy to filter and review instead of silently picking one manual label.

`manual_conflicts` contains `cannot_link` decisions that could not be fully satisfied in the final connected components:

- `same_hash_id` — the two record ids have identical `match_columns`, so they collapsed to the same `hash_id` before edges were even considered.
- `still_connected` — the direct edge was removed, but the two hash_ids are still connected through another path.

There are now four different ways to correct what `cluster()` decides, and they solve different problems:

- **`manual_decisions`** — the recommended persistent review table. Use `must_link` for true positives / known matches, and `cannot_link` for false positives / known non-matches. Pass this into every `cluster()` call so decisions survive reruns.
- **`manual_identifier_clusters`** — a persistent identifier-alias table. Use this when the reviewed truth is "these field values are aliases everywhere they appear" rather than "these two specific records match".
- **`manual_links`** — older positive-only shortcut. Rows are treated as `must_link`; keep using it if you only need manual positive links.
- **`merge_clusters()` / `split_out()`** — one-off edits to a single already-computed `records` snapshot. They don't persist and aren't replayed on the next `cluster()` call — you'd reapply them again after any re-run. Use these for a quick correction you're not planning to encode as a standing rule.

### Persistent manual review (`manual_decisions`)

Manual review decisions should be stored against source-record primary keys, not current `cluster_id` values:

```python
manual_decisions = pl.DataFrame({
    "record_id_a": ["R1", "R3"],
    "record_id_b": ["R2", "R4"],
    "decision": ["must_link", "cannot_link"],
    "reviewer": ["alice", "alice"],
    "reason": ["known same business", "shared agent address"],
    "active": [True, True],
})

out = linker.cluster(df, manual_decisions=manual_decisions)

# Any cannot-link decision that still lands in one component needs another look.
out["manual_conflicts"]
```

Accepted decision labels are `must_link` / `cannot_link`; `true_positive` and `false_positive` are accepted aliases. If an `active` column is present, only `active == True` rows are applied.

A `cannot_link` decision removes the direct algorithmic edge between the two records' `hash_id`s. It does not guess how to split a larger component if another path still connects them; those cases are reported in `manual_conflicts` so you can review the bridge that still holds the component together.

### Persistent identifier aliases (`manual_identifier_clusters`)

When the human-reviewed truth is about identifier values rather than individual records, store that as a separate identifier source-of-truth table:

```python
manual_identifier_clusters = pl.DataFrame({
    "manual_cluster_id": ["EIN_ALIAS_001", "EIN_ALIAS_001"],
    "field": ["ein", "ein"],
    "value": ["12-3456789", "98-7654321"],
    "reviewer": ["alice", "alice"],
    "reason": ["known EIN rollover", "known EIN rollover"],
    "active": [True, True],
})

out = linker.cluster(
    df,
    manual_identifier_clusters=manual_identifier_clusters,
)
```

Every current record whose `ein` is `12-3456789` is connected to every current record whose `ein` is `98-7654321` by a compact manual edge structure. New rows with either EIN will be pulled into that same component on the next run as long as you keep passing this table.

The resulting rows and cluster stats will have `stable_cluster_id = "EIN_ALIAS_001"`, so this is the ID to use for durable review workflows. The run-local `cluster_id` may still be a simple number.

The `field` column can reference any column in `df`, not only a configured `match_column`. Values are matched exactly after casting both sides to strings, so keep using the same normalization you use for exact matching.

### Linking records with no shared identifier (`manual_links`)

`match_columns`/`k_required`/`block_on` can only ever produce a link when two records agree on *something*. If two hash_ids should be linked despite sharing zero match_column values — a business that changed its EIN, say — pass `manual_links` to `cluster()`: a dataframe of `record_id_a`/`record_id_b` pairs. These bypass `k_required` entirely and participate in transitive clustering exactly like algorithmic edges (a manual link plus a real match chains together into one cluster):

```python
manual_links = pl.DataFrame({
    "record_id_a": ["R1"],
    "record_id_b": ["R2"],
})
out = linker.cluster(df, manual_links=manual_links)
```

Both `record_id_a` and `record_id_b` must already exist in `df` — an unknown `record_id` raises a `ValueError`. Prefer `manual_decisions` for new workflows; `manual_links` remains useful for positive-only compatibility.

## Correcting clusters after review

`cluster()` only knows what the match graph (plus any manual decisions) tells it. When `cluster_stats` (or manual inspection) turns up a bad merge or a missed one that you don't want to encode as a standing `manual_decisions` rule, `merge_clusters()` and `split_out()` let you correct the *output* directly instead of re-running the pipeline. Both only relabel `cluster_id` on `records` — they never touch the underlying match graph — so follow either with `recompute_cluster_stats()` to get a `cluster_stats` table that reflects the correction:

```python
out = linker.cluster(df)

# Two clusters turned out to be the same entity despite no edge connecting them:
records = linker.merge_clusters(out["records"], cluster_ids=["12", "47"])

# A cluster wrongly swept in some records (e.g. linked only by a shared
# registered-agent address) -- pull them back out:
records = linker.split_out(records, record_ids=["538365", "545074"])

# Refresh cluster_stats to reflect the corrections (reuses the original match
# graph -- no need to re-run cluster() from scratch). Pass out["manual_edges"]
# too if the original cluster() call used manual must-link decisions.
stats = linker.recompute_cluster_stats(records, out["edge_signals"], out["manual_edges"])
```

A few things worth knowing:
- `split_out` without `new_cluster_id` gives the pulled-out records their own `UN_<hash_id>`-style cluster — pass an existing `cluster_id` instead to move them into a different cluster rather than carving out a new one.
- Records that share a `hash_id` (identical values across every `match_column`) can't be split apart from each other — they're indistinguishable under the current config, so `split_out` raises a `ValueError` if you try, telling you which record_ids also need to be included.
- After a `merge_clusters`, the merged cluster's `n_edges`/`edge_density` won't show a connecting edge between its two original halves — there genuinely isn't one in the match graph. That's expected: it's what makes it a manual override rather than an algorithmic match. Likewise, after a `split_out`, an edge that used to hold the cluster together but now crosses between two different `cluster_id`s stops counting toward either side's `n_edges`.

`preview_blocks(df)` returns `(match_column, block, value, n_hash_ids)`, sorted largest-first — call it before `cluster()` on a new dataset to catch a runaway value up front.

`block` is the extra `block_on` scope, not the identifier value itself. If a
match column has no `block_on`, `block` is an empty string; that still represents a
real candidate-generation block keyed by `(match_column, value)`. With `block_on`,
the block key becomes `(match_column, block, value)`, where `block` is the joined
scope value.

## Examples

### Requiring agreement across columns (K-of-N)

Two records sharing only one of several identifiers is weaker evidence than sharing two. Raise `k_required` to demand corroboration:

```python
linker = EntityFusionPolars(
    record_id_col="record_id",
    match_columns=["tax_id", "phone", "email"],
    k_required=2,   # needs 2 of the 3 to agree
)
```

A pair sharing only `phone` no longer links; a pair sharing `phone` _and_ `email` does.

### Pre-flight check before running on a new dataset

```python
blocks = linker.preview_blocks(df)
blocks.head(10)  # largest blocks first
```

If a value's `n_hash_ids` is enormous (a placeholder like `"N/A"`, a shared mailbox address), decide whether to set `block_cap`, add a `block_on` scope, or fix the underlying data before running `cluster()`.

### Scoping a noisy column with `block_on`

Say `BorrowerName` alone is too coarse nationally, but scoping it to the same state keeps it useful:

```python
linker = EntityFusionPolars(
    record_id_col="record_id",
    match_columns=[
        "BorrowerAddress",
        "BorrowerZip",
        {"column": "BorrowerName", "block_on": ["BorrowerState"]},
    ],
    k_required=2,
)
```

Two rows now only get compared on `BorrowerName` if they're also in the same `BorrowerState` — shrinking the self-join without dropping the block entirely (which is what `block_cap` alone would do) and without making `BorrowerState` count as its own K-of-N signal.

### Reviewing a cluster before trusting it

This is a real result from running on `test_data/100k.csv` (`match_columns=["BorrowerName", "BorrowerAddress", "BorrowerZip"]`, `k_required=2`) — the largest cluster looked fine by size (56 rows) but its stats immediately flagged a problem:

```
cluster_id  n_primary_ids  n_hash_ids  n_edges  n_distinct_BorrowerName  n_distinct_BorrowerAddress  n_distinct_BorrowerZip
1           56             48          910      45                       3                            1
```

45 distinct borrower names sharing essentially one address/zip — pulling the actual rows showed a run of differently-named LLCs (`APPLE HOTEL LLC`, `CONCORD AZTEC BRICKELL LLC`, `CLA WEBSTER HOTEL OPERATORS LP`, ...) all registered at the same street address, almost certainly a shared registered-agent address rather than one real business. High `n_distinct_<name column>` alongside low `n_distinct_<address/zip column>` is exactly the shape to watch for.

## Performance

On the full `test_data/100k.csv` (968,525 rows, despite the name), `cluster()` — fingerprinting, block-size check, pairing, K-of-N, clustering, and stats — completes in **under 8 seconds**, with the largest identifier block sitting at 838 rows. See `Test_Entity_Fusion_Polars.ipynb` for the full run.
