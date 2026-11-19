"""Exact Polars entity clustering with nested rules and offline HTML review."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from html import escape
from itertools import combinations, product
import json
import math
from pathlib import Path
from typing import Union

import polars as pl


def _present(value):
    return value is not None and not (isinstance(value, float) and math.isnan(value)) and not (
        isinstance(value, str) and not value.strip()
    )


@dataclass(frozen=True, order=True)
class _FieldCondition:
    field: str
    block_on: tuple[str, ...] = ()

    @property
    def columns(self) -> tuple[str, ...]:
        return (self.field, *self.block_on)

    def as_dict(self) -> dict:
        return {"field": self.field, "block_on": list(self.block_on)}

    def describe(self) -> str:
        suffix = " [block_on: " + ", ".join(self.block_on) + "]" if self.block_on else ""
        return self.field + suffix


@dataclass(frozen=True)
class _ThresholdGroup:
    """All logical groups share threshold semantics; retain syntax for reports."""

    at_least: int
    conditions: tuple[_RuleNode, ...]
    operator: str
    path: str

    def as_dict(self) -> dict:
        children = [condition.as_dict() for condition in self.conditions]
        if self.operator == "at_least":
            return {"at_least": self.at_least, "conditions": children}
        return {self.operator: children}


_RuleNode = Union[_FieldCondition, _ThresholdGroup]
_ExactTerm = frozenset[_FieldCondition]


def _parse_rule(config: dict, path: str = "rule") -> _RuleNode:
    """Validate syntax and construct a rule tree without expanding combinations."""
    if not isinstance(config, dict):
        raise ValueError(f"{path}: condition must be a dictionary")
    if "field" in config:
        if set(config) - {"field", "block_on"}:
            raise ValueError(f"{path}: unexpected field options")
        name = config["field"]
        blockers = config.get("block_on", [])
        if not isinstance(name, str) or not name:
            raise ValueError(f"{path}: field must be a nonempty string")
        if (not isinstance(blockers, (list, tuple)) or
                any(not isinstance(c, str) or not c for c in blockers) or
                len(set(blockers)) != len(blockers) or name in blockers):
            raise ValueError(f"{path}: block_on must contain distinct other column names")
        return _FieldCondition(name, tuple(sorted(blockers)))

    operators = set(config) & {"and", "or", "at_least"}
    if len(operators) != 1:
        raise ValueError(f"{path}: use exactly one of and, or, at_least")
    operator = next(iter(operators))
    expected_keys = {"at_least", "conditions"} if operator == "at_least" else {operator}
    if set(config) != expected_keys:
        raise ValueError(f"{path}: unexpected group options")
    children = config["conditions"] if operator == "at_least" else config[operator]
    if not isinstance(children, (list, tuple)) or not children:
        raise ValueError(f"{path}: group must have nonempty conditions")
    threshold = config[operator] if operator == "at_least" else (1 if operator == "or" else len(children))
    if type(threshold) is not int or not 1 <= threshold <= len(children):
        raise ValueError(f"{path}: at_least must be an integer from 1 to the child count")
    conditions = tuple(_parse_rule(child, f"{path}.{i}") for i, child in enumerate(children))
    return _ThresholdGroup(threshold, conditions, operator, path)


def _rule_leaves(rule: _RuleNode) -> set[_FieldCondition]:
    if isinstance(rule, _FieldCondition):
        return {rule}
    return set().union(*(_rule_leaves(child) for child in rule.conditions))


@dataclass(frozen=True)
class _ExactMatchGroup:
    """One sufficient conjunction, executed by a single Polars grouping pass."""

    conditions: tuple[_FieldCondition, ...]

    @property
    def columns(self) -> tuple[str, ...]:
        return tuple(sorted({column for condition in self.conditions for column in condition.columns}))


@dataclass(frozen=True)
class _ExactMatchPlan:
    rule: _RuleNode
    groups: tuple[_ExactMatchGroup, ...]
    leaves: tuple[_FieldCondition, ...]


@dataclass
class _CompilationBudget:
    max_terms: int
    remaining_work: int

    def spend(self) -> None:
        self.remaining_work -= 1
        if self.remaining_work < 0:
            raise ValueError("rule compilation exceeds work limit; simplify the rule")


def _minimal_terms(terms: set[_ExactTerm], budget: _CompilationBudget) -> set[_ExactTerm]:
    """Discard redundant supersets, e.g. A OR (A AND B)."""
    by_size = defaultdict(set)
    for term in sorted(terms, key=len):
        smaller = (other for size, group in by_size.items() if size < len(term) for other in group)
        for other in smaller:
            budget.spend()
            if other <= term:
                break
        else:
            by_size[len(term)].add(term)
    return set().union(*by_size.values())


def _expand_exact_terms(rule: _RuleNode, budget: _CompilationBudget) -> set[_ExactTerm]:
    if isinstance(rule, _FieldCondition):
        return {frozenset([rule])}
    children = [_expand_exact_terms(child, budget) for child in rule.conditions]
    signatures = [frozenset(terms) for terms in children]
    if len(set(signatures)) != len(signatures):
        raise ValueError(f"{rule.path}: duplicate conditions would count the same evidence twice")
    terms = set()
    for selected in combinations(children, rule.at_least):
        for parts in product(*selected):
            # Count generated combinations, including duplicates, not just stored terms.
            budget.spend()
            terms.add(frozenset().union(*parts))
            if len(terms) > budget.max_terms:
                raise ValueError("rule expands to too many combinations; simplify the rule")
    return _minimal_terms(terms, budget)


def _compile_exact_rule(
    rule: _RuleNode, *, max_terms: int = 10000, max_work: int = 1000000,
) -> _ExactMatchPlan:
    """Compile a parsed tree into deterministic sufficient exact-match groups.

    Equivalent child conditions are rejected so a threshold cannot count the
    same logical evidence twice. Expansion and subsumption share a work budget
    across the whole tree, including combinations that deduplicate away.
    """
    budget = _CompilationBudget(max_terms, max_work)
    terms = _expand_exact_terms(rule, budget)
    groups = tuple(_ExactMatchGroup(tuple(sorted(term)))
                   for term in sorted(terms, key=lambda term: sorted(term)))
    return _ExactMatchPlan(rule, groups, tuple(sorted(_rule_leaves(rule))))


@dataclass
class FusionResult:
    """Clustered rows plus compact, complete evidence groups (not all pairs).

    Every pair of rows in a match group satisfies the complete rule. Overlapping
    groups explain transitive links without storing a quadratic pair table.
    row_positions always refer to zero-based positions in records.
    """

    records: pl.DataFrame
    match_groups: pl.DataFrame
    rule: dict
    cluster_column: str
    match_columns: tuple[str, ...]

    def export_html(self, path, **options):
        """See export_cluster_review for review options."""
        return export_cluster_review(self, path, **options)


def cluster_entities(
    df: pl.DataFrame,
    match_columns: Sequence[str] | None = None,
    *,
    min_matches: int = 1,
    match_rule: dict | None = None,
    cluster_column: str = "cluster_id",
    return_details: bool = False,
) -> pl.DataFrame | FusionResult:
    """Cluster exact matching records using transitive connected components.

    Supply either match_columns with min_matches, or a nested match_rule using
    field/block_on leaves and and, or, at_least/conditions groups. A blocker must
    agree and be nonmissing but does not count as a separate condition. Rules
    apply to direct pairs, then transitivity applies to their connected components.

    Null, NaN and blank string values never match. Values are otherwise exact;
    normalize identifiers/placeholder values before calling. Input rows and order
    are preserved, including singletons. Labels are earliest input row positions,
    not persistent entity IDs. Nested and object identifier types are unsupported.

    Uses Polars grouping and union-find, avoiding quadratic pair expansion.
    Complex rules expand into sufficient combinations (limited to 10,000).
    return_details=True retains compact evidence for offline HTML review.
    """
    if not isinstance(df, pl.DataFrame):
        raise TypeError("df must be a polars.DataFrame")
    if not isinstance(cluster_column, str) or not cluster_column:
        raise ValueError("cluster_column must be a nonempty string")
    if cluster_column in df.columns:
        raise ValueError(f"output column already exists: {cluster_column!r}")
    if return_details and cluster_column in {"group_id", "row_positions", "conditions", "shared_values"}:
        raise ValueError("return_details requires a cluster_column other than reserved evidence names")
    if match_rule is not None:
        if match_columns is not None or min_matches != 1:
            raise ValueError("use match_rule OR match_columns/min_matches, not both")
    else:
        if isinstance(match_columns, (str, bytes)) or not isinstance(match_columns, Sequence):
            raise TypeError("match_columns must be a sequence of column names")
        columns = list(match_columns)
        if not columns or any(not isinstance(c, str) or not c for c in columns):
            raise ValueError("match_columns must contain one or more column names")
        if len(set(columns)) != len(columns):
            raise ValueError("match_columns must not contain duplicates")
        match_rule = {"at_least": min_matches, "conditions": [{"field": c} for c in columns]}
    rule = _parse_rule(match_rule)
    plan = _compile_exact_rule(rule)
    return _execute_exact_plan(df, plan, cluster_column, return_details)


def _execute_exact_plan(
    df: pl.DataFrame, plan: _ExactMatchPlan, cluster_column: str, return_details: bool,
) -> pl.DataFrame | FusionResult:
    """Execute grouping and union-find independently of rule parsing/expansion."""
    columns = sorted({column for leaf in plan.leaves for column in leaf.columns})
    missing = set(columns) - set(df.columns)
    if missing:
        raise ValueError(f"missing match or blocker columns: {sorted(missing)}")
    for column in columns:
        dtype = df.schema[column]
        if dtype.is_nested() or dtype == pl.Object:
            raise TypeError(f"unsupported identifier type for {column!r}: {dtype}")

    parent = list(range(df.height))
    sizes = [1] * df.height
    earliest = list(range(df.height))

    def find(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left, right):
        left, right = find(left), find(right)
        if left == right:
            return
        if sizes[left] < sizes[right]:
            left, right = right, left
        parent[right] = left
        sizes[left] += sizes[right]
        earliest[left] = min(earliest[left], earliest[right])

    aliases = {c: f"key_{i}" for i, c in enumerate(columns)}
    keys = df.select(pl.col(c).alias(a) for c, a in aliases.items()).with_row_index("row_position")
    valid = {}
    for column, alias in aliases.items():
        expr = pl.col(alias).is_not_null()
        dtype = df.schema[column]
        if dtype.is_float():
            expr = expr & ~pl.col(alias).is_nan()
        elif dtype in (pl.String, pl.Categorical) or isinstance(dtype, pl.Enum):
            expr = expr & (pl.col(alias).cast(pl.String).str.strip_chars() != "")
        valid[column] = expr

    evidence = []
    if df.height > 1:
        for match_group in plan.groups:
            fields = match_group.columns
            groups = (keys.filter(pl.all_horizontal([valid[c] for c in fields]))
                      .group_by([aliases[c] for c in fields])
                      .agg(pl.col("row_position"))
                      .filter(pl.col("row_position").list.len() > 1))
            for group in groups.iter_rows(named=True):
                members = sorted(group["row_position"])
                for member in members[1:]:
                    union(members[0], member)
                if return_details:
                    evidence.append({
                        "row_positions": members,
                        "conditions": [condition.describe() for condition in match_group.conditions],
                        "shared_values": json.dumps({c: group[aliases[c]] for c in fields}, default=str),
                    })
    labels = [earliest[find(i)] for i in range(df.height)]
    records = df.with_columns(pl.Series(cluster_column, labels, dtype=pl.UInt64))
    if not return_details:
        return records
    evidence.sort(key=lambda e: (e["row_positions"], e["conditions"]))
    for group_id, group in enumerate(evidence):
        group["group_id"] = group_id
        group[cluster_column] = labels[group["row_positions"][0]]
    schema = {"group_id": pl.UInt64, cluster_column: pl.UInt64,
              "row_positions": pl.List(pl.UInt64), "conditions": pl.List(pl.String),
              "shared_values": pl.String}
    return FusionResult(records, pl.DataFrame(evidence, schema=schema), plan.rule.as_dict(), cluster_column,
                        tuple(sorted({leaf.field for leaf in plan.leaves})))


def _bridges(adjacency):
    """Iterative Tarjan traversal; only used on complete bounded review graphs."""
    discovery, low, bridges = {}, {}, []
    tick = 0
    for root in adjacency:
        if root in discovery:
            continue
        discovery[root] = low[root] = tick
        tick += 1
        stack = [(root, None, iter(sorted(adjacency[root])))]
        while stack:
            node, parent, neighbors = stack[-1]
            child = next(neighbors, None)
            if child is None:
                stack.pop()
                if parent is not None:
                    low[parent] = min(low[parent], low[node])
                    if low[node] > discovery[parent]:
                        bridges.append(tuple(sorted((parent, node))))
            elif child != parent:
                if child in discovery:
                    low[node] = min(low[node], discovery[child])
                else:
                    discovery[child] = low[child] = tick
                    tick += 1
                    stack.append((child, node, iter(sorted(adjacency[child]))))
    return sorted(bridges)


def export_cluster_review(
    result: FusionResult,
    path,
    *,
    top_n: int = 10,
    min_cluster_size: int = 2,
    max_records: int = 100,
    max_groups: int = 200,
    record_id_column: str | None = None,
    display_columns: Sequence[str] | None = None,
    conflict_columns: Sequence[str] | None = None,
    common_value_min_count: int = 5,
) -> Path:
    """Write a self-contained HTML report of the largest qualifying clusters.

    Frequencies and conflicts use FULL clusters; record/group tables are capped
    deterministic samples. Graphs and bridge analysis require a complete cluster
    of at most max_records rows. Path lookup uses complete compact evidence even
    for sampled clusters. Report embeds identifier values; choose display_columns
    to limit the record table (match evidence still includes matching values).
    Conflict flags are review hints, never automatic cannot-link rules.
    """
    if not isinstance(result, FusionResult):
        raise TypeError("use cluster_entities(..., return_details=True) first")
    for name, value in [("top_n", top_n), ("min_cluster_size", min_cluster_size),
                        ("max_records", max_records), ("max_groups", max_groups),
                        ("common_value_min_count", common_value_min_count)]:
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    df = result.records
    display = list(df.columns if display_columns is None else display_columns)
    conflicts = list(result.match_columns if conflict_columns is None else conflict_columns)
    for selected in (display, conflicts):
        if any(c not in df.columns for c in selected):
            raise ValueError("unknown display/conflict column")
    if record_id_column is not None and record_id_column not in df.columns:
        raise ValueError("unknown record_id_column")
    members = defaultdict(list)
    for position, label in enumerate(df[result.cluster_column]):
        members[label].append(position)
    selected = sorted(((label, rows) for label, rows in members.items()
                       if len(rows) >= min_cluster_size), key=lambda item: (-len(item[1]), item[0]))[:top_n]
    groups_by_cluster = defaultdict(list)
    for group in result.match_groups.iter_rows(named=True):
        groups_by_cluster[group[result.cluster_column]].append(group)

    def table(headers, rows):
        return ('<table><thead><tr>' + ''.join('<th>' + escape(str(h)) + '</th>' for h in headers)
                + '</tr></thead><tbody>' + ''.join('<tr>' + ''.join(
                    '<td>' + escape(str(v)) + '</td>' for v in row) + '</tr>' for row in rows)
                + '</tbody></table>')

    panels, payload = [], []
    for index, (label, positions) in enumerate(selected):
        cluster = df[positions]
        conflicting, frequency = {}, []
        for column in sorted(set(result.match_columns) | set(conflicts)):
            if cluster.schema[column].is_nested() or cluster.schema[column] == pl.Object:
                raise TypeError("review conflict columns must contain scalar values")
            counts = Counter(value for value in cluster[column].to_list() if _present(value))
            if column in conflicts and len(counts) > 1:
                conflicting[column] = len(counts)
            if column in result.match_columns:
                frequency.extend((column, str(value), count) for value, count in counts.items()
                                 if count >= common_value_min_count)
        frequency.sort(key=lambda row: (-row[2], row[0], row[1]))
        groups = groups_by_cluster[label]
        labels = {p: str(df[record_id_column][p]) if record_id_column else str(p) for p in positions}
        adjacency = {p: set() for p in positions}
        complete = len(positions) <= max_records
        if complete:
            for group in groups:
                for left, right in combinations(group["row_positions"], 2):
                    adjacency[left].add(right)
                    adjacency[right].add(left)
        bridges = _bridges(adjacency) if complete else []
        graph_edges = [(a, b) for a in positions for b in sorted(adjacency[a]) if a < b] if complete else []
        data_groups = [{"id": g["group_id"], "members": g["row_positions"],
                        "reason": ' + '.join(g["conditions"]),
                        "values": g["shared_values"]} for g in groups]
        payload.append({"labels": labels, "groups": data_groups,
                        "positions": positions if complete else [], "edges": graph_edges,
                        "bridges": bridges})
        rows_html = []
        for p in positions[:max_records]:
            row = df.row(p, named=True)
            cells = [f'<td>{p}</td>']
            for c in display:
                value = row[c]
                css = 'missing' if not _present(value) else ('conflict' if c in conflicting else '')
                cells.append(f'<td class="{css}">{escape(str(value))}</td>')
            rows_html.append('<tr>' + ''.join(cells) + '</tr>')
        record_table = ('<table><thead><tr><th>Row position</th>' + ''.join(
            '<th>' + escape(c) + '</th>' for c in display) + '</tr></thead><tbody>'
                        + ''.join(rows_html) + '</tbody></table>')
        evidence_table = table(["Group", "Rule evidence", "Shared values (including blockers)", "Members"],
                               [(g["group_id"], ' + '.join(g["conditions"]), g["shared_values"],
                                 ', '.join(map(str, g["row_positions"][:max_records])) +
                                 (' …' if len(g["row_positions"]) > max_records else ''))
                                for g in groups[:max_groups]])
        flag_text = ', '.join(f'{c}: {n} distinct values' for c, n in conflicting.items()) or 'None'
        graph = (f'<svg id="graph-{index}" viewBox="0 0 700 540" role="img" aria-label="Direct match graph"></svg>'
                 if complete else '<p>Graph and bridge analysis omitted: cluster exceeds max_records. '
                 'No bridge conclusions are drawn from a sample.</p>')
        panels.append(f'''<details class="cluster" {'open' if index == 0 else ''}>
<summary>Cluster {label} — {len(positions)} records · {len(conflicting)} conflicting fields · {len(frequency)} common values</summary>
<p>Distinct identifier flags (review hints): {escape(flag_text)}.</p>
<p>Complete cluster counts; showing first {min(len(positions), max_records)} of {len(positions)} records
and {min(len(groups), max_groups)} of {len(groups)} evidence groups. Member lists are capped at {max_records}.</p>
<h3>Identifier frequencies</h3>{table(['Field', 'Value', 'Record count'], frequency) if frequency else '<p>No values meet the common-value threshold.</p>'}
<h3>Records</h3><p>Yellow: conflicting field. Gray: missing value.</p><div class="scroll">{record_table}</div>
<h3>Why did these records cluster?</h3>
<p>Enter zero-based row positions from the record table. Paths use full cluster evidence, including records outside the displayed sample.
Record IDs appear as labels; row positions disambiguate duplicates.</p>
<label>From <input id="from-{index}" type="number" min="0" value="{positions[0]}"></label>
<label>To <input id="to-{index}" type="number" min="0" value="{positions[-1]}"></label>
<button data-path="{index}">Explain path</button><pre id="path-{index}" aria-live="polite"></pre>
{graph}<p>{'Complete direct-match graph; red edges are true graph bridges. Hover edges for evidence and nodes for row labels.' if complete else ''}</p>
<p>{('Bridge links: ' + escape(str(bridges)) + '. A bridge is the sole direct connection between two groups; it is a review hint, not proof of error.') if complete else ''}</p>
<h3>Match evidence</h3><p>Each group represents a complete set of mutually linked records satisfying the rule through the listed conditions.
Blockers constrain a condition; they do not add votes. Groups overlap to create transitive clusters.</p>
<div class="scroll">{evidence_table}</div></details>''')
    serialized = json.dumps(payload, default=str).replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
    html = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Entity fusion cluster review</title><style>
body{font:15px system-ui,sans-serif;margin:24px auto;max-width:1200px;padding:0 16px;color:#183047;background:#f5f7fa}
h1,h3{color:#183047}details{background:white;border:1px solid #ccd5df;border-radius:8px;padding:16px;margin:16px 0}
summary{font-weight:bold;cursor:pointer}.scroll{overflow:auto;max-height:520px}table{border-collapse:collapse;width:100%;font-size:13px}
th,td{border:1px solid #d8e0e8;padding:8px;text-align:left;white-space:pre-wrap;overflow-wrap:anywhere}th{background:#eaf0f6;position:sticky;top:0}
.conflict{background:#fff0ba}.missing{background:#eee;color:#727272}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#eef3f8;padding:12px}
input{width:90px;margin:8px}button{padding:8px;cursor:pointer}svg{width:100%;max-height:540px}line{stroke:#92a9bc;stroke-opacity:.55}circle{fill:#245c89}text{font-size:11px}
</style><h1>Entity fusion cluster review</h1>'''
    html += (f'<p>Largest {len(selected)} clusters with at least {min_cluster_size} records; ranked by size, then cluster ID. '
             f'Input: {df.height} records, {len(members)} clusters. Common-value threshold: {common_value_min_count} records per cluster.</p>'
             '<p>All matching is exact and pairwise before transitive clustering. Conflicting IDs do not automatically block links. '
             'This offline report embeds record and match values; no external scripts or services are used.</p>'
             '<details><summary>Configured rule</summary><pre>' + escape(json.dumps(result.rule, indent=2)) + '</pre></details>')
    html += ''.join(panels) or '<p>No clusters meet the requested size.</p>'
    html += '<script type="application/json" id="review-data">' + serialized + '</script>'
    html += r'''<script>
const data=JSON.parse(document.getElementById('review-data').textContent);
function explain(i){
 const d=data[i],a=Number(document.getElementById('from-'+i).value),b=Number(document.getElementById('to-'+i).value),out=document.getElementById('path-'+i);
 if(!Object.hasOwn(d.labels,a)||!Object.hasOwn(d.labels,b)){out.textContent='Both positions must belong to this cluster.';return;}
 const byRow=new Map();d.groups.forEach(g=>g.members.forEach(p=>{if(!byRow.has(p))byRow.set(p,[]);byRow.get(p).push(g);}));
 const seen=new Map([[a,null]]),used=new Set(),queue=[a];let head=0;
 while(head<queue.length&&!seen.has(b)){const p=queue[head++];for(const g of byRow.get(p)||[]){if(used.has(g.id))continue;used.add(g.id);for(const q of g.members){if(!seen.has(q)){seen.set(q,{p,g});queue.push(q);}}}}
 if(!seen.has(b)){out.textContent='No path found.';return;}
 const steps=[];let p=b;while(p!==a){const s=seen.get(p);steps.push('Row '+s.p+' ('+d.labels[s.p]+') → Row '+p+' ('+d.labels[p]+')\n  '+s.g.reason+'\n  '+s.g.values);p=s.p;}
 out.textContent=steps.reverse().join('\n\n')||'Same record.';
}
document.querySelectorAll('[data-path]').forEach(el=>el.addEventListener('click',()=>explain(Number(el.dataset.path))));
const ns='http://www.w3.org/2000/svg';
function element(tag,attrs,title){const el=document.createElementNS(ns,tag);for(const[k,v]of Object.entries(attrs))el.setAttribute(k,v);if(title){const t=document.createElementNS(ns,'title');t.textContent=title;el.append(t);}return el;}
data.forEach((d,i)=>{
 const svg=document.getElementById('graph-'+i);if(!svg)return;
 const points=new Map(d.positions.map((p,k)=>{const angle=k*2*Math.PI/d.positions.length;return[p,[350+235*Math.cos(angle),270+220*Math.sin(angle)]];}));
 const bridges=new Set(d.bridges.map(e=>e.join(',')));
 for(const[a,b]of d.edges){const[x1,y1]=points.get(a),[x2,y2]=points.get(b);const evidence=d.groups.filter(g=>g.members.includes(a)&&g.members.includes(b)).map(g=>g.reason+': '+g.values).join('\n');const line=element('line',{x1,y1,x2,y2},'Rows '+a+' ↔ '+b+'\n'+evidence);if(bridges.has([a,b].join(',')))line.setAttribute('style','stroke:#c93434;stroke-width:3;stroke-opacity:1');svg.append(line);}
 for(const[p,[x,y]]of points){svg.append(element('circle',{cx:x,cy:y,r:6},'Row '+p+': '+d.labels[p]));const text=element('text',{x:x+8,y:y+4});text.textContent=p;svg.append(text);}
});
</script></html>'''
    destination = Path(path)
    destination.write_text(html, encoding="utf-8")
    return destination
