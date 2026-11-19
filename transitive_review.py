"""Optional name diagnostics and explicit manual splits for exact fusion results.

Clustering remains Polars-only. Name evaluation additionally requires
scikit-learn and sparse_dot_topn (v1 API). Similarity never changes membership.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
import math
from pathlib import Path

import polars as pl

from transitive_fusion import FusionResult, _present


def _validate_ids(records, column):
    if column not in records.columns:
        raise ValueError(f"missing record ID column: {column}")
    values = records[column].to_list()
    if records.schema[column].is_nested() or records.schema[column] == pl.Object:
        raise ValueError("record IDs must be scalar")
    if any(not _present(v) for v in values) or len(set(values)) != len(values):
        raise ValueError("record IDs must be unique and nonmissing")
    return values


@dataclass
class NameReview:
    records: pl.DataFrame
    clusters: pl.DataFrame
    name_pairs: pl.DataFrame
    proposed_splits: pl.DataFrame
    configuration: dict

    def export_html(self, path, *, fusion: FusionResult | None = None):
        """Export a searchable offline dashboard. Pass fusion for exact evidence.

        The export contains every supplied review row. Select a subset before
        export to bound file size. fusion must be the matching clustering result;
        its original columns and shared-value evidence enrich the selected rows.
        """
        config = dict(self.configuration)
        record_id_column = config['record_id_column']
        name_column = config['name_column']
        source_rows, groups = {}, []
        reviewed_ids = set(self.records['record_id'].to_list())
        selected_clusters = set(self.clusters['cluster_id'].to_list())
        source_columns = []
        if fusion is not None:
            ids = _validate_ids(fusion.records, record_id_column)
            if not reviewed_ids <= set(ids):
                raise ValueError('fusion is missing reviewed record IDs')
            source_columns = [c for c in fusion.records.columns
                              if c not in (record_id_column, name_column, fusion.cluster_column)]
            reviewed_labels = dict(self.records.select('record_id', 'cluster_id').iter_rows())
            for row in fusion.records.iter_rows(named=True):
                rid = row[record_id_column]
                if rid in reviewed_ids:
                    if row[fusion.cluster_column] != reviewed_labels[rid]:
                        raise ValueError('fusion memberships differ from the reviewed clusters')
                    source_rows[str(rid)] = {c: row[c] for c in source_columns}
            for group in fusion.match_groups.filter(
                pl.col(fusion.cluster_column).is_in(list(selected_clusters))
            ).iter_rows(named=True):
                groups.append({
                    'cluster_id': str(group[fusion.cluster_column]),
                    'group_id': str(group['group_id']),
                    'members': [str(ids[p]) for p in group['row_positions']],
                    'conditions': group['conditions'],
                    'shared_values': json.loads(group['shared_values']),
                })
        def rows(frame, id_columns):
            output = []
            for row in frame.iter_rows(named=True):
                for c in id_columns:
                    if row[c] is not None:
                        row[c] = str(row[c])
                output.append(row)
            return output
        payload = {
            'configuration': config,
            'clusters': rows(self.clusters, ['cluster_id']),
            'records': rows(self.records, ['record_id', 'cluster_id', 'best_name_neighbor_id', 'proposed_cluster_id']),
            'pairs': rows(self.name_pairs, ['cluster_id', 'record_id_a', 'record_id_b']),
            'groups': groups, 'source_rows': source_rows, 'source_columns': source_columns,
            'has_exact_evidence': fusion is not None,
            'exact_rule': fusion.rule if fusion is not None else None,
        }
        def json_safe(value):
            if isinstance(value, float) and not math.isfinite(value):
                return None
            if isinstance(value, dict):
                return {k: json_safe(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [json_safe(v) for v in value]
            return value
        data = json.dumps(json_safe(payload), default=str, allow_nan=False).replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
        template = Path(__file__).with_name('transitive_review_dashboard.html').read_text(encoding='utf-8')
        graph_script = Path(__file__).with_name('transitive_review_graph.js').read_text(encoding='utf-8')
        template = template.replace('/* MATCH_GRAPH_SCRIPT */', graph_script)
        destination = Path(path)
        destination.write_text(template.replace('<!-- REVIEW_DATA -->', data), encoding='utf-8')
        return destination

def evaluate_clusters(
    result: FusionResult, *, name_column: str, record_id_column: str,
    threshold: float = 0.85, top_n: int = 10, complete_limit: int = 100,
    min_cluster_size: int = 2, max_clusters: int | None = None,
) -> NameReview:
    """Evaluate names only within existing clusters and propose name components.

    TF-IDF char_wb (2,4), L2 normalization, fitted over usable names in selected
    clusters; comparisons use lowercase whitespace-normalized names. Scores
    depend on that corpus and are not probabilities. Original names are retained.
    For clusters <= complete_limit, all positive pair scores are retrieved.
    Larger clusters retrieve top_n neighbors per row (self excluded), symmetrized
    by union. Best scores below the threshold are retained for diagnostics.
    Missing/empty names and names without vector features have separate statuses.
    Each such record gets its own proposed group, never an automatic split.
    CSR feature presence is computed once; two-name comparisons are batched and
    singletons skip similarity search. All selected usable names still participate
    in TF-IDF fitting, including singletons when requested.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sparse_dot_topn import sp_matmul_topn

    if not isinstance(result, FusionResult):
        raise TypeError('result must be a FusionResult')
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not 0 < threshold <= 1:
        raise ValueError('threshold must be > 0 and <= 1')
    for key, value in [('top_n', top_n), ('complete_limit', complete_limit), ('min_cluster_size', min_cluster_size)]:
        if type(value) is not int or value < 1:
            raise ValueError(f'{key} must be a positive integer')
    if max_clusters is not None and (type(max_clusters) is not int or max_clusters < 1):
        raise ValueError('max_clusters must be a positive integer or None')
    df = result.records
    ids = _validate_ids(df, record_id_column)
    if name_column not in df.columns or df.schema[name_column] != pl.String:
        raise ValueError('name_column must be a string column')
    cluster_col = result.cluster_column
    # Review tables use fixed names so they export independently of input naming.
    original_labels = df[cluster_col].to_list()
    names = df[name_column].to_list()
    memberships = defaultdict(list)
    for pos, label in enumerate(original_labels):
        memberships[label].append(pos)
    selected = sorted(((label, rows) for label, rows in memberships.items() if len(rows) >= min_cluster_size),
                      key=lambda item: (-len(item[1]), item[0]))
    if max_clusters is not None:
        selected = selected[:max_clusters]
    usable = [p for _, rows in selected for p in rows if _present(names[p])]
    index = {p: i for i, p in enumerate(usable)}
    vectors = None
    if usable:
        corpus = [' '.join(names[p].lower().split()) for p in usable]
        vectors = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4), lowercase=False).fit_transform(corpus)
    # Inspect CSR row pointers once instead of allocating a sparse row per record.
    feature_positions = {p for i, p in enumerate(usable)
                         if vectors.indptr[i + 1] > vectors.indptr[i]}
    # Two-name clusters need one dot product, not a separate top-N search each.
    # Batch them to keep sparse slicing overhead and temporary memory bounded.
    two_name_clusters = []
    for label, positions in selected:
        if len(positions) > 1:
            active = [p for p in positions if p in feature_positions]
            if len(active) == 2:
                two_name_clusters.append((label, index[active[0]], index[active[1]]))
    two_name_scores = {}
    for start in range(0, len(two_name_clusters), 4096):
        batch = two_name_clusters[start:start + 4096]
        left = vectors[[a for _, a, _ in batch]]
        right = vectors[[b for _, _, b in batch]]
        scores = left.multiply(right).sum(axis=1).A1
        two_name_scores.update((label, min(1.0, max(0.0, float(score))))
                               for (label, _, _), score in zip(batch, scores))
    record_rows, cluster_rows, pair_rows, proposals = [], [], [], []
    for label, positions in selected:
        if len(positions) == 1:
            p = positions[0]
            has_features = p in feature_positions
            status = ('missing_name' if not _present(names[p]) else
                      'singleton' if has_features else 'no_features')
            proposed_id = f'name:{label}:0'
            record_rows.append({'record_id': ids[p], 'cluster_id': label, 'name': names[p],
                                'best_name_neighbor_id': None, 'best_name_similarity': None,
                                'name_status': status, 'has_neighbor_above_threshold': False,
                                'proposed_cluster_id': proposed_id})
            proposals.append({'record_id': ids[p], 'cluster_id': label,
                              'proposed_cluster_id': proposed_id,
                              'reason': 'name_similarity_proposal; requires review'})
            cluster_rows.append({'cluster_id': label, 'record_count': 1,
                                 'usable_name_count': int(has_features), 'name_threshold': threshold,
                                 'unsupported_name_count': 0, 'supported_name_fraction': None,
                                 'name_component_count': int(has_features), 'proposed_component_count': 1,
                                 'graph_complete': True, 'requires_review': not has_features,
                                 'review_reasons': [] if has_features else ['missing_or_unusable_names']})
            continue
        active = [p for p in positions if p in feature_positions]
        active_set = set(active)
        parent = {p: p for p in positions}
        def find(p):
            while parent[p] != p:
                parent[p] = parent[parent[p]]
                p = parent[p]
            return p
        best = {p: (None, None) for p in positions}
        supported = set()
        complete = len(positions) <= complete_limit
        pairs = {}
        if len(active) == 2:
            score = two_name_scores[label]
            if score > 0:
                pairs[(active[0], active[1])] = score
        elif len(active) > 2:
            matrix = vectors[[index[p] for p in active]].tocsr()
            scores = sp_matmul_topn(matrix, matrix.T.tocsr(),
                                   top_n=len(active) if complete else min(len(active), top_n + 1),
                                   threshold=0.0, sort=True)
            for i, p in enumerate(active):
                neighbors = [(active[int(j)], min(1.0, max(0.0, float(s))))
                             for j, s in zip(scores.indices[scores.indptr[i]:scores.indptr[i + 1]],
                                             scores.data[scores.indptr[i]:scores.indptr[i + 1]]) if int(j) != i]
                neighbors.sort(key=lambda item: (-item[1], item[0]))
                if not complete:
                    neighbors = neighbors[:top_n]
                for q, score in neighbors:
                    pair = tuple(sorted((p, q)))
                    pairs[pair] = max(pairs.get(pair, 0.0), score)
        for (p, q), score in sorted(pairs.items()):
            accepted = score >= threshold - 1e-12
            pair_rows.append({'cluster_id': label, 'record_id_a': ids[p], 'record_id_b': ids[q],
                              'name_similarity': score, 'meets_threshold': accepted})
            for a, b in [(p, q), (q, p)]:
                old_neighbor, old_score = best[a]
                if old_score is None or score > old_score or (score == old_score and b < old_neighbor):
                    best[a] = (b, score)
            if accepted:
                supported.update((p, q))
                left, right = find(p), find(q)
                parent[max(left, right)] = min(left, right)
        components = sorted({find(p) for p in positions})
        component_ids = {root: f'name:{label}:{i}' for i, root in enumerate(components)}
        for p in positions:
            neighbor, score = best[p]
            status = ('missing_name' if not _present(names[p]) else
                      'no_features' if p not in active_set else
                      'singleton' if len(positions) == 1 else
                      'no_comparable_name' if len(active) < 2 else
                      'supported' if p in supported else 'below_threshold')
            # With another comparable name but no shared n-grams, cosine is zero.
            if status == 'below_threshold' and score is None:
                score = 0.0
                neighbor = next(q for q in active if q != p)
            record_rows.append({'record_id': ids[p], 'cluster_id': label, 'name': names[p],
                                'best_name_neighbor_id': ids[neighbor] if neighbor is not None else None,
                                'best_name_similarity': score, 'name_status': status,
                                'has_neighbor_above_threshold': p in supported,
                                'proposed_cluster_id': component_ids[find(p)]})
            proposals.append({'record_id': ids[p], 'cluster_id': label,
                              'proposed_cluster_id': component_ids[find(p)],
                              'reason': 'name_similarity_proposal; requires review'})
        unsupported = len(active) - len(supported) if len(active) > 1 else 0
        reasons = []
        name_components = len({find(p) for p in active})
        if name_components > 1:
            reasons.append('multiple_name_components')
        if len(active) < len(positions):
            reasons.append('missing_or_unusable_names')
        if unsupported:
            reasons.append('names_without_threshold_support')
        if len(active) < 2 and len(positions) > 1:
            reasons.append('insufficient_comparable_names')
        cluster_rows.append({'cluster_id': label, 'record_count': len(positions),
                             'usable_name_count': len(active), 'name_threshold': threshold,
                             'unsupported_name_count': unsupported,
                             'supported_name_fraction': len(supported) / len(active) if len(active) > 1 else None,
                             'name_component_count': name_components,
                             'proposed_component_count': len(components),
                             'graph_complete': complete, 'requires_review': bool(reasons), 'review_reasons': reasons})
    id_type, cluster_type = df.schema[record_id_column], df.schema[cluster_col]
    def frame(rows, schema):
        return pl.DataFrame(rows, schema=schema)
    return NameReview(
        frame(record_rows, {'record_id': id_type, 'cluster_id': cluster_type, 'name': pl.String,
                           'best_name_neighbor_id': id_type, 'best_name_similarity': pl.Float64,
                           'name_status': pl.String, 'has_neighbor_above_threshold': pl.Boolean,
                           'proposed_cluster_id': pl.String}),
        frame(cluster_rows, {'cluster_id': cluster_type, 'record_count': pl.UInt64, 'usable_name_count': pl.UInt64,
                            'name_threshold': pl.Float64, 'unsupported_name_count': pl.UInt64,
                            'supported_name_fraction': pl.Float64, 'name_component_count': pl.UInt64,
                            'proposed_component_count': pl.UInt64,
                            'graph_complete': pl.Boolean, 'requires_review': pl.Boolean,
                            'review_reasons': pl.List(pl.String)}),
        frame(pair_rows, {'cluster_id': cluster_type, 'record_id_a': id_type, 'record_id_b': id_type,
                         'name_similarity': pl.Float64, 'meets_threshold': pl.Boolean}),
        frame(proposals, {'record_id': id_type, 'cluster_id': cluster_type, 'proposed_cluster_id': pl.String,
                          'reason': pl.String}),
        {'name_column': name_column, 'record_id_column': record_id_column, 'threshold': threshold,
         'top_n': top_n, 'complete_limit': complete_limit, 'min_cluster_size': min_cluster_size,
         'max_clusters': max_clusters, 'evaluated_clusters': len(selected), 'input_clusters': len(memberships),
         'method': 'TF-IDF char_wb (2,4), corpus = usable names in selected clusters; cosine; symmetric union top-N',
         'component_policy': 'threshold connected components; missing names proposed separately; no membership changes'},
    )


@dataclass
class SplitResult:
    records: pl.DataFrame
    decisions: pl.DataFrame
    crossing_evidence: pl.DataFrame


def apply_cluster_splits(
    result: FusionResult, decisions: pl.DataFrame, *, record_id_column: str,
    refined_column: str = 'refined_cluster_id',
) -> SplitResult:
    """Apply explicit reviewed memberships without modifying original labels.

    decisions columns: record_id, refined_cluster_id (string), reason (string).
    Include EVERY current record in each affected original cluster. At least two
    refined groups per affected cluster are required. Groups cannot span original
    clusters. Save decisions as Parquet and reapply by stable IDs on future runs;
    changed membership fails closed for review. No decisions are applied implicitly.
    """
    if not isinstance(result, FusionResult) or not isinstance(decisions, pl.DataFrame):
        raise TypeError('expected FusionResult and Polars decisions DataFrame')
    df = result.records
    if not isinstance(refined_column, str) or not refined_column or refined_column in df.columns:
        raise ValueError('refined_column must be a new nonempty column name')
    ids = _validate_ids(df, record_id_column)
    required = {'record_id', 'refined_cluster_id', 'reason'}
    if not required <= set(decisions.columns):
        raise ValueError('decisions require record_id, refined_cluster_id, reason')
    decision_ids = _validate_ids(decisions, 'record_id')
    if decisions.schema['record_id'] != df.schema[record_id_column]:
        raise ValueError('decision record IDs must have the same dtype as input record IDs')
    for col in ('refined_cluster_id', 'reason'):
        if decisions.schema[col] != pl.String or any(not _present(v) for v in decisions[col]):
            raise ValueError(f'{col} must contain nonempty strings')
    known = set(ids)
    if set(decision_ids) - known:
        raise ValueError('decisions reference unknown records')
    assigned = dict(zip(decision_ids, decisions['refined_cluster_id']))
    original = df[result.cluster_column].to_list()
    members = defaultdict(list)
    for rid, cluster in zip(ids, original):
        members[cluster].append(rid)
    affected = {cluster for rid, cluster in zip(ids, original) if rid in assigned}
    for cluster in affected:
        if any(rid not in assigned for rid in members[cluster]):
            raise ValueError('decisions must cover every current record in each affected cluster')
        if len({assigned[rid] for rid in members[cluster]}) < 2:
            raise ValueError('each affected cluster must be split into at least two groups')
    refined = [assigned[rid] if rid in assigned else 'original:' + json.dumps(cluster, default=str)
               for rid, cluster in zip(ids, original)]
    owner = {}
    for label, cluster in zip(refined, original):
        if label in owner and owner[label] != cluster:
            raise ValueError('refined IDs must not merge different original clusters')
        owner[label] = cluster
    crossing = []
    for group in result.match_groups.iter_rows(named=True):
        labels = sorted({refined[p] for p in group['row_positions']})
        if len(labels) > 1:
            crossing.append({**group, 'refined_cluster_ids': labels})
    evidence_schema = {**result.match_groups.schema, 'refined_cluster_ids': pl.List(pl.String)}
    return SplitResult(df.with_columns(pl.Series(refined_column, refined, dtype=pl.String)),
                       decisions.clone(), pl.DataFrame(crossing, schema=evidence_schema))
