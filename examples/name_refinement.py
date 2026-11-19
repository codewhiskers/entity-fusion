"""Run a synthetic reviewed split, or review names in the local PPP Parquet file.

python examples/name_refinement.py
python examples/name_refinement.py --input test_data/100k.parquet
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import polars as pl
from transitive_fusion import cluster_entities
from transitive_review import evaluate_clusters, apply_cluster_splits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, help='Optional PPP Parquet file; otherwise use synthetic records')
    parser.add_argument('--record-id-column', help='Stable unique source ID column, if available')
    parser.add_argument('--output-dir', type=Path, default=Path('test_data/_debug_outputs/name_refinement'))
    parser.add_argument('--threshold', type=float, default=.85)
    parser.add_argument('--all-clusters', action='store_true',
                        help='Evaluate all nonsingleton clusters; HTML still shows only the ten largest')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    if args.input:
        df = pl.read_parquet(args.input)
        if args.record_id_column:
            record_id_column = args.record_id_column
        else:
            record_id_column = '_demo_record_id'
            df = df.with_row_index(record_id_column)
        name_column = 'BorrowerName'
        rule = {'or': [{'field': c, 'block_on': ['BorrowerCity', 'BorrowerState']}
                       for c in ('BorrowerName', 'BorrowerAddress')]}
        prefix = 'ppp'
    else:
        df = pl.DataFrame({
            'record_id': list('ABCDEF'), 'identifier': ['shared'] * 5 + ['other'],
            'name': ['NORTH HARBOR CREDIT UNION', 'north harbor credit union',
                     'DESERT VALLEY FINANCE', 'DESERT VALLEY FINANCE', None, 'NORTH HARBOR CREDIT UNION'],
        })
        record_id_column, name_column = 'record_id', 'name'
        rule = {'field': 'identifier'}
        prefix = 'synthetic'
    result = cluster_entities(df, match_rule=rule, return_details=True)
    exact_seconds = perf_counter() - started
    print(f'Exact clustering complete: {df.height:,} rows in {exact_seconds:.2f}s', flush=True)
    started = perf_counter()
    review = evaluate_clusters(result, name_column=name_column, record_id_column=record_id_column,
                               threshold=args.threshold, max_clusters=None if args.all_clusters else 10)
    review_seconds = perf_counter() - started
    print(f'Name evaluation complete: {review.clusters.height:,} clusters in {review_seconds:.2f}s', flush=True)
    for title, table in [('records', review.records), ('clusters', review.clusters),
                         ('name_pairs', review.name_pairs), ('proposed_splits', review.proposed_splits)]:
        table.write_parquet(args.output_dir / f'{prefix}_{title}.parquet')
    result.records.write_parquet(args.output_dir / f'{prefix}_labeled_records.parquet')
    result.match_groups.write_parquet(args.output_dir / f'{prefix}_match_groups.parquet')
    # Keep the HTML bounded while exporting complete review tables as Parquet.
    selected_ids = review.clusters['cluster_id'].head(10).to_list()
    report_review = replace(
        review,
        **{name: getattr(review, name).filter(pl.col('cluster_id').is_in(selected_ids))
           for name in ('records', 'clusters', 'name_pairs', 'proposed_splits')},
        configuration={**review.configuration, 'html_scope': 'ten largest evaluated clusters; Parquet contains all evaluated rows'},
    )
    report = report_review.export_html(args.output_dir / f'{prefix}_name_review.html', fusion=result)
    result.export_html(args.output_dir / f'{prefix}_exact_review.html', record_id_column=record_id_column)
    if not args.input:
        # Explicit synthetic reviewer decision, NOT automatic acceptance of proposals.
        # E's missing name is assigned to A/B using assumed external review evidence.
        decisions = pl.DataFrame({
            'record_id': list('ABCDE'),
            'refined_cluster_id': ['FI-NORTH', 'FI-NORTH', 'FI-DESERT', 'FI-DESERT', 'FI-NORTH'],
            'reason': ['Synthetic reviewer confirmed two institutions; E belongs to NORTH'] * 5,
        })
        decisions.write_parquet(args.output_dir / 'synthetic_accepted_decisions.parquet')
        split = apply_cluster_splits(result, decisions, record_id_column=record_id_column)
        split.records.write_parquet(args.output_dir / 'synthetic_refined_records.parquet')
        split.crossing_evidence.write_parquet(args.output_dir / 'synthetic_crossing_evidence.parquet')
        print(split.records)
    summary = {'input_rows': df.height, 'exact_clustering_and_load_seconds': round(exact_seconds, 3),
               'name_evaluation_seconds': round(review_seconds, 3),
               'requires_review_count': review.clusters['requires_review'].sum(),
               'evaluated_records': review.records.height,
               'clusters_with_multiple_name_components': review.clusters.filter(pl.col('name_component_count') > 1).height,
               'record_name_status_counts': review.records.group_by('name_status').len().to_dicts(),
               'name_pairs': review.name_pairs.height, 'configuration': review.configuration,
               'manual_split_applied': not bool(args.input), 'report_path': str(report.resolve()),
               'record_id_note': 'Generated row IDs are demonstration-only, not stable across extractions'
               if args.input and not args.record_id_column else 'Source record IDs retained'}
    (args.output_dir / f'{prefix}_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
