"""Benchmark local PPP data without importing the larger entity linker.

Run each case in a separate process for comparable process peak memory:
  python examples/benchmark_transitive_fusion.py --mode blocked
Outputs a JSON summary and a self-contained review report. No input is modified.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import resource
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import polars as pl
from transitive_fusion import cluster_entities

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / 'test_data/100k.parquet')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'test_data/_debug_outputs/transitive_benchmark')
    parser.add_argument('--mode', choices=['unblocked', 'blocked', 'strict'], default='blocked')
    parser.add_argument('--limit', type=int, help='Optional first N records; default is the entire file')
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')
    started = perf_counter()
    scan = pl.scan_parquet(args.input) if args.input.suffix == '.parquet' else pl.scan_csv(args.input)
    scan = scan.select('BorrowerName', 'BorrowerAddress', 'BorrowerCity', 'BorrowerState')
    if args.limit is not None:
        scan = scan.head(args.limit)
    df = scan.collect()
    load_seconds = perf_counter() - started
    # Exact raw values, with no extra normalization, for reproducibility.
    leaves = [{'field': field, **({} if args.mode == 'unblocked' else
               {'block_on': ['BorrowerCity', 'BorrowerState']})}
              for field in ['BorrowerName', 'BorrowerAddress']]
    rule = {'and' if args.mode == 'strict' else 'or': leaves}
    started = perf_counter()
    result = cluster_entities(df, match_rule=rule, return_details=True)
    clustering_seconds = perf_counter() - started
    started = perf_counter()
    # Full-data invariants: every input row is retained, labels equal the earliest
    # member position, and every evidence group stays within its assigned cluster.
    assert result.records.drop('cluster_id').equals(df)
    sizes = (result.records.with_row_index('_position').group_by('cluster_id')
             .agg(pl.len().alias('size'), pl.col('_position').min().alias('earliest'))
             .sort(['size', 'cluster_id'], descending=[True, False]))
    assert sizes.filter(pl.col('cluster_id') != pl.col('earliest')).height == 0
    labels = result.records['cluster_id'].to_list()
    for group in result.match_groups.iter_rows(named=True):
        assert all(labels[p] == group['cluster_id'] for p in group['row_positions'])
    validation_seconds = perf_counter() - started
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f'{args.mode}_{df.height}'
    started = perf_counter()
    report = result.export_html(args.output_dir / f'{stem}.html', top_n=10,
                                conflict_columns=['BorrowerName'], common_value_min_count=5)
    report_seconds = perf_counter() - started
    # ru_maxrss is bytes on macOS, KiB on Linux. Includes load, validation, export.
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_mib = rss / (1024 ** 2 if sys.platform == 'darwin' else 1024)
    summary = {
        'input': str(args.input.resolve()), 'mode': args.mode, 'rows': df.height,
        'polars_version': pl.__version__, 'rule': rule, 'normalization': 'none; exact raw values',
        'load_seconds': round(load_seconds, 3), 'clustering_seconds': round(clustering_seconds, 3),
        'validation_seconds': round(validation_seconds, 3), 'report_seconds': round(report_seconds, 3),
        'process_peak_mib': round(peak_mib, 1),
        'clusters': sizes.height,
        'singleton_clusters': sizes.filter(pl.col('size') == 1).height,
        'records_in_non_singleton_clusters': sizes.filter(pl.col('size') > 1)['size'].sum(),
        'largest_cluster': sizes['size'].max(), 'match_groups': result.match_groups.height,
        'top_10_clusters': sizes.select('cluster_id', 'size').head(10).to_dicts(),
        'checks': 'input preserved; earliest-member labels; all evidence memberships consistent',
        'report_path': str(report.resolve()), 'report_mib': round(report.stat().st_size / 1024 ** 2, 3),
    }
    (args.output_dir / f'{stem}.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
