"""Two-table dominant-value evaluation; synthetic data or a local PPP Parquet sample.

python examples/dominant_review.py
python examples/dominant_review.py --input test_data/100k.parquet --limit 100000
"""
from pathlib import Path
from time import perf_counter
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import polars as pl
from transitive_fusion import cluster_entities
from transitive_review import evaluate_clusters


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path)
    parser.add_argument('--limit', type=int, default=10000, help='0 = full file')
    parser.add_argument('--output-dir', type=Path, default=Path('test_data/_debug_outputs/dominant_review'))
    args = parser.parse_args()
    if args.limit < 0:
        parser.error('--limit must be nonnegative')
    if args.input:
        scan = pl.scan_parquet(args.input)
        if args.limit:
            scan = scan.head(args.limit)
        records = scan.collect().with_row_index('record_id')
        fusion = cluster_entities(records, match_rule={'or': [
            {'field': c, 'block_on': ['BorrowerCity', 'BorrowerState']}
            for c in ['BorrowerName', 'BorrowerAddress']]}, return_details=True)
        fields = {'BorrowerName': {'method': 'tfidf', 'threshold': .85},
                  'BorrowerAddress': {'method': 'levenshtein', 'threshold': .85}}
        filing_options = {}
        prefix = f'ppp_{records.height}'
    else:
        fusion = pl.DataFrame({
            'record_id': list('ABCDE'), 'cluster_id': [1, 1, 1, 2, 2],
            'name': ['NORTH HARBOR BANK', 'North Harbor Bank', 'DESERT FINANCE', 'SOLO BANK', None],
            'address': ['100 Main St', '100 MAIN ST', '100 Main Street', '20 Elm St', '20 Elm St'],
            'ein': ['SYNTH-1', 'SYNTH-1', 'SYNTH-2', 'SYNTH-3', 'SYNTH-3'],
            'filing_count': [4, 2, 3, 1, None],
            'filing_id': ['F1', 'F1', 'F2', 'F3', None],
        })
        fields = {'name': {'method': 'tfidf', 'threshold': .85},
                  'address': {'method': 'levenshtein', 'threshold': .8}, 'ein': {'method': 'exact'}}
        filing_options = {'filing_count_column': 'filing_count', 'filing_id_column': 'filing_id'}
        prefix = 'synthetic'
    started = perf_counter()
    evaluation = evaluate_clusters(fusion, record_id_column='record_id', fields=fields, **filing_options)
    elapsed = perf_counter() - started
    record_review, cluster_review = evaluation
    args.output_dir.mkdir(parents=True, exist_ok=True)
    record_review.write_parquet(args.output_dir / f'{prefix}_records.parquet')
    cluster_review.write_parquet(args.output_dir / f'{prefix}_clusters.parquet')
    metadata = {**evaluation.configuration, 'seconds': round(elapsed, 3),
                'record_rows': record_review.height, 'cluster_rows': cluster_review.height,
                'flagged_records': record_review['requires_review'].sum(),
                'flagged_clusters': cluster_review['requires_review'].sum(),
                'id_note': 'PPP row IDs are extraction-specific; use stable source IDs for durable review'}
    (args.output_dir / f'{prefix}_summary.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(json.dumps(metadata, indent=2))
    display = ['record_id', 'cluster_id', 'requires_review', 'review_reasons']
    print(record_review.select(display).head(10))
    print(cluster_review.select('cluster_id', 'record_count', 'total_filings', 'summed_filing_count', 'distinct_filing_ids', 'review_record_count', 'requires_review').head(10))


if __name__ == '__main__':
    main()
