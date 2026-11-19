import tempfile
from pathlib import Path
import unittest

import polars as pl

from transitive_fusion import cluster_entities
from transitive_review import evaluate_clusters, apply_cluster_splits


class NameReviewTests(unittest.TestCase):
    def setUp(self):
        self.df = pl.DataFrame({
            'id': ['A', 'B', 'C', 'D', 'E', 'F'],
            'shared': ['x', 'x', 'x', 'x', 'x', 'other'],
            'name': ['NORTH HARBOR CREDIT UNION', 'north harbor credit union',
                     'DESERT VALLEY FINANCE', 'DESERT VALLEY FINANCE', None,
                     'NORTH HARBOR CREDIT UNION'],
        })
        self.result = cluster_entities(self.df, ['shared'], return_details=True)

    def test_diagnostics_proposals_and_original_memberships(self):
        review = evaluate_clusters(self.result, name_column='name', record_id_column='id', threshold=1)
        self.assertEqual(self.result.records['cluster_id'].to_list(), [0, 0, 0, 0, 0, 5])
        cluster = review.clusters.row(0, named=True)
        self.assertEqual(cluster['usable_name_count'], 4)
        self.assertEqual(cluster['name_component_count'], 2)
        self.assertEqual(cluster['proposed_component_count'], 3)
        self.assertEqual(cluster['supported_name_fraction'], 1.0)
        self.assertTrue(cluster['requires_review'])
        rows = {r['record_id']: r for r in review.records.to_dicts()}
        self.assertEqual(rows['A']['proposed_cluster_id'], rows['B']['proposed_cluster_id'])
        self.assertNotEqual(rows['A']['proposed_cluster_id'], rows['C']['proposed_cluster_id'])
        self.assertEqual(rows['E']['name_status'], 'missing_name')
        self.assertIsNone(rows['E']['best_name_similarity'])
        self.assertNotIn('F', rows)  # singleton omitted by default
        self.assertNotIn('F', review.name_pairs['record_id_a'].to_list())

    def test_complete_scores_against_independent_cosine(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        df = pl.DataFrame({'id': list('ABCD'), 'shared': ['x'] * 4,
                           'name': ['FIRST NATIONAL BANK', 'FIRST NATL BANK', 'NATIONAL TRUST', 'DESERT CREDIT']})
        result = cluster_entities(df, ['shared'], return_details=True)
        review = evaluate_clusters(result, name_column='name', record_id_column='id', threshold=.7)
        vectors = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4), lowercase=False).fit_transform(
            df['name'].str.to_lowercase().to_list())
        scores = cosine_similarity(vectors)
        ids = df['id'].to_list()
        for row in review.records.to_dicts():
            i = ids.index(row['record_id'])
            expected = max(scores[i, j] for j in range(4) if i != j)
            self.assertAlmostEqual(row['best_name_similarity'], expected)
            self.assertEqual(row['has_neighbor_above_threshold'], expected >= .7)

    def test_sparse_graph_and_missing_names(self):
        review = evaluate_clusters(self.result, name_column='name', record_id_column='id',
                                   top_n=1, complete_limit=2)
        self.assertFalse(review.clusters['graph_complete'][0])
        self.assertEqual(review.clusters['name_component_count'][0], 2)
        df = pl.DataFrame({'id': [1, 2], 'key': ['x', 'x'], 'name': [None, ' ']}, schema_overrides={'name': pl.String})
        review = evaluate_clusters(cluster_entities(df, ['key'], return_details=True),
                                   name_column='name', record_id_column='id')
        self.assertEqual(review.clusters['name_component_count'][0], 0)
        self.assertEqual(review.name_pairs.height, 0)
        self.assertEqual(review.records['best_name_similarity'].null_count(), 2)

    def test_singletons_and_zero_overlap(self):
        df = pl.DataFrame({'id': [1, 2, 3], 'key': ['x', 'x', 'y'], 'name': ['aaa', 'zzz', 'aaa']})
        review = evaluate_clusters(cluster_entities(df, ['key'], return_details=True),
                                   name_column='name', record_id_column='id', min_cluster_size=1)
        rows = {r['record_id']: r for r in review.records.to_dicts()}
        self.assertEqual(rows[1]['best_name_similarity'], 0)
        self.assertEqual(rows[1]['best_name_neighbor_id'], 2)
        self.assertEqual(rows[3]['name_status'], 'singleton')
        self.assertIsNone(rows[3]['best_name_similarity'])

    def test_small_clusters_skip_topn_and_keep_singletons_in_tfidf(self):
        from unittest.mock import patch
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        df = pl.DataFrame({'id': list('ABCDE'), 'key': ['x', 'x', 'x', 'y', 'z'],
                           'name': ['FIRST NATIONAL BANK', 'FIRST NATL BANK', None,
                                    'FIRST NATIONAL TRUST', ' ']})
        fusion = cluster_entities(df, ['key'], return_details=True)
        with patch('sparse_dot_topn.sp_matmul_topn', side_effect=AssertionError('unnecessary top-N call')):
            review = evaluate_clusters(fusion, name_column='name', record_id_column='id',
                                       min_cluster_size=1)
        vectors = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4), lowercase=False).fit_transform(
            ['first national bank', 'first natl bank', 'first national trust'])
        expected = cosine_similarity(vectors)[0, 1]
        self.assertAlmostEqual(review.name_pairs['name_similarity'][0], expected, places=13)
        rows = {r['record_id']: r for r in review.records.to_dicts()}
        self.assertEqual(rows['D']['name_status'], 'singleton')
        self.assertEqual(rows['E']['name_status'], 'missing_name')
        self.assertIsNone(rows['C']['best_name_similarity'])
        self.assertEqual(review.records.height, 5)

    def test_two_name_batches_preserve_all_pairs(self):
        from unittest.mock import patch
        count = 4100  # Cross the bounded sparse-dot batch boundary.
        df = pl.DataFrame({'id': list(range(count * 2)),
                           'key': [i // 2 for i in range(count * 2)],
                           'name': ['EXAMPLE CREDIT UNION'] * (count * 2)})
        fusion = cluster_entities(df, ['key'], return_details=True)
        with patch('sparse_dot_topn.sp_matmul_topn', side_effect=AssertionError('unnecessary top-N call')):
            review = evaluate_clusters(fusion, name_column='name', record_id_column='id', threshold=1)
        self.assertEqual(review.name_pairs.height, count)
        self.assertTrue(review.name_pairs['meets_threshold'].all())
        self.assertTrue(review.records['has_neighbor_above_threshold'].all())
        self.assertEqual(review.clusters['name_component_count'].unique().to_list(), [1])

    def test_invalid_configuration_and_record_ids(self):
        for options in ({'threshold': 0}, {'threshold': float('nan')}, {'top_n': True},
                        {'complete_limit': 0}, {'max_clusters': 0}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                evaluate_clusters(self.result, name_column='name', record_id_column='id', **options)
        duplicate = self.df.with_columns(pl.lit('duplicate').alias('id'))
        result = cluster_entities(duplicate, ['shared'], return_details=True)
        with self.assertRaisesRegex(ValueError, 'unique and nonmissing'):
            evaluate_clusters(result, name_column='name', record_id_column='id')

    def test_dashboard_evidence_and_interactions(self):
        import json
        import re
        import shutil
        import subprocess
        df = self.df.with_columns(pl.lit('Sample city').alias('source_city'),
                                  pl.lit(float('nan')).alias('optional_number'))
        fusion = cluster_entities(df, ['shared'], return_details=True)
        review = evaluate_clusters(fusion, name_column='name', record_id_column='id', min_cluster_size=1)
        with tempfile.TemporaryDirectory() as tmp:
            path = review.export_html(Path(tmp) / 'dashboard.html', fusion=fusion)
            html = path.read_text()
            payload = json.loads(re.search(r'id="review-data" type="application/json">(.*?)</script>', html, re.S)[1])
            self.assertTrue(payload['has_exact_evidence'])
            self.assertEqual(payload['groups'][0]['members'], list('ABCDE'))
            self.assertEqual(payload['groups'][0]['shared_values'], {'shared': 'x'})
            self.assertEqual(payload['source_rows']['A']['source_city'], 'Sample city')
            self.assertIsNone(payload['source_rows']['A']['optional_number'])
            node = shutil.which('node')
            if node:
                subprocess.run([node, str(Path(__file__).with_name('check_review_dashboard.js')), str(path)],
                               check=True, capture_output=True, text=True)

    def test_dashboard_rejects_mismatched_fusion(self):
        review = evaluate_clusters(self.result, name_column='name', record_id_column='id')
        other = cluster_entities(self.df, ['id'], return_details=True)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, 'memberships differ'):
                review.export_html(Path(tmp) / 'bad.html', fusion=other)

    def test_empty_selection_and_html_escape(self):
        review = evaluate_clusters(self.result, name_column='name', record_id_column='id', min_cluster_size=99)
        self.assertEqual(review.records.height, 0)
        self.assertEqual(review.clusters.height, 0)
        bad = pl.DataFrame({'id': ['<script>'], 'key': ['a'], 'name': ['<script>alert(1)</script>']})
        review = evaluate_clusters(cluster_entities(bad, ['key'], return_details=True),
                                   name_column='name', record_id_column='id', min_cluster_size=1)
        with tempfile.TemporaryDirectory() as tmp:
            html = review.export_html(Path(tmp) / 'review.html').read_text()
            self.assertNotIn('<script>alert(1)</script>', html)
            self.assertIn(r'\u003cscript\u003e', html)


class ManualSplitTests(unittest.TestCase):
    def setUp(self):
        self.df = pl.DataFrame({'id': list('ABCDE'), 'key': ['x', 'x', 'x', 'x', 'y']})
        self.result = cluster_entities(self.df, ['key'], return_details=True)
        self.decisions = pl.DataFrame({'record_id': list('ABCD'),
                                       'refined_cluster_id': ['FI-A', 'FI-A', 'FI-B', 'FI-B'],
                                       'reason': ['reviewed name difference'] * 4})

    def test_split_preserves_original_and_exposes_crossing_evidence(self):
        split = apply_cluster_splits(self.result, self.decisions, record_id_column='id')
        self.assertTrue(split.records.drop('refined_cluster_id').equals(self.result.records))
        self.assertEqual(split.records['refined_cluster_id'].to_list(), ['FI-A', 'FI-A', 'FI-B', 'FI-B', 'original:4'])
        self.assertEqual(split.crossing_evidence.height, 1)
        self.assertEqual(split.crossing_evidence['refined_cluster_ids'].to_list(), [['FI-A', 'FI-B']])
        # Evidence explicitly shows that this decision overrides the shared identifier.
        self.assertIn('key', split.crossing_evidence['conditions'][0])

    def test_reapply_after_reordering_by_stable_id(self):
        reordered = cluster_entities(self.df.reverse(), ['key'], return_details=True)
        split = apply_cluster_splits(reordered, self.decisions, record_id_column='id')
        mapping = dict(zip(split.records['id'], split.records['refined_cluster_id']))
        self.assertEqual(mapping['A'], 'FI-A')
        self.assertEqual(mapping['C'], 'FI-B')

    def test_incomplete_or_stale_decisions_fail(self):
        with self.assertRaisesRegex(ValueError, 'every current record'):
            apply_cluster_splits(self.result, self.decisions.head(3), record_id_column='id')
        expanded = pl.concat([self.df, pl.DataFrame({'id': ['F'], 'key': ['x']})])
        result = cluster_entities(expanded, ['key'], return_details=True)
        with self.assertRaisesRegex(ValueError, 'every current record'):
            apply_cluster_splits(result, self.decisions, record_id_column='id')
        unknown = self.decisions.with_columns(pl.lit('unknown').alias('record_id')).head(1)
        with self.assertRaisesRegex(ValueError, 'unknown records'):
            apply_cluster_splits(self.result, unknown, record_id_column='id')

    def test_persisted_decisions_and_custom_cluster_column(self):
        result = cluster_entities(self.df, ['key'], cluster_column='entity', return_details=True)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'decisions.parquet'
            self.decisions.write_parquet(path)
            split = apply_cluster_splits(result, pl.read_parquet(path), record_id_column='id',
                                         refined_column='reviewed_entity')
        self.assertIn('entity', split.records.columns)
        self.assertIn('reviewed_entity', split.records.columns)
        self.assertEqual(split.crossing_evidence['entity'][0], 0)

    def test_cross_cluster_merges_and_noop_rejected(self):
        extra = pl.DataFrame({'record_id': ['E'], 'refined_cluster_id': ['FI-A'], 'reason': ['reviewed']})
        # Include another member in E's cluster so it can be split as well.
        df = pl.concat([self.df, pl.DataFrame({'id': ['F'], 'key': ['y']})])
        result = cluster_entities(df, ['key'], return_details=True)
        decisions = pl.concat([self.decisions, extra,
                               pl.DataFrame({'record_id': ['F'], 'refined_cluster_id': ['FI-C'], 'reason': ['reviewed']})])
        with self.assertRaisesRegex(ValueError, 'different original clusters'):
            apply_cluster_splits(result, decisions, record_id_column='id')
        with self.assertRaisesRegex(ValueError, 'at least two'):
            apply_cluster_splits(self.result, self.decisions.with_columns(pl.lit('same').alias('refined_cluster_id')),
                                 record_id_column='id')


if __name__ == '__main__':
    unittest.main()
