import unittest

import polars as pl
from polars.testing import assert_frame_equal

from transitive_fusion import cluster_entities
from transitive_review import evaluate_clusters
from cluster_evaluation import _edit_distance


class DominantEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.df = pl.DataFrame({
            'id': list('ABCDE'), 'entity': [1, 1, 1, 2, 3],
            'name': ['BANK ONE', ' bank   one ', 'OTHER BANK', 'SOLO', None],
            'address': ['Main', 'Main', 'Maine', 'Solo', None],
            'ein': [10, 10, 20, 30, None], 'filings': [3, 4, 5, 1, None],
            'filing_id': ['F1', 'F1', 'F2', 'F3', None],
        })

    def run_eval(self, **kwargs):
        return evaluate_clusters(self.df, record_id_column='id', cluster_id_column='entity',
                                 fields={'name': {'method': 'exact'}}, **kwargs)

    def test_two_tables_full_coverage_and_local_flags(self):
        result = self.run_eval(filing_count_column='filings')
        records, clusters = result
        self.assertEqual(records['record_id'].to_list(), list('ABCDE'))
        rows = {r['record_id']: r for r in records.to_dicts()}
        self.assertFalse(rows['A']['requires_review'])
        self.assertFalse(rows['B']['requires_review'])
        self.assertTrue(rows['C']['requires_review'])
        self.assertEqual(rows['C']['review_reasons'], ['name:below_threshold'])
        self.assertTrue(rows['E']['requires_review'])
        self.assertEqual(rows['E']['name__status'], 'missing')
        one = clusters.filter(pl.col('cluster_id') == 1).row(0, named=True)
        self.assertEqual(one['record_count'], 3)
        self.assertEqual(one['total_filings'], 12)
        self.assertEqual(one['review_record_count'], 1)
        self.assertEqual(one['name__dominant_value'], 'bank one')
        self.assertEqual(one['name__dominant_count'], 2)
        self.assertAlmostEqual(one['name__dominant_fraction'], 2/3)
        self.assertEqual(one['name__below_threshold_count'], 1)
        self.assertIsNone(clusters.filter(pl.col('cluster_id') == 3)['total_filings'][0])

    def test_scope_keeps_all_rows_and_marks_unassessed(self):
        result = self.run_eval(max_clusters=1, min_cluster_size=2)
        self.assertEqual(result.records.height, 5)
        self.assertEqual(result.clusters.height, 3)
        outside = result.records.filter(~pl.col('evaluated'))
        self.assertEqual(outside.height, 2)
        self.assertEqual(outside['requires_review'].null_count(), 2)
        self.assertEqual(outside['name__similarity'].null_count(), 2)
        self.assertEqual(set(outside['name__status']), {'not_evaluated'})
        self.assertEqual(result.clusters.filter(~pl.col('evaluated'))['name__compared_count'].sum(), 0)
        # Same value/reference pair inside and outside the selected scope must not leak scores.
        df = pl.DataFrame({'id': [1, 2, 3], 'cluster_id': [1, 1, 2], 'v': ['x', 'x', 'x']})
        review = evaluate_clusters(df, record_id_column='id', fields={'v': {}}, max_clusters=1)
        self.assertIsNone(review.records['v__similarity'][2])

    def test_ties_deterministic_and_optional_review(self):
        df = pl.DataFrame({'id': [1, 2], 'cluster_id': [1, 1], 'name': ['Z', 'A']})
        result = evaluate_clusters(df, record_id_column='id', fields={'name': {}})
        self.assertEqual(result.clusters['name__dominant_value'][0], 'a')
        self.assertTrue(result.clusters['name__dominant_tied'][0])
        self.assertTrue(result.records['requires_review'].all())
        reversed_result = evaluate_clusters(df.reverse(), record_id_column='id', fields={'name': {}})
        self.assertEqual(reversed_result.clusters['name__dominant_value'][0], 'a')
        unflagged = evaluate_clusters(df, record_id_column='id', fields={'name': {'review_on_tie': False}})
        self.assertFalse(unflagged.records.filter(pl.col('record_id') == 2)['requires_review'][0])

    def test_levenshtein_distance_and_tfidf_against_reference(self):
        result = evaluate_clusters(self.df, record_id_column='id', cluster_id_column='entity', fields={
            'address': {'method': 'levenshtein', 'threshold': .8}, 'name': {'method': 'tfidf', 'threshold': .85}})
        row = result.records.filter(pl.col('record_id') == 'C').row(0, named=True)
        self.assertEqual(row['address__distance'], 1)
        self.assertAlmostEqual(row['address__similarity'], .8)
        self.assertFalse(row['address__requires_review'])  # inclusive threshold
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        values = ['bank one', 'other bank', 'solo']
        matrix = TfidfVectorizer(analyzer='char_wb', ngram_range=(2,4), lowercase=False).fit_transform(values)
        self.assertAlmostEqual(row['name__similarity'], cosine_similarity(matrix)[1,0], places=12)
        self.assertEqual(_edit_distance('kitten', 'sitting'), 3)
        self.assertEqual(_edit_distance('', 'abc'), 3)

    def test_exact_numeric_and_distinct_filings(self):
        result = evaluate_clusters(self.df, record_id_column='id', cluster_id_column='entity',
                                   fields={'ein': {'method': 'exact'}}, filing_id_column='filing_id')
        one = result.clusters.filter(pl.col('cluster_id') == 1).row(0, named=True)
        self.assertEqual(one['total_filings'], 2)
        self.assertEqual(one['ein__dominant_value'], 10)
        self.assertEqual(result.records['ein__value'].dtype, pl.Int64)
        self.assertEqual(result.records['requires_review'].to_list(), [False, False, True, False, True])

    def test_both_filing_measures_are_separate(self):
        result = self.run_eval(filing_count_column='filings', filing_id_column='filing_id')
        row = result.clusters.filter(pl.col('cluster_id') == 1).row(0, named=True)
        self.assertEqual(row['summed_filing_count'], 12)
        self.assertEqual(row['distinct_filing_ids'], 2)
        self.assertEqual(row['total_filings'], 12)  # never sum the two measures
        self.assertEqual(row['filing_counts_missing_count'], 0)
        missing = result.clusters.filter(pl.col('cluster_id') == 3).row(0, named=True)
        self.assertIsNone(missing['summed_filing_count'])
        self.assertIsNone(missing['distinct_filing_ids'])
        self.assertEqual(missing['filing_ids_missing_count'], 1)
        # Literal source names can contain spaces.
        spaced = self.df.rename({'filings': 'filing total'})
        result = evaluate_clusters(spaced, record_id_column='id', cluster_id_column='entity',
                                   fields={'name': {}}, filing_count_column='filing total')
        self.assertEqual(result.clusters['summed_filing_count'][0], 12)

    def test_all_missing_empty_and_missing_ignore(self):
        df = pl.DataFrame({'id': [1, 2], 'cluster_id': [1, 1], 'v': [None, None]})
        result = evaluate_clusters(df, record_id_column='id', fields={'v': {'method': 'tfidf', 'missing': 'ignore'}})
        self.assertEqual(result.records['v__similarity'].null_count(), 2)
        self.assertFalse(result.records['requires_review'].any())
        self.assertEqual(result.clusters['v__dominant_count'][0], 0)
        empty = evaluate_clusters(df.head(0), record_id_column='id', fields={'v': {'method': 'tfidf'}})
        self.assertEqual(empty.records.height, 0)
        self.assertEqual(empty.clusters.height, 0)

    def test_validation_and_fusion_input(self):
        for options in [{'method': 'bad'}, {'threshold': float('nan')}, {'normalize': 'yes'}, {'missing': 'zero'}]:
            with self.subTest(options=options), self.assertRaises(ValueError):
                evaluate_clusters(self.df, record_id_column='id', cluster_id_column='entity', fields={'name': options})
        with self.assertRaises(ValueError):
            evaluate_clusters(self.df.with_columns(pl.lit('same').alias('id')), record_id_column='id',
                              cluster_id_column='entity', fields={'name': {}})
        for count in [-1.0, 1.5, float('inf'), float('nan')]:
            with self.subTest(count=count), self.assertRaises(ValueError):
                evaluate_clusters(self.df.with_columns(pl.lit(count).alias('filings')), record_id_column='id',
                                  cluster_id_column='entity', fields={'name': {}}, filing_count_column='filings')
        fused = cluster_entities(self.df, ['ein'], return_details=True)
        result = evaluate_clusters(fused, record_id_column='id', fields={'name': {}})
        self.assertEqual(result.records.height, self.df.height)


if __name__ == '__main__':
    unittest.main()
