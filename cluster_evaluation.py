"""Per-record diagnostics and per-cluster evaluation against dominant values."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import polars as pl

from transitive_fusion import FusionResult


@dataclass
class ClusterEvaluation:
    """Two joinable tables; unpack as record_review, cluster_review = evaluation."""
    records: pl.DataFrame
    clusters: pl.DataFrame
    configuration: dict

    def __iter__(self):
        yield self.records
        yield self.clusters


def _edit_distance(a, b):
    """Fallback when optional rapidfuzz is unavailable."""
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def _parse_fields(df, fields):
    if not isinstance(fields, Mapping) or not fields:
        raise ValueError('fields must be a nonempty mapping of column names to evaluation options')
    parsed = {}
    for column, options in fields.items():
        if not isinstance(column, str) or column not in df.columns:
            raise ValueError(f'unknown evaluation field: {column!r}')
        if not isinstance(options, Mapping) or set(options) - {'method', 'threshold', 'normalize', 'missing', 'review_on_tie'}:
            raise ValueError(f'{column}: invalid evaluation options')
        method = options.get('method', 'exact')
        threshold = options.get('threshold', 1.0 if method == 'exact' else .85)
        normalize = options.get('normalize', True)
        missing = options.get('missing', 'review')
        on_tie = options.get('review_on_tie', True)
        if method not in ('exact', 'tfidf', 'levenshtein'):
            raise ValueError(f'{column}: method must be exact, tfidf, or levenshtein')
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1:
            raise ValueError(f'{column}: threshold must be between 0 and 1')
        if type(normalize) is not bool or type(on_tie) is not bool or missing not in ('review', 'ignore'):
            raise ValueError(f'{column}: normalize/review_on_tie must be booleans; missing must be review or ignore')
        dtype = df.schema[column]
        is_text = dtype in (pl.String, pl.Categorical) or isinstance(dtype, pl.Enum)
        if dtype.is_nested() or dtype == pl.Object or (method != 'exact' and not is_text and dtype != pl.Null):
            raise TypeError(f'{column}: {method} does not support identifier type {dtype}')
        parsed[column] = dict(method=method, threshold=float(threshold), normalize=normalize,
                              missing=missing, review_on_tie=on_tie)
    return parsed


def _scores(pairs, method):
    """Compare distinct value/reference pairs; never construct all record pairs."""
    if not pairs.height:
        return pairs.with_columns(pl.lit(None, dtype=pl.Float64).alias('_score'),
                                  pl.lit(None, dtype=pl.UInt64).alias('_distance'))
    if method == 'exact':
        return pairs.with_columns((pl.col('_value') == pl.col('_dominant')).cast(pl.Float64).alias('_score'),
                                  pl.lit(None, dtype=pl.UInt64).alias('_distance'))
    if method == 'levenshtein':
        try:
            from rapidfuzz.distance.Levenshtein import distance
        except ImportError:
            distance = _edit_distance
        distances = [distance(a, b) for a, b in pairs.iter_rows()]
        similarities = [1.0 - d / max(len(a), len(b), 1)
                        for (a, b), d in zip(pairs.iter_rows(), distances)]
        return pairs.with_columns(pl.Series('_score', similarities, dtype=pl.Float64),
                                  pl.Series('_distance', distances, dtype=pl.UInt64))
    from sklearn.feature_extraction.text import TfidfVectorizer
    # One document per distinct normalized value, not per repeated filing.
    values = sorted(set(pairs['_value']) | set(pairs['_dominant']))
    index = {value: i for i, value in enumerate(values)}
    matrix = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4), lowercase=False).fit_transform(values)
    scores = []
    for start in range(0, pairs.height, 4096):
        batch = pairs.slice(start, 4096)
        left = matrix[[index[v] for v in batch['_value']]]
        right = matrix[[index[v] for v in batch['_dominant']]]
        scores.extend(float(min(1.0, max(0.0, s))) for s in left.multiply(right).sum(axis=1).A1)
    return pairs.with_columns(pl.Series('_score', scores, dtype=pl.Float64),
                              pl.lit(None, dtype=pl.UInt64).alias('_distance'))


def evaluate_dominant_values(
    result: FusionResult | pl.DataFrame, *, record_id_column: str, fields: Mapping,
    cluster_id_column: str | None = None, filing_count_column: str | None = None,
    filing_id_column: str | None = None, min_cluster_size: int = 1,
    max_clusters: int | None = None,
) -> ClusterEvaluation:
    """Evaluate each requested field against its cluster's most frequent value.

    Returns ALL input record IDs in input order and ALL clusters, including when
    scope limits are supplied. Out-of-scope rows have evaluated=False and null
    requires_review. Missing normalized values are excluded from dominance.
    Ties choose the lowest sorted value and are flagged by default. Dominance
    counts records, not filings; filing totals are independent summary metrics.
    """
    df = result.records if isinstance(result, FusionResult) else result
    if not isinstance(df, pl.DataFrame):
        raise TypeError('result must be a FusionResult or Polars DataFrame')
    cluster_col = cluster_id_column or (result.cluster_column if isinstance(result, FusionResult) else 'cluster_id')
    if record_id_column not in df.columns or cluster_col not in df.columns:
        raise ValueError('record ID and cluster ID columns must exist')
    for column in (record_id_column, cluster_col):
        dtype = df.schema[column]
        if dtype.is_nested() or dtype == pl.Object:
            raise TypeError(f'{column}: ID columns must be scalar')
        invalid = pl.col(column).is_null()
        if dtype.is_float():
            invalid = invalid | pl.col(column).is_nan()
        if dtype in (pl.String, pl.Categorical) or isinstance(dtype, pl.Enum):
            invalid = invalid | (pl.col(column).cast(pl.String).str.strip_chars() == '')
        if df.select(invalid.any()).item():
            raise ValueError(f'{column}: IDs must be nonmissing')
    if df[record_id_column].n_unique() != df.height:
        raise ValueError('record IDs must be unique; aggregate filings by record or use a unique source-row ID')
    if type(min_cluster_size) is not int or min_cluster_size < 1:
        raise ValueError('min_cluster_size must be positive')
    if max_clusters is not None and (type(max_clusters) is not int or max_clusters < 1):
        raise ValueError('max_clusters must be positive or None')
    for column in (filing_count_column, filing_id_column):
        if column is not None and column not in df.columns:
            raise ValueError(f'missing filing column: {column}')
    options = _parse_fields(df, fields)
    records = df.select(pl.col(record_id_column).alias('record_id'), pl.col(cluster_col).alias('cluster_id'))
    clusters = records.group_by('cluster_id').len(name='record_count').sort(
        ['record_count', 'cluster_id'], descending=[True, False])
    selected = clusters.filter(pl.col('record_count') >= min_cluster_size).select('cluster_id')
    if max_clusters is not None:
        selected = selected.head(max_clusters)
    clusters = clusters.join(selected.with_columns(pl.lit(True).alias('evaluated')), on='cluster_id', how='left',
                             maintain_order='left').with_columns(pl.col('evaluated').fill_null(False))
    records = records.join(clusters.select('cluster_id', 'evaluated'), on='cluster_id', how='left', maintain_order='left')
    totals = clusters.select('cluster_id')
    if filing_count_column is not None:
        values = df[filing_count_column]
        if values.dtype != pl.Null and (not values.dtype.is_numeric() or values.dtype == pl.Boolean):
            raise TypeError('filing counts must be numeric nonnegative integers')
        check = df.select(pl.col(filing_count_column).cast(pl.Float64).alias('_n'))
        if check.filter(pl.col('_n').is_not_null() & (~pl.col('_n').is_finite() | (pl.col('_n') < 0) |
                                                    (pl.col('_n') != pl.col('_n').floor()))).height:
            raise ValueError('filing counts must be finite nonnegative integers or null')
        filings = df.select(pl.col(cluster_col).alias('cluster_id'), pl.col(filing_count_column).cast(pl.UInt64).alias('_filings'))
        counts = filings.group_by('cluster_id').agg(
            pl.when(pl.col('_filings').count() > 0).then(pl.col('_filings').sum()).otherwise(None).alias('summed_filing_count'),
            pl.col('_filings').null_count().alias('filing_counts_missing_count'))
        totals = totals.join(counts, on='cluster_id', how='left', maintain_order='left')
    else:
        totals = totals.with_columns(pl.lit(None, dtype=pl.UInt64).alias('summed_filing_count'),
                                     pl.lit(None, dtype=pl.UInt32).alias('filing_counts_missing_count'))
    if filing_id_column is not None:
        dtype = df.schema[filing_id_column]
        if dtype.is_nested() or dtype == pl.Object:
            raise TypeError('filing IDs must be scalar')
        expr = pl.col(filing_id_column)
        if dtype.is_float():
            expr = pl.when(expr.is_nan()).then(None).otherwise(expr)
        if dtype in (pl.String, pl.Categorical) or isinstance(dtype, pl.Enum):
            expr = pl.when(expr.cast(pl.String).str.strip_chars() == '').then(None).otherwise(expr)
        filings = df.select(pl.col(cluster_col).alias('cluster_id'), expr.alias('_filings'))
        distinct = filings.group_by('cluster_id').agg(
            pl.when(pl.col('_filings').count() > 0).then(pl.col('_filings').drop_nulls().n_unique().cast(pl.UInt64))
            .otherwise(None).alias('distinct_filing_ids'), pl.col('_filings').null_count().alias('filing_ids_missing_count'))
        totals = totals.join(distinct, on='cluster_id', how='left', maintain_order='left')
    else:
        totals = totals.with_columns(pl.lit(None, dtype=pl.UInt64).alias('distinct_filing_ids'),
                                     pl.lit(None, dtype=pl.UInt32).alias('filing_ids_missing_count'))
    # Backward-compatible convenience columns: count totals take precedence,
    # without falling back row-by-row or adding incompatible measures together.
    count_source = filing_count_column is not None
    totals = totals.with_columns(
        pl.col('summed_filing_count' if count_source else 'distinct_filing_ids').alias('total_filings'),
        pl.col('filing_counts_missing_count' if count_source else 'filing_ids_missing_count').alias('filings_missing_count'))
    clusters = clusters.join(totals, on='cluster_id', how='left', maintain_order='left')
    reason_columns = []
    for number, (column, spec) in enumerate(options.items()):
        dtype = df.schema[column]
        is_text = dtype in (pl.String, pl.Categorical) or isinstance(dtype, pl.Enum)
        value = pl.col(column).cast(pl.String) if is_text or dtype == pl.Null else pl.col(column)
        if is_text:
            value = pl.when(value.str.strip_chars() == '').then(None).otherwise(value)
            if spec['normalize']:
                # Unicode lowercase plus collapsed whitespace, without domain-specific cleaning.
                value = value.str.to_lowercase().str.replace_all(r'\s+', ' ').str.strip_chars()
        elif dtype.is_float():
            value = pl.when(value.is_nan()).then(None).otherwise(value)
        work = records.select('record_id', 'cluster_id', 'evaluated').with_columns(df[column].alias('_raw'))
        work = work.with_columns(df.select(value.alias('_value'))['_value'])
        frequencies = work.filter(pl.col('_value').is_not_null()).group_by('cluster_id', '_value').len(name='_count')
        refs = frequencies.sort(['cluster_id', '_count', '_value'], descending=[False, True, False]).group_by('cluster_id').agg(
            pl.col('_value').first().alias('_dominant'), pl.col('_count').first().alias('_dominant_count'),
            pl.col('_count').sum().alias('_valid_count'), pl.len().alias('_distinct'),
            (pl.col('_count') == pl.col('_count').max()).sum().alias('_tie_count'))
        work = work.join(refs, on='cluster_id', how='left', maintain_order='left').with_columns(
            (pl.col('_tie_count').fill_null(0) > 1).alias('_tied'))
        pairs = work.filter(pl.col('evaluated') & pl.col('_value').is_not_null() & pl.col('_dominant').is_not_null()).select('_value', '_dominant').unique()
        scored = _scores(pairs, spec['method'])
        work = work.join(scored, on=['_value', '_dominant'], how='left', maintain_order='left').with_columns(
            pl.when(pl.col('evaluated')).then(pl.col('_score')).otherwise(None).alias('_score'),
            pl.when(pl.col('evaluated')).then(pl.col('_distance')).otherwise(None).alias('_distance'))
        missing = pl.col('_value').is_null()
        below = pl.col('_score') < spec['threshold'] - 1e-12
        tied = pl.col('_tied') & pl.lit(spec['review_on_tie'])
        issue = ((missing & pl.lit(spec['missing'] == 'review')) | below.fill_null(False) | tied)
        status = (pl.when(~pl.col('evaluated')).then(pl.lit('not_evaluated'))
                  .when(missing).then(pl.lit('missing'))
                  .when(below).then(pl.lit('below_threshold'))
                  .when(tied).then(pl.lit('dominant_tie')).otherwise(pl.lit('ok')))
        work = work.with_columns(status.alias('_status'),
                                 pl.when(pl.col('evaluated')).then(issue).otherwise(None).alias('_review'))
        prefix = column + '__'
        annotations = work.select(
            pl.col('_raw').alias(prefix + 'value'), pl.col('_dominant').alias(prefix + 'dominant_value'),
            pl.col('_score').alias(prefix + 'similarity'), pl.col('_status').alias(prefix + 'status'),
            pl.col('_review').alias(prefix + 'requires_review'))
        if spec['method'] == 'levenshtein':
            annotations = annotations.with_columns(work['_distance'].alias(prefix + 'distance'))
        records = records.hstack(annotations)
        reasons = [pl.when(pl.col('evaluated') & expr).then(pl.lit(column + ':' + label)).otherwise(None)
                   for expr, label in [(missing & pl.lit(spec['missing'] == 'review'), 'missing'),
                                       (below.fill_null(False), 'below_threshold'), (tied, 'dominant_tie')]]
        key = f'_reasons_{number}'
        records = records.with_columns(work.select(pl.concat_list(reasons).list.drop_nulls().alias(key))[key])
        reason_columns.append(key)
        summary = work.group_by('cluster_id').agg(
            pl.col('_dominant').first().alias(prefix + 'dominant_value'),
            pl.col('_dominant_count').first().fill_null(0).alias(prefix + 'dominant_count'),
            (pl.col('_dominant_count').first() / pl.col('_valid_count').first()).alias(prefix + 'dominant_fraction'),
            pl.col('_tied').first().alias(prefix + 'dominant_tied'),
            pl.col('_distinct').first().fill_null(0).alias(prefix + 'distinct_values'),
            pl.col('_value').null_count().alias(prefix + 'missing_count'),
            pl.col('_score').count().alias(prefix + 'compared_count'),
            (pl.col('_status') == 'below_threshold').sum().alias(prefix + 'below_threshold_count'),
            pl.col('_review').sum().alias(prefix + 'review_count'),
            pl.col('_score').mean().alias(prefix + 'mean_similarity'),
            pl.col('_score').min().alias(prefix + 'min_similarity')).with_columns(
                pl.lit(spec['method']).alias(prefix + 'method'), pl.lit(spec['threshold']).alias(prefix + 'threshold'))
        clusters = clusters.join(summary, on='cluster_id', how='left', maintain_order='left')
    records = records.with_columns(pl.concat_list(reason_columns).alias('review_reasons')).drop(reason_columns)
    records = records.with_columns(pl.when(pl.col('evaluated')).then(pl.col('review_reasons').list.len() > 0)
                                   .otherwise(None).alias('requires_review'))
    review_stats = records.group_by('cluster_id').agg(
        pl.col('requires_review').fill_null(False).sum().alias('review_record_count'),
        pl.col('review_reasons').flatten().drop_nulls().unique().sort().alias('review_reasons'))
    clusters = clusters.join(review_stats, on='cluster_id', how='left', maintain_order='left').with_columns(
        pl.when(pl.col('evaluated')).then(pl.col('review_record_count') > 0).otherwise(None).alias('requires_review'))
    record_core = ['record_id', 'cluster_id', 'requires_review', 'review_reasons', 'evaluated']
    cluster_core = ['cluster_id', 'record_count', 'total_filings', 'review_record_count',
                    'requires_review', 'review_reasons', 'evaluated', 'filings_missing_count']
    records = records.select(record_core + [c for c in records.columns if c not in record_core])
    clusters = clusters.select(cluster_core + [c for c in clusters.columns if c not in cluster_core])
    config = dict(mode='dominant_values', record_id_column=record_id_column, cluster_id_column=cluster_col,
                  fields=options, filing_count_column=filing_count_column, filing_id_column=filing_id_column,
                  min_cluster_size=min_cluster_size, max_clusters=max_clusters,
                  dominance='record-count mode of normalized nonmissing values; sorted-value tie break',
                  tfidf_corpus='distinct normalized values in evaluated comparisons', tfidf_ngram_range=[2, 4])
    return ClusterEvaluation(records, clusters, config)
