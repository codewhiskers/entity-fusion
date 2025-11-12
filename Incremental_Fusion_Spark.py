# incremental_signal_linker_spark.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, Set
from datetime import datetime
import math
import json

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.ml.feature import CountVectorizer, IDF
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql import functions as F, types as T

# Optional GraphFrames (if installed). If not, code falls back to pure-Spark CC.
try:
    from graphframes import GraphFrame  # type: ignore

    GF_AVAILABLE = True
except Exception:
    GF_AVAILABLE = False


# ----------------------------- Helpers -----------------------------
def _path_exists(spark, path: str) -> bool:
    sc = spark.sparkContext
    jvm = sc._jvm
    hc = sc._jsc.hadoopConfiguration()
    p = jvm.org.apache.hadoop.fs.Path(path)
    fs = p.getFileSystem(hc)
    return fs.exists(p)


def _char_wb_ngrams(s: Optional[str], ngram_range: Tuple[int, int]) -> List[str]:
    if s is None:
        return []
    # "char_wb" style: pad with spaces; then sliding window
    padded = f" {s.strip()} "
    lo, hi = ngram_range
    out = []
    for n in range(lo, hi + 1):
        if n <= 0 or n > len(padded):
            continue
        out.extend([padded[i : i + n] for i in range(len(padded) - n + 1)])
    return out


def _char_ngrams(s: Optional[str], ngram_range: Tuple[int, int]) -> List[str]:
    if s is None:
        return []
    lo, hi = ngram_range
    out = []
    for n in range(lo, hi + 1):
        if n <= 0 or n > len(s):
            continue
        out.extend([s[i : i + n] for i in range(len(s) - n + 1)])
    return out


def _cosine(u, v) -> float:
    # u, v: Sparse/Dense vectors
    if u is None or v is None:
        return 0.0
    num = float(u.dot(v))
    du = float(u.norm(2))
    dv = float(v.norm(2))
    if du == 0.0 or dv == 0.0:
        return 0.0
    return num / (du * dv)


# UDFs for Spark
udf_charwb = F.udf(
    lambda s, lo, hi: _char_wb_ngrams(s, (int(lo), int(hi))),
    T.ArrayType(T.StringType()),
)
udf_char = F.udf(
    lambda s, lo, hi: _char_ngrams(s, (int(lo), int(hi))), T.ArrayType(T.StringType())
)
udf_cosine = F.udf(_cosine, T.DoubleType())


@dataclass
class IncrementalSignalLinkerSpark:
    """
    PySpark ER core:
      - Alias building from a blocking plan (exact + cosine on character n-grams)
      - Within-batch & cross-batch pairing
      - K-of-N enforcement
      - IDF-weighted scoring & selection
      - Incremental persistence of alias signatures and edges
      - Optional Connected Components (GraphFrames or pure-Spark fallback)
    """

    spark: SparkSession
    record_id_col: str
    blocking_plan: Dict[str, Dict[str, Any]]
    include_types: Optional[List[str]] = None
    k_required: int = 1
    select_threshold: Optional[float] = None
    select_quantile: float = 0.5

    # Incremental files/dirs
    alias_store_path: Optional[str] = None  # Parquet dir for alias signatures
    edges_path: Optional[str] = (
        None  # Parquet dir for final edges (for incremental clustering)
    )
    matches_json_path: Optional[str] = None  # Optional small JSON line log of matches

    # Safeguards
    block_cap_action: str = "skip"  # "skip" or "keep"
    global_df_cap: Optional[int] = None

    # Cluster computation
    compute_connected_components: bool = False  # If True, compute CC after selection
    max_cc_iters: int = 10  # Fallback CC iterations (if GraphFrames unavailable)

    # Runtime (internal)
    _specs: Dict[str, Dict[str, Any]] = field(default_factory=dict, init=False)

    # ----------------------------- Public API -----------------------------

    def link_incremental(
        self, df_in: DataFrame, batch_id: Optional[str] = None
    ) -> Dict[str, Any]:
        if batch_id is None:
            batch_id = datetime.now().isoformat(timespec="seconds")

        # Compile specs & choose which alias types to include
        self._specs = self._compile_alias_specs(self.blocking_plan)
        include = self.include_types or list(self._specs.keys())

        # Build current-batch alias rows (no write yet)
        alias_curr = self._build_alias_value(df_in, include)

        # Load alias history (cache + materialize so later writes won't clobber lineage)
        alias_hist = None
        if self.alias_store_path and _path_exists(self.spark, self.alias_store_path):
            alias_hist = self.spark.read.parquet(self.alias_store_path).cache()
            alias_hist.count()

        # Build IDF index on a fully materialized basis
        if alias_hist is not None:
            alias_basis = (
                alias_hist.select("alias_key", "alias_type", "owner_id")
                .unionByName(alias_curr.select("alias_key", "alias_type", "owner_id"))
                .dropDuplicates()
                .cache()
            )
            alias_basis.count()
            n_records_total = alias_basis.select(F.countDistinct("owner_id")).first()[0]
            alias_index = self._build_alias_index(alias_basis, n_records_total)
        else:
            n_records_total = df_in.select(F.countDistinct(self.record_id_col)).first()[
                0
            ]
            alias_index = self._build_alias_index(
                alias_curr.select("alias_key", "alias_type", "owner_id"),
                n_records_total,
            )

        # --- Candidate pair generation ---
        pairs_parts: List[DataFrame] = []

        # 1) Within-batch
        for t in include:
            p_intra = self._pairs_from_alias_type(alias_curr, t)
            if p_intra is not None:
                pairs_parts.append(p_intra)

        # 2) Cross-batch (history vs current)
        if alias_hist is not None:
            for t in include:
                p_cross = self._pairs_cross_from_alias_type(alias_hist, alias_curr, t)
                if p_cross is not None:
                    pairs_parts.append(p_cross)

        # K-of-N -> score -> select
        pairs_long = self._candidates_k_of_n(pairs_parts, self.k_required)
        pairs_scored = self._score_pairs(pairs_long, alias_index)
        final_pairs = self._select_pairs(
            pairs_scored, self.select_threshold, self.select_quantile
        )

        # Optional: connected components (computed before any writes)
        out: Dict[str, Any] = {
            "pairs_long": pairs_long,
            "pairs_scored": pairs_scored,
            "final_pairs": final_pairs,
        }
        if self.compute_connected_components:
            cc = self._connected_components_from_edges(final_pairs.select("a", "b"))
            out["connected_components"] = cc

        # ---- Trigger an action BEFORE any dataset writes to finish read-side lineage ----
        # Persist to avoid recomputation across JSON logging and writes
        final_pairs = final_pairs.persist()
        _ = final_pairs.count()

        if self.matches_json_path:
            self._append_pairs_json(final_pairs, batch_id)

        # ---- Writes (append + partition; no overwrite of paths we read) ----
        # Edges (per-batch append)
        if self.edges_path:
            (
                final_pairs.select(
                    "a", "b", "score", "signals"
                )  # keep intended columns
                .dropDuplicates(["a", "b"])  # per-batch dedupe
                .withColumn("batch_id", F.lit(batch_id))
                .withColumn(
                    "timestamp", F.lit(datetime.now().isoformat(timespec="seconds"))
                )
                .write.mode("append")
                .partitionBy("batch_id")
                .parquet(self.edges_path)
            )

        # Alias signatures (append only the current batch)
        if self.alias_store_path:
            (
                alias_curr.withColumn("batch_id", F.lit(batch_id))
                .write.mode("append")
                .partitionBy("alias_type", "batch_id")
                .parquet(self.alias_store_path)
            )

        return out

    # ----------------------------- Specs & Alias -----------------------------

    def _compile_alias_specs(
        self, plan: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        specs: Dict[str, Dict[str, Any]] = {}
        for alias_type, cfg in plan.items():
            fields = cfg["fields"]
            compose = cfg.get("compose")
            block_on = cfg.get("block_on")
            weight = float(cfg.get("weight", 1.0))
            df_cap = cfg.get("df_cap", self.global_df_cap)
            bonus = float(cfg.get("bonus", 0.0))
            similarity = cfg.get("similarity", {"type": "exact"})
            stype = similarity.get("type", "exact").lower()
            if stype not in ("cosine", "exact"):
                stype = "cosine"
            sim = {
                "type": stype,
                "field": similarity.get("field", fields[0]),
                "threshold": float(similarity.get("threshold", 0.8)),
                "analyzer": similarity.get("analyzer", "char_wb"),
                "ngram_range": tuple(similarity.get("ngram_range", (3, 5))),
                "min_df": int(similarity.get("min_df", 1)),
            }
            specs[alias_type] = dict(
                fields=fields,
                compose=compose,
                block_on=block_on,
                weight=weight,
                df_cap=df_cap,
                bonus=bonus,
                similarity=sim,
                alias_type=alias_type,
            )
        return specs

    def _build_alias_value(self, df: DataFrame, include: List[str]) -> DataFrame:
        rid = self.record_id_col
        parts: List[DataFrame] = []

        for atype in include:
            cfg = self._specs[atype]
            fields: List[str] = cfg["fields"]
            compose: Optional[str] = cfg.get("compose")
            block_on: Optional[List[str]] = cfg.get("block_on")
            sim_field: str = cfg["similarity"]["field"]

            if compose:
                # Build a format string replacement by columns; Spark SQL format_string can help
                # But compose may contain {col} patterns; we emulate with successive replacements.
                # Simple approach: concat with delimiter if compose is complex; or use expr with REPLACE is messy.
                # Here: fall back to pipe-join if custom compose provided; you can customize as needed.
                alias_val_col = F.concat_ws(
                    "|",
                    *[F.coalesce(F.col(c).cast("string"), F.lit("")) for c in fields],
                )
            else:
                alias_val_col = F.concat_ws(
                    "|",
                    *[F.coalesce(F.col(c).cast("string"), F.lit("")) for c in fields],
                )

            if block_on:
                block_val_col = F.concat_ws(
                    "|",
                    *[F.coalesce(F.col(c).cast("string"), F.lit("")) for c in block_on],
                )
            else:
                block_val_col = alias_val_col

            part = (
                df.select(
                    F.col(rid).alias("owner_id"),
                    F.lit(atype).alias("alias_type"),
                    alias_val_col.alias("alias_value"),
                    block_val_col.alias("block_value"),
                    F.coalesce(F.col(sim_field).cast("string"), F.lit("")).alias(
                        "sim_value"
                    ),
                )
                .withColumn(
                    "alias_key", F.concat_ws(":", F.lit(atype), F.col("alias_value"))
                )
                .withColumn(
                    "block_key", F.concat_ws(":", F.lit(atype), F.col("block_value"))
                )
                .select("owner_id", "alias_type", "alias_key", "block_key", "sim_value")
            )
            parts.append(part)

        alias_value = parts[0]
        for p in parts[1:]:
            alias_value = alias_value.unionByName(p)
        alias_value = alias_value.dropDuplicates()
        return alias_value

    def _build_alias_index(self, alias_owner: DataFrame, n_records: int) -> DataFrame:
        # alias_owner: cols(alias_key, alias_type, owner_id)
        k1 = alias_owner.groupBy("alias_key", "alias_type").agg(
            F.countDistinct("owner_id").alias("df_count")
        )

        # idf = log((N+1)/(df+1)) + 1
        idx = k1.withColumn(
            "idf",
            F.log((F.lit(n_records) + F.lit(1)) / (F.col("df_count") + F.lit(1)))
            + F.lit(1.0),
        )
        return idx

    # ----------------------------- Pair Generation -----------------------------

    def _pairs_from_alias_type(
        self, alias_value: DataFrame, alias_type: str
    ) -> Optional[DataFrame]:
        cfg = self._specs[alias_type]
        sim = cfg["similarity"]
        sim_type = sim["type"]
        df_cap = cfg.get("df_cap")

        av = alias_value.where(F.col("alias_type") == alias_type).select(
            "alias_key", "alias_type", "owner_id", "block_key", "sim_value"
        )

        if av.rdd.isEmpty():
            return None

        av = av.withColumn(
            "block_key",
            F.when(F.col("block_key").isNull(), F.col("alias_key")).otherwise(
                F.col("block_key")
            ),
        )

        if sim_type == "exact":
            # Group by alias_key, form all 2-combinations of owners
            sizes = av.groupBy("block_key").agg(F.count("*").alias("n"))
            av2 = av.join(sizes, "block_key", "left")
            if df_cap is not None and self.block_cap_action == "skip":
                av2 = av2.where(F.col("n") <= F.lit(int(df_cap)))

            by_alias = av.groupBy("alias_key").agg(
                F.collect_set("owner_id").alias("owners")
            )

            # UDF to make 2-combinations
            make_pairs = F.udf(
                lambda xs: [
                    (min(a, b), max(a, b))
                    for i, a in enumerate(xs)
                    for b in xs[i + 1 :]
                    if a != b
                ],
                T.ArrayType(
                    T.StructType(
                        [
                            T.StructField("a", T.StringType()),
                            T.StructField("b", T.StringType()),
                        ]
                    )
                ),
            )

            pairs = (
                by_alias.withColumn("pairs", make_pairs(F.col("owners")))
                .select(
                    "alias_key", F.explode("pairs").alias("pair")
                )  # <-- keep 'pairs' here
                .select(
                    F.col("pair.a").alias("a"),
                    F.col("pair.b").alias("b"),
                    F.lit(alias_type).alias("signal"),
                    F.col("alias_key"),
                    F.lit(alias_type).alias("alias_type"),
                    F.lit(1.0).alias("sim"),
                )
                .dropDuplicates(["a", "b", "alias_type"])
            )
            return pairs

        # cosine path: compare within each block
        analyzer = sim["analyzer"]
        nlo, nhi = sim["ngram_range"]
        threshold = float(sim["threshold"])
        min_df = int(sim["min_df"])

        # Tokenize to char ngrams
        if analyzer == "char_wb":
            tokens = av.withColumn(
                "tokens", udf_charwb("sim_value", F.lit(nlo), F.lit(nhi))
            )
        else:
            tokens = av.withColumn(
                "tokens", udf_char("sim_value", F.lit(nlo), F.lit(nhi))
            )

        # Vectorize per alias_type + block_key
        # NOTE: We'll vectorize per block to keep vocab small; we do it with CountVectorizer+IDF
        # Build per-block datasets, then self-join pairs
        # First cap block sizes if requested
        sizes = tokens.groupBy("block_key").agg(F.count("*").alias("n"))
        tok2 = tokens.join(sizes, "block_key", "left")
        if df_cap is not None and self.block_cap_action == "skip":
            tok2 = tok2.where(F.col("n") <= F.lit(int(df_cap)))

        # For performance, compute per-block sequentially (collect blocks). In practice, replace with mapGroupsInPandas if needed.
        blocks = [r["block_key"] for r in tok2.select("block_key").distinct().collect()]
        out_parts: List[DataFrame] = []

        for blk in blocks:
            g = tok2.where(F.col("block_key") == F.lit(blk))
            if g.count() < 2:
                continue

            cv = CountVectorizer(inputCol="tokens", outputCol="tf", minDF=min_df)
            cvm = cv.fit(g)
            g_tf = cvm.transform(g)

            idf = IDF(inputCol="tf", outputCol="tfidf")
            idfm = idf.fit(g_tf)
            g_vec = idfm.transform(g_tf).select(
                "owner_id", "alias_type", "alias_key", "block_key", "tfidf"
            )

            # self-join on block_key to compare pairs (upper triangle)
            a = g_vec.alias("a")
            b = g_vec.alias("b")
            paired = a.join(
                b,
                (F.col("a.block_key") == F.col("b.block_key"))
                & (F.col("a.owner_id") < F.col("b.owner_id")),
            ).select(
                F.col("a.owner_id").alias("a"),
                F.col("b.owner_id").alias("b"),
                F.col("a.alias_key").alias("alias_key_a"),
                F.col("a.alias_type").alias("alias_type"),
                F.col("a.tfidf").alias("va"),
                F.col("b.tfidf").alias("vb"),
            )

            # cosine
            pairs_blk = (
                paired.withColumn("sim", udf_cosine(F.col("va"), F.col("vb")))
                .where(F.col("sim") >= F.lit(threshold))
                .select(
                    "a",
                    "b",
                    F.lit(alias_type).alias("signal"),
                    F.col("alias_key_a").alias("alias_key"),
                    F.lit(alias_type).alias("alias_type"),
                    "sim",
                )
            )
            if pairs_blk.rdd.isEmpty():
                continue
            out_parts.append(pairs_blk)

        if not out_parts:
            return None
        res = out_parts[0]
        for p in out_parts[1:]:
            res = res.unionByName(p)
        return res.dropDuplicates(["a", "b", "alias_type"])

    def _pairs_cross_from_alias_type(
        self, av_hist: DataFrame, av_curr: DataFrame, alias_type: str
    ) -> Optional[DataFrame]:
        cfg = self._specs[alias_type]
        sim = cfg["similarity"]
        sim_type = sim["type"]
        df_cap = cfg.get("df_cap")

        H = av_hist.where(F.col("alias_type") == alias_type).select(
            "alias_key", "alias_type", "owner_id", "block_key", "sim_value"
        )
        C = av_curr.where(F.col("alias_type") == alias_type).select(
            "alias_key", "alias_type", "owner_id", "block_key", "sim_value"
        )

        if H.rdd.isEmpty() or C.rdd.isEmpty():
            return None

        if sim_type == "exact":
            joined = (
                H.alias("h")
                .join(C.alias("c"), on=["alias_key", "alias_type"], how="inner")
                .select(
                    F.col("h.owner_id").alias("a"),
                    F.col("c.owner_id").alias("b"),
                    F.col("h.alias_key").alias("alias_key"),
                )
                .where(F.col("a") != F.col("b"))
            )

            if df_cap is not None:
                blk_sizes = joined.groupBy("alias_key").agg(F.count("*").alias("n"))
                joined = (
                    joined.join(blk_sizes, "alias_key", "left")
                    .where(F.col("n") <= F.lit(int(df_cap)))
                    .drop("n")
                )

            dedup = joined.dropDuplicates(["a", "b", "alias_key"])
            if dedup.rdd.isEmpty():
                return None

            return dedup.select(
                "a",
                "b",
                F.lit(alias_type).alias("signal"),
                "alias_key",
                F.lit(alias_type).alias("alias_type"),
                F.lit(1.0).alias("sim"),
            )

        # cosine: compare within shared blocks
        analyzer = sim["analyzer"]
        nlo, nhi = sim["ngram_range"]
        threshold = float(sim["threshold"])
        min_df = int(sim["min_df"])

        Ht = H.withColumn(
            "tokens",
            (
                udf_charwb("sim_value", F.lit(nlo), F.lit(nhi))
                if analyzer == "char_wb"
                else udf_char("sim_value", F.lit(nlo), F.lit(nhi))
            ),
        )
        Ct = C.withColumn(
            "tokens",
            (
                udf_charwb("sim_value", F.lit(nlo), F.lit(nhi))
                if analyzer == "char_wb"
                else udf_char("sim_value", F.lit(nlo), F.lit(nhi))
            ),
        )

        shared = (
            Ht.select("block_key")
            .distinct()
            .join(Ct.select("block_key").distinct(), "block_key", "inner")
        )
        blocks = [r["block_key"] for r in shared.collect()]

        out_parts: List[DataFrame] = []

        for blk in blocks:
            gH = Ht.where(F.col("block_key") == F.lit(blk))
            gC = Ct.where(F.col("block_key") == F.lit(blk))
            n = gH.count() + gC.count()
            if n < 2:
                continue
            if (
                df_cap is not None
                and self.block_cap_action == "skip"
                and n > int(df_cap)
            ):
                continue

            # Fit CV/IDF on combined vocab (H + C)
            base = gH.unionByName(gC)
            cv = CountVectorizer(inputCol="tokens", outputCol="tf", minDF=min_df)
            cvm = cv.fit(base)
            base_tf = cvm.transform(base)
            idf = IDF(inputCol="tf", outputCol="tfidf")
            idfm = idf.fit(base_tf)
            base_vec = idfm.transform(base_tf)

            Hvec = base_vec.join(gH.select("owner_id"), ["owner_id"], "inner").select(
                "owner_id", "tfidf"
            )
            Cvec = base_vec.join(gC.select("owner_id"), ["owner_id"], "inner").select(
                "owner_id", "tfidf"
            )

            a = Hvec.alias("h")
            b = Cvec.alias("c")

            paired = (
                a.crossJoin(b)
                .where(F.col("h.owner_id") < F.col("c.owner_id"))
                .select(
                    F.col("h.owner_id").alias("a"),
                    F.col("c.owner_id").alias("b"),
                    F.col("h.tfidf").alias("va"),
                    F.col("c.tfidf").alias("vb"),
                )
            )

            pblk = (
                paired.withColumn("sim", udf_cosine(F.col("va"), F.col("vb")))
                .where(F.col("sim") >= F.lit(threshold))
                .select(
                    "a",
                    "b",
                    F.lit(alias_type).alias("signal"),
                    F.lit(blk).alias("alias_key"),
                    F.lit(alias_type).alias("alias_type"),
                    "sim",
                )
            )
            if pblk.rdd.isEmpty():
                continue
            out_parts.append(pblk)

        if not out_parts:
            return None
        res = out_parts[0]
        for p in out_parts[1:]:
            res = res.unionByName(p)
        return res.dropDuplicates(["a", "b", "alias_type"])

    # ----------------------------- K-of-N & Scoring -----------------------------

    def _candidates_k_of_n(self, parts: List[DataFrame], k_required: int) -> DataFrame:
        nonempty = [p for p in parts if p is not None and not p.rdd.isEmpty()]
        if not nonempty:
            schema = T.StructType(
                [
                    T.StructField("a", T.StringType()),
                    T.StructField("b", T.StringType()),
                    T.StructField("signal", T.StringType()),
                    T.StructField("alias_key", T.StringType()),
                    T.StructField("alias_type", T.StringType()),
                    T.StructField("sim", T.DoubleType()),
                ]
            )
            return self.spark.createDataFrame(
                self.spark.sparkContext.emptyRDD(), schema
            )

        temp = nonempty[0]
        for p in nonempty[1:]:
            temp = temp.unionByName(p)

        keep = (
            temp.select("a", "b", "signal")
            .dropDuplicates()
            .groupBy("a", "b")
            .agg(F.count("*").alias("n_signals"))
            .where(F.col("n_signals") >= F.lit(int(k_required)))
            .select("a", "b")
        )

        return temp.join(keep, ["a", "b"], "inner")

    def _score_pairs(self, pairs_long: DataFrame, alias_index: DataFrame) -> DataFrame:
        if pairs_long.rdd.isEmpty():
            schema = T.StructType(
                [
                    T.StructField("a", T.StringType()),
                    T.StructField("b", T.StringType()),
                    T.StructField("score", T.DoubleType()),
                    T.StructField("signals", T.ArrayType(T.StringType())),
                ]
            )
            return self.spark.createDataFrame(
                self.spark.sparkContext.emptyRDD(), schema
            )

        weight_map = {t: self._specs[t]["weight"] for t in self._specs}
        bonus_map = {t: self._specs[t]["bonus"] for t in self._specs}

        # join IDF
        tmp = (
            pairs_long.join(
                alias_index.select("alias_key", "alias_type", "idf"),
                ["alias_key", "alias_type"],
                "left",
            )
            .withColumn("idf", F.coalesce(F.col("idf"), F.lit(0.0)))
            .withColumn(
                "w",
                F.create_map(
                    [
                        F.lit(kv)
                        for kv in sum(
                            ([k, F.lit(v)] for k, v in weight_map.items()), []
                        )
                    ]
                ).getItem(F.col("signal")),
            )
            .withColumn("w", F.coalesce(F.col("w"), F.lit(1.0)))
            .withColumn("sim", F.coalesce(F.col("sim"), F.lit(1.0)))
            .withColumn("part_score", F.col("idf") * F.col("w") * F.col("sim"))
        )

        agg = tmp.groupBy("a", "b").agg(
            F.sum("part_score").alias("score"),
            F.array_sort(F.array_distinct(F.collect_list("signal"))).alias("signals"),
        )

        pres = pairs_long.select("a", "b", "signal").dropDuplicates()
        bonus = (
            pres.withColumn(
                "bonus",
                F.create_map(
                    [
                        F.lit(kv)
                        for kv in sum(([k, F.lit(v)] for k, v in bonus_map.items()), [])
                    ]
                ).getItem(F.col("signal")),
            )
            .groupBy("a", "b")
            .agg(F.sum(F.coalesce(F.col("bonus"), F.lit(0.0))).alias("bonus"))
        )

        s = (
            agg.join(bonus, ["a", "b"], "left")
            .withColumn(
                "score", F.col("score") + F.coalesce(F.col("bonus"), F.lit(0.0))
            )
            .drop("bonus")
        )
        return s

    def _select_pairs(
        self, pairs_scored: DataFrame, threshold: Optional[float], quantile: float
    ) -> DataFrame:
        if pairs_scored.rdd.isEmpty():
            return pairs_scored
        if threshold is None:
            q = pairs_scored.approxQuantile("score", [float(quantile)], 0.0)[0]
            t = float(q)
        else:
            t = float(threshold)
        return pairs_scored.where(F.col("score") >= F.lit(t))

    # ----------------------------- Clustering (optional) -----------------------------

    from pyspark.storagelevel import StorageLevel

    def _connected_components_from_edges(self, edges: DataFrame) -> DataFrame:
        if edges.rdd.isEmpty():
            return edges

        verts = (
            edges.select(F.col("a").alias("id"))
            .unionByName(edges.select(F.col("b").alias("id")))
            .dropDuplicates()
        )
        # If GraphFrames are present, keep using them
        if GF_AVAILABLE:
            gf = GraphFrame(
                verts,
                edges.select(
                    F.col("a").alias("src"), F.col("b").alias("dst")
                ).dropDuplicates(),
            )
            return gf.connectedComponents()

        # ---- SMALL GRAPH FAST PATH ----
        n_verts = verts.count()
        if n_verts <= 10000:
            # driver-side union-find; trivial for your 6 records
            edgelist = [(r["a"], r["b"]) for r in edges.select("a", "b").collect()]
            parent = {}

            def find(x):
                parent.setdefault(x, x)
                if parent[x] != x:
                    parent[x] = find(parent[x])
                return parent[x]

            def union(x, y):
                rx, ry = find(x), find(y)
                if rx != ry:
                    parent[ry] = rx

            for a, b in edgelist:
                union(a, b)
            comps = [(vid, find(vid)) for vid, in verts.select("id").collect()]
            out = self.spark.createDataFrame(
                comps,
                schema=T.StructType(
                    [
                        T.StructField("id", T.StringType()),
                        T.StructField("component", T.StringType()),
                    ]
                ),
            )
            return out

        # ---- else fall back to distributed min-label with stricter hygiene ----
        e = (
            edges.select(F.col("a").alias("src"), F.col("b").alias("dst"))
            .dropDuplicates()
            .persist(StorageLevel.MEMORY_AND_DISK)
        )
        _ = e.count()

        lbl = verts.withColumn("component", F.col("id")).persist(
            StorageLevel.MEMORY_AND_DISK
        )
        _ = lbl.count()

        for _ in range(self.max_cc_iters):
            prop1 = e.join(
                lbl.select(
                    F.col("id").alias("nbr"), F.col("component").alias("nbr_comp")
                ),
                F.col("src") == F.col("nbr"),
                "left",
            ).select(F.col("dst").alias("id"), F.col("nbr_comp").alias("candidate"))
            prop2 = e.join(
                lbl.select(
                    F.col("id").alias("nbr"), F.col("component").alias("nbr_comp")
                ),
                F.col("dst") == F.col("nbr"),
                "left",
            ).select(F.col("src").alias("id"), F.col("nbr_comp").alias("candidate"))

            next_lbl = (
                prop1.unionByName(prop2)
                .groupBy("id")
                .agg(F.min("candidate").alias("min_cand"))
                .join(lbl, "id", "right")
                .withColumn(
                    "component",
                    F.least(
                        F.col("component"),
                        F.coalesce(F.col("min_cand"), F.col("component")),
                    ),
                )
                .select("id", "component")
                .persist(StorageLevel.MEMORY_AND_DISK)
            )
            _ = next_lbl.count()

            changed_any = (
                next_lbl.alias("nl")
                .join(lbl.alias("lb"), ["id"], "inner")
                .where(F.col("nl.component") != F.col("lb.component"))
                .limit(1)
                .collect()
            )

            lbl.unpersist()
            lbl = next_lbl
            if not changed_any:
                break

        # e.unpersist()  # optional
        return lbl

    # ----------------------------- Persistence -----------------------------

    def _append_pairs_json(self, final_pairs: DataFrame, batch_id: str) -> None:
        # Collect only light metadata for JSON logging
        rows = (
            final_pairs.withColumn("batch_id", F.lit(batch_id))
            .withColumn(
                "timestamp", F.lit(datetime.now().isoformat(timespec="seconds"))
            )
            .select("a", "b", "score", "signals", "batch_id", "timestamp")
            .toLocalIterator()
        )
        data = [
            {
                "a": r["a"],
                "b": r["b"],
                "score": float(r["score"]),
                "signals": list(r["signals"]) if r["signals"] is not None else [],
                "batch_id": r["batch_id"],
                "timestamp": r["timestamp"],
            }
            for r in rows
        ]
        try:
            with open(self.matches_json_path, "a") as f:
                for d in data:
                    f.write(json.dumps(d) + "\n")
        except Exception:
            pass


# ----------------------------- Example -----------------------------
if __name__ == "__main__":
    spark = (
        SparkSession.builder.appName("ER-Incremental-PySpark")
        # .config("spark.jars.packages", "graphframes:graphframes:0.8.3-spark3.5-s_2.12")  # if you want GraphFrames
        .getOrCreate()
    )

    # Example input
    records_batch1 = spark.createDataFrame(
        [
            {
                "record_id": "R1",
                "name_key": "ACMSPPLY",
                "zip5": "10011",
                "phone": "+12125550101",
                "geo": "NY|100",
            },
            {
                "record_id": "R2",
                "name_key": "ACMSPPLY",
                "zip5": "10011",
                "phone": "+12125550101",
                "geo": "NY|100",
            },
            {
                "record_id": "R3",
                "name_key": "ACMSPPLY",
                "zip5": "10012",
                "phone": "+12125550101",
                "geo": "NY|100",
            },
        ]
    )

    records_batch2 = spark.createDataFrame(
        [
            {
                "record_id": "R4",
                "name_key": "GLOBEXLLC",
                "zip5": "94107",
                "phone": "+14155550123",
                "geo": "CA|941",
            },
            {
                "record_id": "R5",
                "name_key": "GLOBEXLTD",
                "zip5": "94107",
                "phone": "+14155550123",
                "geo": "CA|941",
            },
            {
                "record_id": "R6",
                "name_key": "ACMESUPPLY",
                "zip5": "10012",
                "phone": "+12125550101",
                "geo": "NY|100",
            },
        ]
    )

    blocking_plan = {
        "PHONE": {
            "fields": ["phone"],
            "weight": 1.0,
            "bonus": 0.5,
            "df_cap": 50000,
        },
        "NAME_KEY_COSINE": {
            "fields": ["name_key"],
            "block_on": ["zip5"],  # compare within zip5 buckets
            "weight": 0.95,
            "df_cap": 20000,
            "similarity": {
                "type": "cosine",
                "field": "name_key",
                "threshold": 0.25,
                "analyzer": "char_wb",  # "char_wb" or "char"
                "ngram_range": (2, 3),
                "min_df": 1,
            },
        },
    }

    linker = IncrementalSignalLinkerSpark(
        spark=spark,
        record_id_col="record_id",
        blocking_plan=blocking_plan,
        include_types=["NAME_KEY_COSINE", "PHONE"],
        k_required=1,
        select_threshold=None,
        select_quantile=0.5,
        alias_store_path="parquet_alias_store",
        edges_path="parquet_edges",
        matches_json_path="matches_log.json",
        compute_connected_components=True,  # set False if you don't need CC now
        max_cc_iters=12,
    )

    print("=== Batch 1 ===")
    out1 = linker.link_incremental(records_batch1, batch_id="batch1")
    out1["final_pairs"].show(truncate=False)
    if "connected_components" in out1:
        out1["connected_components"].show(truncate=False)

    print("=== Batch 2 ===")
    out2 = linker.link_incremental(records_batch2, batch_id="batch2")
    out2["final_pairs"].show(truncate=False)
    if "connected_components" in out2:
        out2["connected_components"].show(truncate=False)

    # Edges persisted in parquet_edges; alias signatures in parquet_alias_store
    spark.stop()
