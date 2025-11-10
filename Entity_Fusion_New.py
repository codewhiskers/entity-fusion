import pandas as pd
import pdb
from typing import Callable, Dict, List, Optional, Any, Iterable, Tuple
import pandas as pd
import numpy as np
import json, pathlib
import networkx as nx
from itertools import combinations
import pandas as pd
import numpy as np
import math
import networkx as nx


records = pd.DataFrame(
    [
        {
            "record_id": "R1",
            "name_key": "ACMSPPLY",
            "zip5": "10011",
            "phone": "+12125550101",
            "geo": "NY|100",
            "name_zip5": "ACMSPPLY|10011",
        },
        {
            "record_id": "R2",
            "name_key": "ACMSPPLY",
            "zip5": "10011",
            "phone": "+12125550101",
            "geo": "NY|100",
            "name_zip5": "ACMSPPLY|10011",
        },
        {
            "record_id": "R3",
            "name_key": "ACMSPPLY",
            "zip5": "10012",
            "phone": "+12125550101",
            "geo": "NY|100",
            "name_zip5": "ACMSPPLY|10012",
        },
        {
            "record_id": "R4",
            "name_key": "GLOBEXLLC",
            "zip5": "94107",
            "phone": "+14155550123",
            "geo": "CA|941",
            "name_zip5": "GLOBEXLLC|94107",
        },
        {
            "record_id": "R5",
            "name_key": "GLOBEXLTD",
            "zip5": "94107",
            "phone": "+14155550123",
            "geo": "CA|941",
            "name_zip5": "GLOBEXLTD|94107",
        },
    ]
)

# Example: choose exactly which normalized fields to use
blocking_plan = {
    "NAME_KEY": {
        "fields": ["name_key"],  # single field
        "weight": 1.0,  # used in scoring (optional)
        "df_cap": None,  # cap giant buckets (optional)
    },
    "ZIP5": {"fields": ["zip5"], "weight": 1.0, "df_cap": 20000},
    "PHONE": {
        "fields": ["phone"],
        "weight": 1.0,
        "bonus": 0.5,  # extra score if present (optional)
        "df_cap": 50000,
    },
    "GEO": {"fields": ["geo"], "weight": 0.9},
    "NAME_KEY+ZIP5": {
        "fields": ["name_key", "zip5"],  # composite
        "compose": "{name_key}|{zip5}",  # default is join with '|', but you can set explicit template
        "weight": 0.95,
        "df_cap": 20000,
    },
}


def compile_alias_specs(blocking_plan):
    """Turn plan into: {alias_type: (extractor_fn, meta)}"""
    specs = {}
    for alias_type, cfg in blocking_plan.items():
        fields = cfg["fields"]
        compose = cfg.get("compose")
        weight = float(cfg.get("weight", 1.0))
        df_cap = cfg.get("df_cap")
        bonus = float(cfg.get("bonus", 0.0))

        if compose:
            # template compose
            def make_fn(fields, tmpl):
                return lambda row, _fields=fields, _tmpl=tmpl: (
                    _tmpl.format(
                        **{
                            f: (row.get(f) if pd.notna(row.get(f)) else "")
                            for f in _fields
                        }
                    )
                )

            fn = make_fn(fields, compose)
        else:
            # default: single field or join with '|'
            if len(fields) == 1:
                f = fields[0]
                fn = lambda row, _f=f: row.get(_f)
            else:

                def make_fn(fields):
                    return lambda row, _fields=fields: "|".join(
                        [(row.get(f) if pd.notna(row.get(f)) else "") for f in _fields]
                    )

                fn = make_fn(fields)

        specs[alias_type] = {
            "fn": fn,
            "weight": weight,
            "df_cap": df_cap,
            "bonus": bonus,
        }
    return specs


def make_alias_specs():
    """
    For already-normalized records.
    Each key maps to (lambda row: row[colname], confidence).
    """
    return {
        "NAME_KEY": (lambda r: r.get("name"), 1.0),
        "ZIP5": (lambda r: r.get("zip5"), 1.0),
        "PHONE": (lambda r: r.get("phone"), 1.0),
        "GEO": (lambda r: r.get("state"), 0.9),
        "NAME_KEY+ZIP5": (lambda r: r.get("name_zip5"), 0.95),
    }


def build_alias_value(
    records: pd.DataFrame, record_id_col: str, specs: dict, include_types=None
) -> pd.DataFrame:
    if include_types is None:
        include_types = list(specs.keys())

    rows = []
    for _, row in records.iterrows():
        rid = row[record_id_col]
        for atype in include_types:
            fn = specs[atype]["fn"]
            key = fn(row)
            if key:
                rows.append(
                    {
                        "alias_key": f"{atype}:{key}",
                        "alias_type": atype,
                        "owner_id": rid,
                    }
                )
    return pd.DataFrame(rows, columns=["alias_key", "alias_type", "owner_id"])


def build_alias_index(alias_value: pd.DataFrame, n_records: int) -> pd.DataFrame:
    idx = alias_value.groupby(["alias_key", "alias_type"], as_index=False).agg(
        df_count=("owner_id", "nunique")
    )
    # smoothed IDF
    idx["idf"] = idx["df_count"].apply(
        lambda df: math.log((n_records + 1) / (df + 1)) + 1.0
    )
    return idx  # alias_key, alias_type, df_count, idf


def pairs_from_alias_type(
    alias_value: pd.DataFrame, alias_type: str, df_cap: int = None
) -> pd.DataFrame:
    av = alias_value[alias_value["alias_type"] == alias_type][["alias_key", "owner_id"]]
    if df_cap is not None:
        counts = av["alias_key"].value_counts()
        av = av[av["alias_key"].isin(counts[counts <= df_cap].index)]

    rows = []
    for key, g in av.groupby("alias_key"):
        owners = sorted(g["owner_id"].unique())
        for i in range(len(owners) - 1):
            for j in range(i + 1, len(owners)):
                rows.append(
                    {
                        "a": owners[i],
                        "b": owners[j],
                        "signal": alias_type,
                        "alias_key": key,
                    }
                )
    return (
        pd.DataFrame(rows, columns=["a", "b", "signal", "alias_key"])
        if rows
        else pd.DataFrame(columns=["a", "b", "signal", "alias_key"])
    )


def candidates_k_of_n(pairs_by_signal, k_required=2):
    parts = [p for p in pairs_by_signal if p is not None and not p.empty]
    if not parts:
        return pd.DataFrame(columns=["a", "b", "signal", "alias_key"])
    temp = pd.concat(parts, ignore_index=True)
    keep = (
        temp.groupby(["a", "b"])["signal"]
        .nunique()
        .reset_index()
        .query("signal >= @k_required")[["a", "b"]]
    )
    return temp.merge(keep, on=["a", "b"], how="inner")


def score_pairs(
    pairs_long: pd.DataFrame, alias_index: pd.DataFrame, specs: dict
) -> pd.DataFrame:
    if pairs_long.empty:
        return pd.DataFrame(columns=["a", "b", "score", "signals"])

    idf_map = alias_index.set_index("alias_key")["idf"].to_dict()
    weight_map = {t: specs[t]["weight"] for t in specs.keys()}
    bonus_map = {t: specs[t]["bonus"] for t in specs.keys()}

    # base score = sum(idf * type_weight) over shared alias_keys
    tmp = pairs_long.copy()
    tmp["idf"] = tmp["alias_key"].map(idf_map).fillna(0.0)
    tmp["w"] = tmp["signal"].map(weight_map).fillna(1.0)
    tmp["part_score"] = tmp["idf"] * tmp["w"]

    s = tmp.groupby(["a", "b"], as_index=False).agg(
        score=("part_score", "sum"), signals=("signal", lambda x: sorted(set(x)))
    )

    # add bonuses once per type appearance
    pres = pairs_long[["a", "b", "signal"]].drop_duplicates()
    pres["bonus"] = pres["signal"].map(bonus_map).fillna(0.0)
    bonus = pres.groupby(["a", "b"], as_index=False)["bonus"].sum()
    s = s.merge(bonus, on=["a", "b"], how="left")
    s["score"] = s["score"] + s["bonus"].fillna(0.0)
    s.drop(columns=["bonus"], inplace=True)
    return s


def select_pairs(
    pairs_scored: pd.DataFrame, threshold: float = None, quantile: float = 0.5
) -> pd.DataFrame:
    if pairs_scored.empty:
        return pairs_scored
    t = (
        threshold
        if threshold is not None
        else float(pairs_scored["score"].quantile(quantile))
    )
    return pairs_scored[pairs_scored["score"] >= t].copy()


specs = compile_alias_specs(blocking_plan)
include = ["NAME_KEY", "ZIP5", "PHONE", "GEO"]  # choose any subset dynamically

alias_value = build_alias_value(
    records, record_id_col="record_id", specs=specs, include_types=include
)
alias_index = build_alias_index(alias_value, n_records=records["record_id"].nunique())
pairs_list = [
    pairs_from_alias_type(alias_value, t, df_cap=specs[t]["df_cap"]) for t in include
]


pairs_long = candidates_k_of_n(pairs_list, k_required=2)  # K is dynamic too

pairs_scored = score_pairs(pairs_long, alias_index, specs)
final_pairs = select_pairs(pairs_scored, quantile=0.5)
pdb.set_trace()
