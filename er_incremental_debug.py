from __future__ import annotations

import argparse
import json
import pdb
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from Incremental_Fusion import IncrementalSignalLinker


ROOT = Path(__file__).resolve().parent
DEBUG_DIR = ROOT / "test_data" / "_debug_outputs"


def make_sample_batches() -> tuple[pd.DataFrame, pd.DataFrame]:
    batch1 = pd.DataFrame(
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

    batch2 = pd.DataFrame(
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
    return batch1, batch2


def make_flat_blocking_plan() -> Dict[str, Dict[str, Any]]:
    return {
        "PHONE": {
            "fields": ["phone"],
            "weight": 1.0,
            "bonus": 0.5,
            "df_cap": 50000,
        },
        "NAME_KEY_COSINE": {
            "fields": ["name_key"],
            "block_on": ["zip5"],
            "weight": 0.95,
            "df_cap": 20000,
            "similarity": {
                "type": "cosine",
                "field": "name_key",
                "threshold": 0.25,
                "analyzer": "char_wb",
                "ngram_range": (2, 3),
                "min_df": 1,
            },
        },
    }


def make_condition_tree_plan() -> Dict[str, Any]:
    flat = make_flat_blocking_plan()
    return {
        "or": [
            {
                "and": [
                    {"PHONE": flat["PHONE"]},
                    {"NAME_KEY_COSINE": flat["NAME_KEY_COSINE"]},
                ]
            },
            {"PHONE": flat["PHONE"]},
        ]
    }


def build_linker(
    *,
    condition_tree: bool = False,
    persist_matches: bool = False,
    create_network: bool = False,
) -> IncrementalSignalLinker:
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    matches_file = DEBUG_DIR / "matches_debug.json" if persist_matches else None
    cluster_history_file = DEBUG_DIR / "cluster_history_debug.json"

    return IncrementalSignalLinker(
        record_id_col="record_id",
        blocking_plan=(
            make_condition_tree_plan() if condition_tree else make_flat_blocking_plan()
        ),
        include_types=["NAME_KEY_COSINE", "PHONE"],
        k_required=1,
        select_threshold=None,
        select_quantile=0.5,
        enable_incremental=True,
        matches_file=str(matches_file) if matches_file else None,
        cluster_history_file=str(cluster_history_file),
        create_network=create_network,
        network_output_dir=str(DEBUG_DIR / "network_outputs"),
        return_intermediates=True,
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def summarize_result(name: str, result: Dict[str, Any]) -> None:
    print(f"\n=== {name} ===")
    print("final_pairs")
    print(result["final_pairs"])
    print("\nclusters")
    print(json.dumps(_json_safe(result["clusters"]), indent=2))
    print("\ncluster_stats")
    print(json.dumps(_json_safe(result["cluster_stats"]), indent=2, default=str))
    print("\nlabeled_records")
    print(result["labeled_records"])


def run_incremental_demo(
    *,
    condition_tree: bool = False,
    persist_matches: bool = False,
    create_network: bool = False,
) -> Dict[str, Any]:
    batch1, batch2 = make_sample_batches()
    linker = build_linker(
        condition_tree=condition_tree,
        persist_matches=persist_matches,
        create_network=create_network,
    )

    result1 = linker.link_incremental(batch1, batch_id="batch1")
    result2 = linker.link_incremental(batch2, batch_id="batch2")

    summarize_result("batch1", result1)
    summarize_result("batch2", result2)

    return {
        "linker": linker,
        "batch1": batch1,
        "batch2": batch2,
        "result1": result1,
        "result2": result2,
    }


def run_single_batch_demo(*, condition_tree: bool = False) -> Dict[str, Any]:
    batch1, batch2 = make_sample_batches()
    all_records = pd.concat([batch1, batch2], ignore_index=True)
    linker = build_linker(condition_tree=condition_tree, persist_matches=False)
    result = linker.link(all_records)
    summarize_result("single_batch", result)
    return {"linker": linker, "records": all_records, "result": result}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Debug harness for IncrementalSignalLinker."
    )
    parser.add_argument(
        "--mode",
        choices=["incremental", "single-batch"],
        default="incremental",
    )
    parser.add_argument(
        "--condition-tree",
        action="store_true",
        help="Use the AND/OR tree blocking plan instead of the flat plan.",
    )
    parser.add_argument(
        "--persist-matches",
        action="store_true",
        help="Write and reuse matches in test_data/_debug_outputs/matches_debug.json.",
    )
    parser.add_argument(
        "--network",
        action="store_true",
        help="Create pyvis HTML outputs in test_data/_debug_outputs/network_outputs.",
    )
    parser.add_argument(
        "--pdb",
        action="store_true",
        help="Drop into pdb after running the selected scenario.",
    )
    args = parser.parse_args()

    if args.mode == "incremental":
        ctx = run_incremental_demo(
            condition_tree=args.condition_tree,
            persist_matches=args.persist_matches,
            create_network=args.network,
        )
    else:
        ctx = run_single_batch_demo(condition_tree=args.condition_tree)

    if args.pdb:
        pdb.set_trace()
        return ctx


if __name__ == "__main__":
    main()
