from __future__ import annotations

import json
import pdb
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from Incremental_Fusion import IncrementalSignalLinker


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "test_data" / "100k.csv"
DEFAULT_OUTPUT_DIR = ROOT / "test_data" / "_debug_outputs"


def make_ppp_blocking_plan(
    name_threshold: float, address_threshold: float
) -> Dict[str, Any]:
    return {
        "or": [
            {
                "BORROWER_NAME": {
                    "fields": ["BorrowerName"],
                    "block_on": ["BorrowerCity"],
                    "weight": 1.0,
                    "df_cap": 20000,
                    "similarity": {
                        "type": "cosine",
                        "field": "BorrowerName",
                        "threshold": name_threshold,
                        "analyzer": "char_wb",
                        "ngram_range": (2, 4),
                        "min_df": 1,
                    },
                }
            },
            {
                "BORROWER_ADDRESS": {
                    "fields": ["BorrowerAddress"],
                    "block_on": ["BorrowerCity"],
                    "weight": 1.0,
                    "df_cap": 20000,
                    "similarity": {
                        "type": "cosine",
                        "field": "BorrowerAddress",
                        "threshold": address_threshold,
                        "analyzer": "char_wb",
                        "ngram_range": (2, 4),
                        "min_df": 1,
                    },
                }
            },
        ]
    }


def prepare_dataframe(path: Path, nrows: int | None) -> pd.DataFrame:
    df = pd.read_csv(path, nrows=nrows)
    df = df.reset_index().rename(columns={"index": "record_id"})
    df["record_id"] = df["record_id"].astype(str)

    # Keep the matching columns string-like and fill nulls so the linker can work
    # directly with the PPP source data.
    for col in ["BorrowerName", "BorrowerAddress", "BorrowerCity"]:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str).str.strip()

    return df


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


def summarize(result: Dict[str, Any]) -> None:
    print("\nfinal_pairs")
    print(result["final_pairs"])
    if "cluster_stats" in result:
        print("\ncluster_stats")
        print(json.dumps(_json_safe(result["cluster_stats"]), indent=2))
    if "clusters" in result:
        print("\ncluster_count")
        print(len(result["clusters"]))

    labeled = result["labeled_records"]
    non_singletons = labeled.filter(
        ~labeled["cluster_label"].str.starts_with("singleton_")
    )
    print("\nlabeled_records preview")
    print(labeled.head(10))
    print(f"\nmatched rows: {non_singletons.height}")
    print(f"total rows: {labeled.height}")
    if "clusters" in result:
        print(f"clusters: {len(result['clusters'])}")


def get_cluster_sizes(result: Dict[str, Any]) -> list[tuple[str, int]]:
    if "clusters" not in result:
        return []
    sizes = [
        (cluster_label, len(members))
        for cluster_label, members in result["clusters"].items()
    ]
    return sorted(sizes, key=lambda item: (-item[1], item[0]))


def main() -> None:
    df = prepare_dataframe(DEFAULT_INPUT, nrows=10_000)

    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    linker = IncrementalSignalLinker(
        record_id_col="record_id",
        blocking_plan=make_ppp_blocking_plan(name_threshold=0.9, address_threshold=0.9),
        include_types=["BORROWER_NAME", "BORROWER_ADDRESS"],
        k_required=1,
        select_threshold=None,
        select_quantile=0.5,
        enable_incremental=True,
        matches_file=None,
        create_network=False,
        network_output_dir=str(DEFAULT_OUTPUT_DIR / "network_outputs"),
        return_intermediates=True,
        show_progress=True,
    )

    result = linker.link_incremental(df, batch_id="100k")
    summarize(result)

    # largest_clusters = get_cluster_sizes(result)[:10]
    # if largest_clusters:
    #     print("\nlargest clusters")
    #     for label, size in largest_clusters:
    #         print(f"  {label}: {size}")

    labeled_path = DEFAULT_OUTPUT_DIR / "100k_labeled_records.csv"
    result["labeled_records"].to_pandas().to_csv(labeled_path, index=False)
    print(f"\nWrote labeled records to: {labeled_path}")
    # pdb.set_trace()
    linker.generate_dashboard(
        fields=["BorrowerName", "BorrowerAddress", "BorrowerCity"],
        output_path=str(DEFAULT_OUTPUT_DIR / "dashboard.html"),
        top_n_clusters=50,
        min_cluster_size=2,
        label_field="BorrowerName",
    )
    print(f"Wrote dashboard to: {DEFAULT_OUTPUT_DIR / 'dashboard.html'}")

    # pdb.set_trace()
    return {"df": df, "linker": linker, "result": result}


if __name__ == "__main__":
    main()
