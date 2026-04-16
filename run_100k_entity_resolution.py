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


def write_cluster_graph_index(
    output_dir: Path,
    graph_paths: list[Path],
    cluster_sizes: list[tuple[str, int]],
) -> Path:
    size_lookup = dict(cluster_sizes)
    rows = []
    for path in graph_paths:
        cluster_label = path.stem.removeprefix("field_network_")
        rows.append(
            "<tr>"
            f"<td>{cluster_label}</td>"
            f"<td>{size_lookup.get(cluster_label, '-')}</td>"
            f"<td><a href='{path.name}'>{path.name}</a></td>"
            "</tr>"
        )

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Cluster Graph Index</title>
  <style>
    body {{
      margin: 0;
      padding: 32px;
      background: #10161d;
      color: #e6edf3;
      font: 16px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    .wrap {{
      max-width: 900px;
      margin: 0 auto;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 30px;
    }}
    p {{
      margin: 0 0 24px;
      color: #9fb1c1;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: #16202a;
      border: 1px solid #263544;
    }}
    th, td {{
      padding: 12px 14px;
      border-bottom: 1px solid #263544;
      text-align: left;
    }}
    th {{
      color: #7dd3fc;
      background: #13212d;
    }}
    a {{
      color: #7dd3fc;
      text-decoration: none;
    }}
    a:hover {{
      text-decoration: underline;
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>Cluster Graph Index</h1>
    <p>Field-aware cluster graphs generated from <code>run_100k_entity_resolution.py</code>.</p>
    <table>
      <thead>
        <tr>
          <th>Cluster</th>
          <th>Size</th>
          <th>Graph</th>
        </tr>
      </thead>
      <tbody>
        {''.join(rows)}
      </tbody>
    </table>
  </div>
</body>
</html>
"""
    index_path = output_dir / "field_network_index.html"
    index_path.write_text(html)
    return index_path


def create_cluster_graph_bundle(
    linker: IncrementalSignalLinker,
    result: Dict[str, Any],
    output_dir: Path,
    *,
    fields: list[str],
    label_field: str,
    top_n_clusters: int = 20,
    min_cluster_size: int = 2,
) -> tuple[list[Path], Path | None]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cluster_sizes = [
        (cluster_label, size)
        for cluster_label, size in get_cluster_sizes(result)
        if size >= min_cluster_size
    ][:top_n_clusters]

    graph_paths: list[Path] = []
    for cluster_label, _size in cluster_sizes:
        output_path = output_dir / f"field_network_{cluster_label}.html"
        linker.visualize_records_with_fields(
            records=result["labeled_records"],
            final_pairs=result["final_pairs"],
            fields=fields,
            cluster_label=cluster_label,
            output_path=str(output_path),
            label_field=label_field,
        )
        graph_paths.append(output_path)

    index_path = (
        write_cluster_graph_index(output_dir, graph_paths, cluster_sizes)
        if len(graph_paths) > 1
        else None
    )
    return graph_paths, index_path


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
    dashboard_path = DEFAULT_OUTPUT_DIR / "dashboard.html"
    linker.generate_dashboard(
        fields=["BorrowerName", "BorrowerAddress", "BorrowerCity"],
        output_path=str(dashboard_path),
        top_n_clusters=50,
        min_cluster_size=2,
        label_field="BorrowerName",
    )
    print(f"Wrote dashboard to: {dashboard_path}")

    graph_paths, index_path = create_cluster_graph_bundle(
        linker=linker,
        result=result,
        output_dir=DEFAULT_OUTPUT_DIR / "network_outputs",
        fields=["BorrowerName", "BorrowerAddress", "BorrowerCity"],
        label_field="BorrowerName",
        top_n_clusters=20,
        min_cluster_size=2,
    )
    if graph_paths:
        print("\nWrote cluster graph files:")
        for path in graph_paths:
            print(path)
    if index_path is not None:
        print(f"Wrote cluster graph index to: {index_path}")

    return {
        "df": df,
        "linker": linker,
        "result": result,
        "dashboard_path": dashboard_path,
        "cluster_graph_paths": graph_paths,
        "cluster_graph_index_path": index_path,
    }


if __name__ == "__main__":
    main()
