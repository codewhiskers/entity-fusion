"""Run from the repository root: python examples/transitive_review.py [report.html]."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import polars as pl
from transitive_fusion import cluster_entities


rule = {
    "or": [
        {"field": "ssn"},
        {
            "at_least": 2,
            "conditions": [
                {"field": "driver_license", "block_on": ["license_state"]},
                {"field": "phone", "block_on": ["country"]},
                {"field": "email"},
            ],
        },
    ],
}

# Synthetic IDs: A-B share SSN; B-C share license/phone; C-D share
# phone/email. Different SSNs and a common phone make this worth reviewing.
records = pl.DataFrame({
    "record_id": ["A", "B", "C", "D", "E"],
    "ssn": ["SYNTH-1", "SYNTH-1", "SYNTH-2", "SYNTH-3", None],
    "driver_license": ["L1", "L2", "L2", "L4", None],
    "license_state": ["NY", "NY", "NY", "CA", None],
    "phone": ["555-0101", "555-0102", "555-0102", "555-0102", None],
    "country": ["US"] * 5,
    "email": ["a@example.test", "b@example.test", "c@example.test", "c@example.test", None],
})

if __name__ == "__main__":
    result = cluster_entities(records, match_rule=rule, return_details=True)
    destination = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("transitive_review.html")
    result.export_html(destination, record_id_column="record_id", conflict_columns=["ssn"],
                       common_value_min_count=3)
    print(result.records)
    print(f"Review report: {destination.resolve()}")
