"""Run with: python -m unittest discover -s tests."""

import math
import random
import unittest

import polars as pl

from transitive_fusion import cluster_entities


class TransitiveFusionTests(unittest.TestCase):
    def test_chain_and_singletons(self):
        df = pl.DataFrame({"ssn": ["A", "A", "B", None],
                           "license": ["X", "Y", "Y", None]})
        result = cluster_entities(df, ["ssn", "license"])
        self.assertEqual(result["cluster_id"].to_list(), [0, 0, 0, 3])
        self.assertTrue(result.drop("cluster_id").equals(df))
        self.assertNotIn("cluster_id", df.columns)

    def test_threshold_is_pairwise_then_transitive(self):
        df = pl.DataFrame({"a": [1, 1, 9, 9], "b": [2, 2, 2, 8],
                           "c": [3, 4, 4, 7]})
        self.assertEqual(cluster_entities(df, ["a", "b", "c"], min_matches=2)
                         ["cluster_id"].to_list(), [0, 0, 0, 3])

    def test_missing_values_and_column_scoping(self):
        df = pl.DataFrame({"a": [None, None, "", " ", "X", "Y"],
                           "b": [None, None, " ", "", "Y", "X"]})
        self.assertEqual(cluster_entities(df, ["a", "b"])["cluster_id"].to_list(),
                         list(range(6)))
        floats = pl.DataFrame({"a": [math.nan, math.nan, 0.0, 0.0]})
        self.assertEqual(cluster_entities(floats, ["a"])["cluster_id"].to_list(),
                         [0, 1, 2, 2])

    def test_empty_and_all_missing_duplicates(self):
        empty = pl.DataFrame(schema={"a": pl.String})
        self.assertEqual(cluster_entities(empty, ["a"]).schema["cluster_id"], pl.UInt64)
        df = pl.DataFrame({"a": [1, 1], "b": [None, None]})
        self.assertEqual(cluster_entities(df, ["a", "b"], min_matches=2)
                         ["cluster_id"].to_list(), [0, 1])

    def test_validation(self):
        df = pl.DataFrame({"a": [1]})
        for columns, kwargs in [([], {}), (["a", "a"], {}), (["missing"], {}),
                                (["a"], {"min_matches": True}),
                                (["a"], {"min_matches": 2}),
                                (["a"], {"cluster_column": "a"})]:
            with self.assertRaises(ValueError):
                cluster_entities(df, columns, **kwargs)

    def test_matches_independent_pairwise_reference(self):
        rng = random.Random(42)
        for _ in range(15):
            rows = [[rng.choice([None, "", "a", "b", "c"]) for _ in range(4)]
                    for _ in range(25)]
            df = pl.DataFrame(rows, schema=["a", "b", "c", "d"], orient="row")
            for threshold in range(1, 5):
                adjacency = [set() for _ in rows]
                for i in range(len(rows)):
                    for j in range(i):
                        count = sum(a not in (None, "") and a == b
                                    for a, b in zip(rows[i], rows[j]))
                        if count >= threshold:
                            adjacency[i].add(j)
                            adjacency[j].add(i)
                expected = []
                for i in range(len(rows)):
                    seen, stack = {i}, [i]
                    while stack:
                        for neighbor in adjacency[stack.pop()] - seen:
                            seen.add(neighbor)
                            stack.append(neighbor)
                    expected.append(min(seen))
                actual = cluster_entities(df, df.columns, min_matches=threshold)
                self.assertEqual(actual["cluster_id"].to_list(), expected)


class RuleAndReviewTests(unittest.TestCase):
    rule = {"or": [
        {"field": "ssn"},
        {"at_least": 2, "conditions": [
            {"field": "license", "block_on": ["state"]},
            {"field": "phone", "block_on": ["country"]},
            {"field": "email"},
        ]},
    ]}

    def test_nested_rules_and_evidence(self):
        df = pl.DataFrame({
            "ssn": ["s1", "s1", "s2", "s3", None],
            "license": ["a", "b", "b", "b", None],
            "state": ["NY", "NY", "NY", "CA", None],
            "phone": ["1", "2", "2", "2", None],
            "country": ["US"] * 5,
            "email": ["a", "b", "c", "d", None],
        })
        result = cluster_entities(df, match_rule=self.rule, return_details=True)
        self.assertEqual(result.records["cluster_id"].to_list(), [0, 0, 0, 3, 4])
        self.assertEqual(result.match_groups.height, 2)
        evidence = result.match_groups["shared_values"].to_list()
        self.assertTrue(any('"state": "NY"' in e and '"country": "US"' in e for e in evidence))
        self.assertTrue(any("block_on: state" in c for cs in result.match_groups["conditions"] for c in cs))

    def test_blockers_missing_and_not_extra_votes(self):
        df = pl.DataFrame({"a": ["x"] * 4, "state": ["NY", "NY", None, None],
                           "b": ["1", "2", "3", "4"]})
        rule = {"and": [{"field": "a", "block_on": ["state"]}, {"field": "b"}]}
        self.assertEqual(cluster_entities(df, match_rule=rule)["cluster_id"].to_list(), [0, 1, 2, 3])
        self.assertEqual(cluster_entities(df, match_rule={"field": "a", "block_on": ["state"]})
                         ["cluster_id"].to_list(), [0, 0, 2, 3])

    def test_rule_validation(self):
        df = pl.DataFrame({"a": [1]})
        invalid = [{"and": []}, {"and": [{"field": "a"}], "or": [{"field": "a"}]},
                   {"at_least": 0, "conditions": [{"field": "a"}]},
                   {"at_least": True, "conditions": [{"field": "a"}]},
                   {"or": [{"field": "a"}, {"field": "a"}]},
                   {"at_least": 2, "conditions": [{"field": "a"}, {"or": [{"field": "a"}]}]},
                   {"field": "a", "block_on": "state"},
                   {"field": "a", "block_on": ["missing"]}]
        for rule in invalid:
            with self.subTest(rule=rule), self.assertRaises(ValueError):
                cluster_entities(df, match_rule=rule)
        with self.assertRaises(ValueError):
            cluster_entities(df, ["a"], match_rule={"field": "a"})

    def test_nested_random_against_pairwise_reference(self):
        rng = random.Random(80)
        def evaluate(rule, left, right):
            if "field" in rule:
                return all(left[c] not in (None, "") and left[c] == right[c]
                           for c in [rule["field"], *rule.get("block_on", [])])
            if "or" in rule:
                return any(evaluate(c, left, right) for c in rule["or"])
            if "and" in rule:
                return all(evaluate(c, left, right) for c in rule["and"])
            return sum(evaluate(c, left, right) for c in rule["conditions"]) >= rule["at_least"]
        for _ in range(12):
            rows = [{c: rng.choice([None, "", "x", "y", "z"]) for c in
                     ["ssn", "license", "state", "phone", "country", "email"]} for _ in range(30)]
            for rule in [self.rule, {"and": [self.rule, {"field": "country"}]}]:
                labels = list(range(len(rows)))
                for i in range(len(rows)):
                    for j in range(i):
                        if evaluate(rule, rows[i], rows[j]):
                            old, new = max(labels[i], labels[j]), min(labels[i], labels[j])
                            labels = [new if label == old else label for label in labels]
                result = cluster_entities(pl.DataFrame(rows), match_rule=rule)
                self.assertEqual(result["cluster_id"].to_list(), labels)

    def test_review_export_and_complete_bridge_graph(self):
        import json
        import re
        import tempfile
        from pathlib import Path
        from transitive_fusion import _bridges
        malicious = "</script><script>alert(1)</script>"
        df = pl.DataFrame({"ssn": [malicious, malicious, "y", None],
                           "phone": ["a", "b", "b", None],
                           "name": ["</script><script>alert(1)</script>", "B", "C", "D"]})
        result = cluster_entities(df, ["ssn", "phone"], return_details=True)
        with tempfile.TemporaryDirectory() as tmp:
            path = result.export_html(Path(tmp) / "review.html", common_value_min_count=2)
            html = path.read_text()
            self.assertIn("ssn: 2 distinct values", html)
            self.assertNotIn("</script><script>alert(1)</script>", html)
            payload = json.loads(re.search(r'id="review-data">(.*?)</script>', html, re.S)[1])
            self.assertEqual(payload[0]["bridges"], [[0, 1], [1, 2]])
            self.assertEqual(payload[0]["edges"], [[0, 1], [1, 2]])
            result.export_html(path, max_records=2, max_groups=1)
            sampled = path.read_text()
            payload = json.loads(re.search(r'id="review-data">(.*?)</script>', sampled, re.S)[1])
            self.assertEqual(payload[0]["edges"], [])
            self.assertEqual(len(payload[0]["groups"]), 2)  # full evidence for path lookup
            self.assertIn("showing first 2 of 3 records", sampled)
            self.assertIn("ssn: 2 distinct values", sampled)
        self.assertEqual(_bridges({0: {1, 2}, 1: {0, 2}, 2: {0, 1}}), [])

    def test_empty_review_and_custom_cluster_name(self):
        import tempfile
        from pathlib import Path
        df = pl.DataFrame(schema={"a": pl.String})
        result = cluster_entities(df, ["a"], cluster_column="entity", return_details=True)
        self.assertEqual(result.match_groups.height, 0)
        with tempfile.TemporaryDirectory() as tmp:
            path = result.export_html(Path(tmp) / "empty.html")
            self.assertIn("No clusters meet", path.read_text())


if __name__ == "__main__":
    unittest.main()
