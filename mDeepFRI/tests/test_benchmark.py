import unittest

import numpy as np

from mDeepFRI.benchmark import is_self_hit, select_hits


def make_hits(*rows):
    return np.rec.fromrecords(rows,
                              names=["query", "target", "fident", "bits"])


class TestIsSelfHit(unittest.TestCase):
    def test_self_hits(self):
        self.assertTrue(is_self_hit("1abc_A", "1abc_A"))
        self.assertTrue(is_self_hit("P12345", "AF-P12345-F1-model_v4"))
        self.assertTrue(
            is_self_hit("AF-P12345-F1-model_v6", "AF-P12345-F1-model_v4"))

    def test_other_hits(self):
        self.assertFalse(is_self_hit("P1234", "AF-P12345-F1-model_v4"))
        self.assertFalse(is_self_hit("1abc_A", "1abc_B"))


class TestSelectHits(unittest.TestCase):
    def test_identity_bin_bounds(self):
        hits = make_hits(("q1", "t1", 0.5, 100), ("q2", "t2", 0.6, 100),
                         ("q3", "t3", 0.49, 100))
        selected = select_hits(hits, (0.5, 0.6))
        self.assertEqual(selected["query"].tolist(), ["q1"])

    def test_bitscore_ties_broken_by_identity(self):
        hits = make_hits(("q1", "t1", 0.5, 100), ("q1", "t2", 0.4, 120),
                         ("q1", "t3", 0.45, 120))
        selected = select_hits(hits, (0.0, 1.01))
        self.assertEqual(selected["target"].tolist(), ["t3"])

    def test_self_hits(self):
        hits = make_hits(("AF-P1-F1-model_v6", "AF-P1-F1-model_v4", 1.0, 500),
                         ("AF-P1-F1-model_v6", "AF-P2-F1-model_v4", 0.6, 200))
        dropped = select_hits(hits, (0.0, 1.01))
        kept = select_hits(hits, (0.0, 1.01), drop_self_hits=False)
        self.assertEqual(dropped["target"].tolist(), ["AF-P2-F1-model_v4"])
        self.assertEqual(kept["target"].tolist(), ["AF-P1-F1-model_v4"])

    def test_random_selection_draws_per_query(self):
        hits = make_hits(("q2", "a", 0.5, 1), ("q1", "b", 0.5, 1),
                         ("q2", "c", 0.5, 1), ("q1", "d", 0.5, 1),
                         ("q2", "e", 0.5, 1), ("q3", "f", 0.9, 1))
        rng = np.random.default_rng(7)
        expected = []
        for query in ["q1", "q2", "q3"]:
            targets = [hit.target for hit in hits if hit.query == query]
            expected.append(targets[rng.integers(0, len(targets))])

        selected = select_hits(hits, (0.0, 1.01), selection="random", seed=7)
        self.assertEqual(selected["query"].tolist(), ["q1", "q2", "q3"])
        self.assertEqual(selected["target"].tolist(), expected)

    def test_empty_bin(self):
        hits = make_hits(("q1", "t1", 0.5, 100))
        for selection in ["bitscore", "random"]:
            selected = select_hits(hits, (0.9, 1.01), selection=selection)
            self.assertEqual(len(selected), 0)


if __name__ == "__main__":
    unittest.main()
