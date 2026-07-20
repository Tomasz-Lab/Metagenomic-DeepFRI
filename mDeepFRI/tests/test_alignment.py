import unittest

import pyopal

from mDeepFRI.alignment import align_pairwise, best_hit_database, insert_gaps


class TestAlignment(unittest.TestCase):
    def setUp(self) -> None:
        self.queries = dict(
            query_seq="MAGFLKVVQLLAKYGSKAVQWAWANKGKILDWLNAGQAIDWVVS")
        self.targets = dict(
            seq1="MESILDLQELETSEEESALMAASTVSNNC",
            seq2="MKKAVIVENKGCATCSIGAACLVDGPIPDFEIAGATGLFGLWG",
            seq3="MAGFLKVVQILAKYGSKAVQWAWANKGKILDWINAGQAIDWVVE",
            seq4="MAGFLKVVQILAKYGSKAVQWAWANKGKILDWINAGQAIDWVVE",
        )
        self.aligner = pyopal.Aligner()

        return super().setUp()

    def test_best_hit_database(self):

        best_hit, _ = best_hit_database(self.queries["query_seq"],
                                        self.targets)
        self.assertEqual(best_hit, "seq3")

    def test_align_pairwise(self):
        alignment, iden, query_coverage, target_coverage, score = align_pairwise(
            self.queries["query_seq"], self.targets["seq3"])
        self.assertEqual(alignment,
                         "MMMMMMMMMXMMMMMMMMMMMMMMMMMMMMMMXMMMMMMMMMMX")
        self.assertAlmostEqual(iden, 0.93, places=2)
        self.assertAlmostEqual(query_coverage, 1.0, places=2)
        self.assertGreater(score, 0)


class TestInsertGaps(unittest.TestCase):
    def test_deletion(self):
        self.assertEqual(insert_gaps('AACT', 'AAT', 'MMDM'), ('AACT', 'AA-T'))

    def test_insertion(self):
        self.assertEqual(insert_gaps('AAT', 'AATC', 'MMMI'), ('AAT-', 'AATC'))

    def test_unaligned(self):
        self.assertEqual(insert_gaps('AAT', 'FGTC', 'XXMI'), ('AAT-', 'FGTC'))
