import tempfile
import unittest
from pathlib import Path

from mDeepFRI.alignment import (AlignmentResult, align_pairwise,
                                format_raw_alignment_fasta,
                                write_raw_alignments)


class TestRawAlignmentExport(unittest.TestCase):
    def test_format_matches_expected_layout(self):
        alignment = AlignmentResult(
            query_name="MGYG000004906_00001",
            query_sequence="MRKILLQVLCYLWVATLAQA",
            target_name="AF-A0A3D3MU54-F1",
            target_sequence="MRRILLQILCYVWVATLAQA",
            alignment="MMXMMMMXMMMXMMMMMMMM",
            query_identity=0.85,
            query_coverage=1.0,
            target_coverage=1.0,
            score=116,
        )
        text = format_raw_alignment_fasta(alignment)
        lines = text.strip().splitlines()
        self.assertEqual(len(lines), 5)
        self.assertEqual(
            lines[0],
            ">MGYG000004906_00001|target=AF-A0A3D3MU54-F1|"
            "identity=0.8500|coverage=1.0000|score=116",
        )
        self.assertEqual(lines[1], "MRKILLQVLCYLWVATLAQA")
        self.assertEqual(
            lines[2],
            ">AF-A0A3D3MU54-F1|query=MGYG000004906_00001|"
            "identity=0.8500|coverage=1.0000|score=116",
        )
        self.assertEqual(lines[3], "MRRILLQILCYVWVATLAQA")
        self.assertEqual(lines[4],
                         "#alignment_string: MMXMMMMXMMMXMMMMMMMM")

    def test_align_pairwise_returns_score_and_export_roundtrip(self):
        alignment_string, identity, qcov, tcov, score = align_pairwise(
            "MRKILLQVLCYLWVATLAQA",
            "MRRILLQILCYVWVATLAQA",
        )
        alignment = AlignmentResult(
            query_name="q1",
            query_sequence="MRKILLQVLCYLWVATLAQA",
            target_name="t1",
            target_sequence="MRRILLQILCYVWVATLAQA",
            alignment=alignment_string,
            query_identity=identity,
            query_coverage=qcov,
            target_coverage=tcov,
            score=score,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            out_file = Path(temp_dir) / "raw_alignments.fasta"
            written = write_raw_alignments([alignment], out_file)
            self.assertEqual(written, 1)
            content = out_file.read_text(encoding="utf-8")
            self.assertIn(f"score={int(round(score))}", content)
            self.assertIn(f"#alignment_string: {alignment_string}", content)
            self.assertIn("X", alignment_string)

    def test_write_raw_alignments_concatenates_into_one_file(self):
        alignments = [
            AlignmentResult(
                query_name="q1",
                query_sequence="AAA",
                target_name="t1",
                target_sequence="AAA",
                alignment="MMM",
                query_identity=1.0,
                query_coverage=1.0,
                target_coverage=1.0,
                score=10,
            ),
            AlignmentResult(
                query_name="q2",
                query_sequence="GGG",
                target_name="t2",
                target_sequence="GGG",
                alignment="MMM",
                query_identity=1.0,
                query_coverage=1.0,
                target_coverage=1.0,
                score=20,
            ),
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            out_file = Path(temp_dir) / "raw_alignments.fasta"
            written = write_raw_alignments(alignments, out_file)
            self.assertEqual(written, 2)
            content = out_file.read_text(encoding="utf-8")
            self.assertEqual(content.count(">q1|"), 1)
            self.assertEqual(content.count(">q2|"), 1)
            self.assertEqual(content.count("#alignment_string:"), 2)
            self.assertFalse((Path(temp_dir) / "q1.fasta").exists())


if __name__ == "__main__":
    unittest.main()
