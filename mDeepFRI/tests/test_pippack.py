import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mDeepFRI.pippack import (PippackConfigError, _count_ca_residues,
                              pack_carved_structures, reinsert_nonstandard_backbone,
                              resolve_pippack_dir, stash_nonstandard_backbone)


class TestPippackConfig(unittest.TestCase):
    def test_resolve_pippack_dir_requires_path(self):
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(PippackConfigError):
                resolve_pippack_dir(None)

    def test_resolve_pippack_dir_validates_layout(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with self.assertRaises(PippackConfigError):
                resolve_pippack_dir(str(root))
            (root / "inference.py").write_text("# stub\n", encoding="utf-8")
            with self.assertRaises(PippackConfigError):
                resolve_pippack_dir(str(root))
            (root / "model_weights").mkdir()
            resolved = resolve_pippack_dir(str(root))
            self.assertEqual(resolved, root.resolve())


class TestNonstandardReinsertion(unittest.TestCase):
    def _ala_unk_gly_pdb(self) -> str:
        return "\n".join([
            "REMARK   1 CARVED BY MDEEPFRI",
            "SEQRES   1 A    3  ALA UNK GLY",
            "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N",
            "ATOM      2  CA  ALA A   1       1.000   0.000   0.000  1.00  0.00           C",
            "ATOM      3  C   ALA A   1       2.000   0.000   0.000  1.00  0.00           C",
            "ATOM      4  O   ALA A   1       3.000   0.000   0.000  1.00  0.00           O",
            "ATOM      5  N   UNK A   2      10.000   0.000   0.000  1.00  0.00           N",
            "ATOM      6  CA  UNK A   2      11.000   0.000   0.000  1.00  0.00           C",
            "ATOM      7  C   UNK A   2      12.000   0.000   0.000  1.00  0.00           C",
            "ATOM      8  O   UNK A   2      13.000   0.000   0.000  1.00  0.00           O",
            "ATOM      9  N   GLY A   3      20.000   0.000   0.000  1.00  0.00           N",
            "ATOM     10  CA  GLY A   3      21.000   0.000   0.000  1.00  0.00           C",
            "ATOM     11  C   GLY A   3      22.000   0.000   0.000  1.00  0.00           C",
            "ATOM     12  O   GLY A   3      23.000   0.000   0.000  1.00  0.00           O",
            "END",
        ]) + "\n"

    def test_stash_counts_unk_sec_pyl(self):
        pdb = self._ala_unk_gly_pdb().replace("UNK A   2", "SEC A   2")
        stashed, counts = stash_nonstandard_backbone(pdb)
        self.assertEqual(counts["SEC"], 1)
        self.assertEqual(counts["UNK"], 0)
        self.assertEqual(len(stashed), 1)
        self.assertEqual(len(stashed[("A", 2)]), 4)

    def test_reinsert_restores_unk_between_packed_residues(self):
        carved = self._ala_unk_gly_pdb()
        stashed, counts = stash_nonstandard_backbone(carved)
        self.assertEqual(counts["UNK"], 1)

        # Simulate PIPPack dropping UNK and adding a side-chain atom on ALA.
        # PIPPack's to_pdb wraps atoms in MODEL / ENDMDL / END.
        packed = "\n".join([
            "MODEL     1",
            "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N",
            "ATOM      2  CA  ALA A   1       1.000   0.000   0.000  1.00  0.00           C",
            "ATOM      3  C   ALA A   1       2.000   0.000   0.000  1.00  0.00           C",
            "ATOM      4  O   ALA A   1       3.000   0.000   0.000  1.00  0.00           O",
            "ATOM      5  CB  ALA A   1       1.500  -1.000   0.000  1.00  0.00           C",
            "ATOM      6  N   GLY A   3      20.000   0.000   0.000  1.00  0.00           N",
            "ATOM      7  CA  GLY A   3      21.000   0.000   0.000  1.00  0.00           C",
            "ATOM      8  C   GLY A   3      22.000   0.000   0.000  1.00  0.00           C",
            "ATOM      9  O   GLY A   3      23.000   0.000   0.000  1.00  0.00           O",
            "TER      10      GLY A   3",
            "ENDMDL",
            "END",
        ]) + "\n"

        merged = reinsert_nonstandard_backbone(packed, stashed)
        self.assertEqual(_count_ca_residues(merged), 3)
        self.assertIn("UNK", merged)
        self.assertIn("CB", merged)

        # UNK backbone should sit between ALA (res 1) and GLY (res 3).
        ca_order = []
        for line in merged.splitlines():
            if line.startswith(("ATOM", "HETATM")) and line[12:16].strip() == "CA":
                ca_order.append(line[17:20].strip())
        self.assertEqual(ca_order, ["ALA", "UNK", "GLY"])

        # Valid PDB order: MODEL before atoms; TER / ENDMDL / END after.
        records = [
            line[:6].strip() for line in merged.splitlines() if line.strip()
        ]
        self.assertEqual(records[0], "MODEL")
        self.assertNotIn("MODEL", records[1:])
        first_atom = next(i for i, r in enumerate(records) if r in ("ATOM", "HETATM"))
        last_atom = max(i for i, r in enumerate(records) if r in ("ATOM", "HETATM"))
        self.assertLess(records.index("MODEL"), first_atom)
        self.assertGreater(records.index("TER"), last_atom)
        self.assertGreater(records.index("ENDMDL"), records.index("TER"))
        self.assertEqual(records[-1], "END")


class TestPackCarvedStructures(unittest.TestCase):
    def _write_backbone_pdb(self, path: Path, res_id: int = 1) -> None:
        # Minimal complete backbone residue.
        lines = [
            "REMARK   1 CARVED BY MDEEPFRI",
            "SEQRES   1 A    1  ALA",
            f"ATOM      1  N   ALA A{res_id:4d}       0.000   0.000   0.000  1.00  0.00           N",
            f"ATOM      2  CA  ALA A{res_id:4d}       1.458   0.000   0.000  1.00  0.00           C",
            f"ATOM      3  C   ALA A{res_id:4d}       2.009   1.420   0.000  1.00  0.00           C",
            f"ATOM      4  O   ALA A{res_id:4d}       1.251   2.390   0.000  1.00  0.00           O",
            "END",
        ]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _write_ca_only_pdb(self, path: Path) -> None:
        lines = [
            "REMARK   1 CARVED BY MDEEPFRI",
            "SEQRES   1 A    1  ALA",
            "ATOM      1  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C",
            "END",
        ]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _write_ala_unk_gly_pdb(self, path: Path) -> None:
        path.write_text(
            "\n".join([
                "REMARK   1 CARVED BY MDEEPFRI",
                "SEQRES   1 A    3  ALA UNK GLY",
                "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N",
                "ATOM      2  CA  ALA A   1       1.000   0.000   0.000  1.00  0.00           C",
                "ATOM      3  C   ALA A   1       2.000   0.000   0.000  1.00  0.00           C",
                "ATOM      4  O   ALA A   1       3.000   0.000   0.000  1.00  0.00           O",
                "ATOM      5  N   UNK A   2      10.000   0.000   0.000  1.00  0.00           N",
                "ATOM      6  CA  UNK A   2      11.000   0.000   0.000  1.00  0.00           C",
                "ATOM      7  C   UNK A   2      12.000   0.000   0.000  1.00  0.00           C",
                "ATOM      8  O   UNK A   2      13.000   0.000   0.000  1.00  0.00           O",
                "ATOM      9  N   GLY A   3      20.000   0.000   0.000  1.00  0.00           N",
                "ATOM     10  CA  GLY A   3      21.000   0.000   0.000  1.00  0.00           C",
                "ATOM     11  C   GLY A   3      22.000   0.000   0.000  1.00  0.00           C",
                "ATOM     12  O   GLY A   3      23.000   0.000   0.000  1.00  0.00           O",
                "END",
            ]) + "\n",
            encoding="utf-8",
        )

    def _make_pippack_stub(self, root: Path) -> None:
        root.mkdir()
        (root / "inference.py").write_text("# stub\n", encoding="utf-8")
        weights = root / "model_weights"
        weights.mkdir()
        (weights / "pippack_model_1_ckpt.pt").write_bytes(b"x")
        (weights / "pippack_model_1_config.pickle").write_bytes(b"x")

    def test_parallel_backbone_validation_matches_serial(self):
        from mDeepFRI.pippack import _run_backbone_validation

        with tempfile.TemporaryDirectory() as temp_dir:
            carve_dir = Path(temp_dir)
            good_a = carve_dir / "a_good.pdb"
            bad = carve_dir / "b_bad.pdb"
            good_c = carve_dir / "c_good.pdb"
            self._write_backbone_pdb(good_a)
            self._write_ca_only_pdb(bad)
            self._write_backbone_pdb(good_c)
            pdb_files = sorted(carve_dir.glob("*.pdb"))

            serial = _run_backbone_validation(pdb_files, threads=1)
            parallel = _run_backbone_validation(pdb_files, threads=3)

            self.assertEqual([p.name for p in serial[0]],
                             [p.name for p in parallel[0]])
            self.assertEqual(serial[4], parallel[4])
            self.assertEqual(set(serial[1]), set(parallel[1]))
            self.assertEqual(serial[4], 1)
            self.assertEqual(len(serial[0]), 2)

    @patch("mDeepFRI.pippack.subprocess.run")
    @patch("mDeepFRI.pippack._run_worker")
    def test_pack_skips_incomplete_and_reattaches_header(self, mock_run_worker,
                                                         mock_subprocess_run):
        # Torch import probe succeeds.
        mock_subprocess_run.return_value.returncode = 0
        mock_subprocess_run.return_value.stdout = "2.0.1\n"
        mock_subprocess_run.return_value.stderr = ""

        with tempfile.TemporaryDirectory() as temp_dir:
            carve_dir = Path(temp_dir) / "carved_pdbs"
            carve_dir.mkdir()
            good = carve_dir / "good.pdb"
            bad = carve_dir / "bad.pdb"
            self._write_backbone_pdb(good)
            self._write_ca_only_pdb(bad)

            pippack_root = Path(temp_dir) / "PIPPack"
            self._make_pippack_stub(pippack_root)

            def fake_worker(*, pdb_paths, packed_dir, worker_index, **kwargs):
                for pdb_path in pdb_paths:
                    # Simulated PIPPack ATOM-only output (header stripped).
                    packed = (
                        "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N\n"
                        "ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C\n"
                        "ATOM      3  C   ALA A   1       2.009   1.420   0.000  1.00  0.00           C\n"
                        "ATOM      4  O   ALA A   1       1.251   2.390   0.000  1.00  0.00           O\n"
                        "ATOM      5  CB  ALA A   1       1.989  -0.773  -1.207  1.00  0.00           C\n"
                        "END\n")
                    (packed_dir / f"{pdb_path.stem}.pdb").write_text(
                        packed, encoding="utf-8")
                return 0, []

            mock_run_worker.side_effect = fake_worker

            packed, skipped = pack_carved_structures(
                carve_dir,
                pippack_dir=str(pippack_root),
                device="cpu",
                threads=2,
                pippack_workers=1,
            )
            self.assertEqual(packed, 1)
            self.assertEqual(skipped, 1)
            mock_run_worker.assert_called_once()

            good_text = good.read_text(encoding="utf-8")
            self.assertIn("REMARK   1 CARVED BY MDEEPFRI", good_text)
            self.assertIn("SEQRES", good_text)
            self.assertIn("CB", good_text)
            bad_text = bad.read_text(encoding="utf-8")
            self.assertNotIn("CB", bad_text)

    @patch("mDeepFRI.pippack.subprocess.run")
    @patch("mDeepFRI.pippack._run_worker")
    def test_pack_reinserts_unk_after_pippack_drops_it(self, mock_run_worker,
                                                       mock_subprocess_run):
        mock_subprocess_run.return_value.returncode = 0
        mock_subprocess_run.return_value.stdout = "2.0.1\n"
        mock_subprocess_run.return_value.stderr = ""

        with tempfile.TemporaryDirectory() as temp_dir:
            carve_dir = Path(temp_dir) / "carved_pdbs"
            carve_dir.mkdir()
            query = carve_dir / "with_unk.pdb"
            self._write_ala_unk_gly_pdb(query)

            pippack_root = Path(temp_dir) / "PIPPack"
            self._make_pippack_stub(pippack_root)

            def fake_worker(*, pdb_paths, packed_dir, worker_index, **kwargs):
                for pdb_path in pdb_paths:
                    # PIPPack drops UNK (res 2); packs ALA/GLY only.
                    packed = "\n".join([
                        "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N",
                        "ATOM      2  CA  ALA A   1       1.000   0.000   0.000  1.00  0.00           C",
                        "ATOM      3  C   ALA A   1       2.000   0.000   0.000  1.00  0.00           C",
                        "ATOM      4  O   ALA A   1       3.000   0.000   0.000  1.00  0.00           O",
                        "ATOM      5  CB  ALA A   1       1.500  -1.000   0.000  1.00  0.00           C",
                        "ATOM      6  N   GLY A   3      20.000   0.000   0.000  1.00  0.00           N",
                        "ATOM      7  CA  GLY A   3      21.000   0.000   0.000  1.00  0.00           C",
                        "ATOM      8  C   GLY A   3      22.000   0.000   0.000  1.00  0.00           C",
                        "ATOM      9  O   GLY A   3      23.000   0.000   0.000  1.00  0.00           O",
                        "END",
                    ]) + "\n"
                    (packed_dir / f"{pdb_path.stem}.pdb").write_text(
                        packed, encoding="utf-8")
                return 0, []

            mock_run_worker.side_effect = fake_worker

            packed, skipped = pack_carved_structures(
                carve_dir,
                pippack_dir=str(pippack_root),
                device="cpu",
                threads=1,
                pippack_workers=1,
            )
            self.assertEqual(packed, 1)
            self.assertEqual(skipped, 0)

            result = query.read_text(encoding="utf-8")
            self.assertIn("REMARK   1 CARVED BY MDEEPFRI", result)
            self.assertIn("SEQRES", result)
            self.assertIn("CB", result)
            self.assertIn("UNK", result)
            self.assertEqual(_count_ca_residues(result), 3)

            ca_order = [
                line[17:20].strip() for line in result.splitlines()
                if line.startswith(("ATOM", "HETATM"))
                and line[12:16].strip() == "CA"
            ]
            self.assertEqual(ca_order, ["ALA", "UNK", "GLY"])


if __name__ == "__main__":
    unittest.main()
