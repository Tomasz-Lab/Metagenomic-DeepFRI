import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mDeepFRI.pippack import (PippackConfigError, pack_carved_structures,
                              resolve_pippack_dir)


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
            pippack_root.mkdir()
            (pippack_root / "inference.py").write_text("# stub\n",
                                                       encoding="utf-8")
            weights = pippack_root / "model_weights"
            weights.mkdir()
            (weights / "pippack_model_1_ckpt.pt").write_bytes(b"x")
            (weights / "pippack_model_1_config.pickle").write_bytes(b"x")

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


if __name__ == "__main__":
    unittest.main()
