import tempfile
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from biotite.structure import concatenate
from biotite.structure.io.pdb import PDBFile

from mDeepFRI.alignment import AlignmentResult
from mDeepFRI.bio_utils import (ONE_TO_THREE, build_target_to_query_map,
                                carve_aligned_pdb, chain_id_from_filename,
                                compress_carved_structures,
                                extract_residues_coordinates,
                                get_residue_atom_groups, load_structure,
                                prefetch_template_structures,
                                resolve_structure_chain,
                                write_carved_pdbs, _get_template_for_carving)


class TestBuildTargetToQueryMap(unittest.TestCase):
    def test_identity_alignment(self):
        result = build_target_to_query_map("AB", "AB")
        self.assertEqual(result, [0, 1])

    def test_gap_in_query_deletion(self):
        result = build_target_to_query_map("A-C", "ABC")
        self.assertEqual(result, [0, -1, 1])

    def test_gap_in_target_insertion(self):
        result = build_target_to_query_map("ABC", "A-C")
        self.assertEqual(result, [0, 2])


class TestChainResolution(unittest.TestCase):
    def test_chain_id_from_filename_uses_last_suffix_character(self):
        self.assertEqual(chain_id_from_filename("5aa0_BZ.pdb"), "Z")
        self.assertEqual(chain_id_from_filename("6sxu_BBB.pdb"), "B")
        self.assertEqual(chain_id_from_filename("8cd1_Le.pdb"), "e")
        self.assertEqual(chain_id_from_filename("8rdw_L6.pdb"), "6")
        self.assertEqual(chain_id_from_filename("8ckb_A001.pdb"), "1")
        self.assertEqual(chain_id_from_filename("5ibb_21.pdb"), "1")

    def test_resolve_structure_chain_from_user_pdb(self):
        repo_root = Path(__file__).resolve().parents[2]
        structure_path = (repo_root / "data/mapped_structures_10k/structures"
                          / "5aa0_BZ.pdb")
        if not structure_path.exists():
            self.skipTest("10k mapped structure sample not available.")
        structure = load_structure(structure_path.read_text(encoding="utf-8"),
                                   filetype="pdb")
        chain = resolve_structure_chain(structure, str(structure_path), "A")
        self.assertEqual(chain, "Z")


class TestCarveAlignedPdb(unittest.TestCase):
    def setUp(self):
        self.data_dir = Path(__file__).parent / "data"
        self.structure_path = self.data_dir / "AF-A0A3B4WVX2-F1-model_v6.cif"
        full_structure_string = self.structure_path.read_text(encoding="utf-8")
        structure = load_structure(full_structure_string, filetype="mmcif")
        self.template_groups = get_residue_atom_groups(structure, chain="A")
        self.full_target_sequence, _ = extract_residues_coordinates(
            full_structure_string, chain="A", filetype="mmcif")

    def _mini_structure_string(self, num_residues: int) -> tuple[str, str]:
        groups = self.template_groups[:num_residues]
        mini_structure = concatenate(groups)
        pdb_file = PDBFile()
        pdb_file.set_structure(mini_structure)
        buffer = StringIO()
        pdb_file.write(buffer)
        mini_pdb = buffer.getvalue()
        target_sequence = self.full_target_sequence[:num_residues]
        return mini_pdb, target_sequence

    def _atom_lines(self, pdb_content: str) -> str:
        atom_lines = [
            line for line in pdb_content.splitlines()
            if line.startswith(("ATOM", "HETATM"))
        ]
        return "\n".join(atom_lines)

    def _make_alignment(self,
                        query_sequence: str,
                        target_sequence: str,
                        gapped_query: str,
                        gapped_target: str,
                        structure_path: str,
                        alignment_string: str = "M") -> AlignmentResult:
        alignment = AlignmentResult(
            query_name="test_query",
            query_sequence=query_sequence,
            target_name="mini_template",
            target_sequence=target_sequence,
            alignment=alignment_string,
            query_identity=1.0,
            query_coverage=1.0,
            target_coverage=1.0,
            db_name="custom_mapping",
            structure_path=structure_path,
        )
        alignment.gapped_sequence = gapped_query
        alignment.gapped_target = gapped_target
        return alignment

    def test_carve_identity_prefix_maps_coordinates_and_numbering(self):
        prefix_len = 5
        mini_pdb, target_sequence = self._mini_structure_string(prefix_len)
        query_sequence = target_sequence
        alignment = self._make_alignment(
            query_sequence=query_sequence,
            target_sequence=target_sequence,
            gapped_query=query_sequence,
            gapped_target=target_sequence,
            structure_path=str(self.structure_path),
            alignment_string="M" * prefix_len,
        )

        pdb_content = carve_aligned_pdb(alignment,
                                        mini_pdb,
                                        filetype="pdb",
                                        chain="A")
        self.assertIn("REMARK   1 CARVED BY MDEEPFRI", pdb_content)
        self.assertIn("REMARK   1 TEMPLATE STRUCTURE: mini_template",
                      pdb_content)
        self.assertIn("REMARK   1 QUERY SEQUENCE: test_query", pdb_content)
        self.assertIn("RESIDUE IDENTITY MAY NOT MATCH ATOM GEOMETRY",
                      pdb_content)
        self.assertIn("SEQRES", pdb_content)

        template_groups = self.template_groups[:prefix_len]
        carved_atoms = PDBFile.read(
            StringIO(self._atom_lines(pdb_content))).get_structure()[0]

        self.assertEqual(len(carved_atoms),
                         sum(len(group) for group in template_groups))
        for query_idx in range(prefix_len):
            expected_res_id = query_idx + 1
            carved_residue = carved_atoms[carved_atoms.res_id == expected_res_id]
            template_residue = template_groups[query_idx]
            np.testing.assert_array_almost_equal(carved_residue.coord,
                                                 template_residue.coord)
            expected_name = query_sequence[query_idx]
            self.assertTrue(
                np.all(carved_residue.res_name == ONE_TO_THREE[expected_name]))

    def test_query_insertion_is_seqres_only(self):
        mini_pdb, target_sequence = self._mini_structure_string(4)
        query_sequence = target_sequence[:2] + "X" + target_sequence[2:]
        gapped_query = query_sequence
        gapped_target = target_sequence[:2] + "-" + target_sequence[2:]
        alignment = self._make_alignment(
            query_sequence=query_sequence,
            target_sequence=target_sequence,
            gapped_query=gapped_query,
            gapped_target=gapped_target,
            structure_path=str(self.structure_path),
            alignment_string="M" * 5,
        )

        pdb_content = carve_aligned_pdb(alignment,
                                        mini_pdb,
                                        filetype="pdb",
                                        chain="A")

        self.assertIn("SEQRES   1 A    5", pdb_content)
        self.assertIn("UNK", pdb_content)
        self.assertIn("QUERY INSERTIONS", pdb_content)

        carved_atoms = PDBFile.read(
            StringIO(self._atom_lines(pdb_content))).get_structure()[0]
        self.assertNotIn(3, carved_atoms.res_id)
        self.assertEqual(sorted(np.unique(carved_atoms.res_id)), [1, 2, 4, 5])


class TestPrefetchAndParallelCarve(unittest.TestCase):
    def setUp(self):
        self.data_dir = Path(__file__).parent / "data"
        self.structure_path = self.data_dir / "AF-A0A3B4WVX2-F1-model_v6.cif"
        self.structure_string = self.structure_path.read_text(encoding="utf-8")

    def test_prefetch_custom_mapping_uses_cached_structure_string(self):
        alignment = AlignmentResult(
            query_name="q1",
            query_sequence="MFSK",
            target_name="template",
            target_sequence="MFSK",
            alignment="MMMM",
            db_name="custom_mapping",
            structure_path=str(self.structure_path),
            structure_string=self.structure_string,
        )
        cache = prefetch_template_structures([alignment], ())
        self.assertIn(f"custom:{self.structure_path}", cache)
        self.assertEqual(cache[f"custom:{self.structure_path}"][0],
                         self.structure_string)

    @patch("mDeepFRI.bio_utils.ProcessPoolExecutor")
    def test_write_carved_pdbs_uses_thread_pool(self, mock_executor):
        mock_executor_instance = mock_executor.return_value
        mock_executor_instance.__enter__.return_value = mock_executor_instance
        mock_executor_instance.map.side_effect = lambda func, shards: [
            [("q1", None), ("q2", None)][:len(shard)] for shard in shards
        ]

        alignments = []
        for index in range(4):
            alignment = AlignmentResult(
                query_name=f"q{index}",
                query_sequence="MFSK",
                target_name="template",
                target_sequence="MFSK",
                alignment="MMMM",
                db_name="custom_mapping",
                structure_path=str(self.structure_path),
            )
            alignment.gapped_sequence = "MFSK"
            alignment.gapped_target = "MFSK"
            alignments.append(alignment)
        aligned_cmaps = [(alignment, np.zeros((4, 4))) for alignment in alignments]

        with tempfile.TemporaryDirectory() as temp_dir:
            carve_dir = Path(temp_dir)
            count = write_carved_pdbs(aligned_cmaps, (), carve_dir, threads=4)
            self.assertEqual(count, 4)
            mock_executor.assert_called_once()
            call_kwargs = mock_executor.call_args.kwargs
            self.assertEqual(call_kwargs["max_workers"], 4)
            shards = mock_executor_instance.map.call_args[0][1]
            self.assertEqual(len(shards), 4)
            self.assertEqual(shards[0][0].query_name, "q0")
            self.assertEqual(shards[0][0].structure_path, str(self.structure_path))

    def test_get_template_for_carving_reads_from_disk(self):
        alignment = AlignmentResult(
            query_name="q1",
            query_sequence="MFSK",
            target_name="template",
            target_sequence="MFSK",
            alignment="MMMM",
            db_name="custom_mapping",
            structure_path=str(self.structure_path),
        )
        structure_string, filetype, chain = _get_template_for_carving(
            alignment, ())
        self.assertEqual(filetype, "mmcif")
        self.assertEqual(chain, "A")
        self.assertGreater(len(structure_string), 1000)
        self.assertEqual(structure_string, self.structure_string)

    @patch("mDeepFRI.bio_utils.FOLDCOMP_PATH")
    @patch("mDeepFRI.bio_utils.subprocess.run")
    def test_compress_carved_structures_invokes_foldcomp(
            self, mock_run, mock_foldcomp_path):
        mock_foldcomp_path.exists.return_value = True
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            carve_dir = temp_path / "carved_pdbs"
            carve_dir.mkdir()
            (carve_dir / "q1.pdb").write_text("ATOM\n", encoding="utf-8")
            (carve_dir / "q2.pdb").write_text("ATOM\n", encoding="utf-8")
            compress_carved_structures(carve_dir,
                                       temp_path / "carved_pdbs.foldcomp",
                                       threads=3)
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        self.assertIn("compress", args)
        self.assertIn("-d", args)
        self.assertIn("-t", args)
        self.assertIn("3", args)
        self.assertFalse(carve_dir.exists())

    @patch("mDeepFRI.bio_utils.FOLDCOMP_PATH")
    def test_compress_carved_structures_requires_binary(self,
                                                        mock_foldcomp_path):
        mock_foldcomp_path.exists.return_value = False
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            carve_dir = temp_path / "carved_pdbs"
            carve_dir.mkdir()
            (carve_dir / "q1.pdb").write_text("ATOM\n", encoding="utf-8")
            with self.assertRaises(FileNotFoundError):
                compress_carved_structures(
                    carve_dir, temp_path / "carved_pdbs.foldcomp")


if __name__ == "__main__":
    unittest.main()
