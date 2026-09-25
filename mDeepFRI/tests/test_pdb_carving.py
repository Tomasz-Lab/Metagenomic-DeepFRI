import tempfile
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from biotite.structure import concatenate
from biotite.structure.io.pdb import PDBFile

from mDeepFRI.alignment import AlignmentResult
from mDeepFRI.bio_utils import (ONE_TO_THREE, BACKBONE_ATOM_NAMES,
                                B_FACTOR_MAX, build_target_to_query_map,
                                carve_aligned_pdb, chain_id_from_filename,
                                compress_carved_structures,
                                extract_pdb_header, extract_residues_coordinates,
                                get_residue_atom_groups, load_structure,
                                prefetch_template_structures,
                                reattach_pdb_header, resolve_structure_chain,
                                validate_backbone_pdb, write_carved_pdbs,
                                _get_template_for_carving, _sanitise_b_factor)


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


class TestQueryInsertionSummary(unittest.TestCase):
    def test_summarize_insertion_runs(self):
        from mDeepFRI.bio_utils import summarize_query_insertions

        # Two runs: length 1 (B) and length 2 (EF); D is matched.
        stats = summarize_query_insertions("ABCDEF", "A-CD--")
        self.assertEqual(stats.query_insertion_runs, 2)
        self.assertEqual(stats.query_insertion_residues, 3)
        self.assertAlmostEqual(stats.mean_query_insertion_length, 1.5)
        self.assertEqual(stats.max_query_insertion_length, 2)

    def test_summarize_no_insertions(self):
        from mDeepFRI.bio_utils import summarize_query_insertions

        stats = summarize_query_insertions("AC", "AC")
        self.assertEqual(stats.query_insertion_runs, 0)
        self.assertEqual(stats.query_insertion_residues, 0)
        self.assertEqual(stats.mean_query_insertion_length, 0.0)
        self.assertEqual(stats.max_query_insertion_length, 0)

    def test_write_query_insertion_summary_tsv(self):
        from mDeepFRI.bio_utils import (QUERY_INSERTION_HEADER,
                                        write_query_insertion_summary)

        alignment = AlignmentResult(
            query_name="q1",
            query_sequence="ABCDEF",
            target_name="t1",
            target_sequence="ACD",
            alignment="MDMMDD",
        )
        alignment.gapped_sequence = "ABCDEF"
        alignment.gapped_target = "A-CD--"

        with tempfile.TemporaryDirectory() as temp_dir:
            out = Path(temp_dir) / "query_insertions.tsv"
            written = write_query_insertion_summary([alignment], out)
            self.assertEqual(written, 1)
            lines = out.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(lines[0].split("\t"), QUERY_INSERTION_HEADER)
            cols = lines[1].split("\t")
            self.assertEqual(cols[0], "q1")
            self.assertEqual(cols[1], "6")
            self.assertEqual(cols[2], "t1")
            self.assertEqual(cols[3], "2")
            self.assertEqual(cols[4], "3")
            self.assertEqual(cols[5], "1.5000")
            self.assertEqual(cols[6], "2")


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
        self.assertIn("SIDE CHAINS ARE REBUILT WITH PIPPACK", pdb_content)
        self.assertIn("SEQRES", pdb_content)

        template_groups = self.template_groups[:prefix_len]
        carved_atoms = PDBFile.read(
            StringIO(self._atom_lines(pdb_content))).get_structure()[0]

        self.assertTrue(
            set(carved_atoms.atom_name.tolist()).issubset(BACKBONE_ATOM_NAMES))
        self.assertEqual(len(carved_atoms), prefix_len * 4)
        for query_idx in range(prefix_len):
            expected_res_id = query_idx + 1
            carved_residue = carved_atoms[carved_atoms.res_id == expected_res_id]
            template_residue = template_groups[query_idx]
            template_backbone = template_residue[np.isin(
                template_residue.atom_name, list(BACKBONE_ATOM_NAMES))]
            # Compare in N/CA/C/O order.
            ordered_template = []
            for atom_name in BACKBONE_ATOM_NAMES:
                ordered_template.append(
                    template_backbone[template_backbone.atom_name == atom_name])
            template_backbone = concatenate(ordered_template)
            np.testing.assert_array_almost_equal(carved_residue.coord,
                                                 template_backbone.coord)
            expected_name = query_sequence[query_idx]
            self.assertTrue(
                np.all(carved_residue.res_name == ONE_TO_THREE[expected_name]))
            self.assertEqual(sorted(carved_residue.atom_name.tolist()),
                             sorted(BACKBONE_ATOM_NAMES))

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

    def test_incomplete_backbone_residue_is_written_and_fails_validation(self):
        mini_pdb, target_sequence = self._mini_structure_string(3)
        # Drop O from the middle residue in the mini template PDB.
        lines = []
        for line in mini_pdb.splitlines():
            if line.startswith("ATOM") and int(line[22:26]) == 2:
                if line[12:16].strip() == "O":
                    continue
            lines.append(line)
        incomplete_pdb = "\n".join(lines) + "\n"

        alignment = self._make_alignment(
            query_sequence=target_sequence,
            target_sequence=target_sequence,
            gapped_query=target_sequence,
            gapped_target=target_sequence,
            structure_path=str(self.structure_path),
            alignment_string="MMM",
        )
        pdb_content = carve_aligned_pdb(alignment,
                                        incomplete_pdb,
                                        filetype="pdb",
                                        chain="A")
        carved_atoms = PDBFile.read(
            StringIO(self._atom_lines(pdb_content))).get_structure()[0]
        # All three residues are present; residue 2 lacks O.
        self.assertEqual(sorted(np.unique(carved_atoms.res_id)), [1, 2, 3])
        res2 = carved_atoms[carved_atoms.res_id == 2]
        self.assertNotIn("O", set(res2.atom_name.tolist()))
        self.assertIn("CA", set(res2.atom_name.tolist()))

        validation = validate_backbone_pdb(pdb_content)
        self.assertFalse(validation.ok)
        self.assertTrue(
            any(res_id == 2 and "O" in missing
                for res_id, missing in validation.incomplete_residues))

    def test_template_insertion_codes_are_cleared(self):
        """Template PDB icodes must not leak into carved query numbering."""
        mini_pdb, target_sequence = self._mini_structure_string(3)
        # Inject insertion codes on residue 2 atoms (column 27).
        lines = []
        for line in mini_pdb.splitlines():
            if line.startswith("ATOM") and int(line[22:26]) == 2:
                padded = line.ljust(80)
                line = f"{padded[:26]}A{padded[27:]}".rstrip()
            lines.append(line)
        icode_pdb = "\n".join(lines) + "\n"

        alignment = self._make_alignment(
            query_sequence=target_sequence,
            target_sequence=target_sequence,
            gapped_query=target_sequence,
            gapped_target=target_sequence,
            structure_path=str(self.structure_path),
            alignment_string="MMM",
        )
        pdb_content = carve_aligned_pdb(alignment,
                                        icode_pdb,
                                        filetype="pdb",
                                        chain="A")
        icodes = {
            line[26]
            for line in pdb_content.splitlines()
            if line.startswith("ATOM")
        }
        self.assertEqual(icodes, {" "})
        ca_ids = [
            int(line[22:26]) for line in pdb_content.splitlines()
            if line.startswith("ATOM") and line[12:16].strip() == "CA"
        ]
        self.assertEqual(ca_ids, [1, 2, 3])


class TestBackboneValidation(unittest.TestCase):
    def setUp(self):
        self.data_dir = Path(__file__).parent / "data"
        self.structure_path = self.data_dir / "AF-A0A3B4WVX2-F1-model_v6.cif"
        full_structure_string = self.structure_path.read_text(encoding="utf-8")
        structure = load_structure(full_structure_string, filetype="mmcif")
        self.template_groups = get_residue_atom_groups(structure, chain="A")

    def _backbone_pdb(self, num_residues: int = 3) -> str:
        groups = []
        for group in self.template_groups[:num_residues]:
            backbone = group[np.isin(group.atom_name, list(BACKBONE_ATOM_NAMES))]
            groups.append(backbone)
        structure = concatenate(groups)
        pdb_file = PDBFile()
        pdb_file.set_structure(structure)
        buffer = StringIO()
        pdb_file.write(buffer)
        return buffer.getvalue()

    def test_validate_complete_backbone(self):
        result = validate_backbone_pdb(self._backbone_pdb(3))
        self.assertTrue(result.ok)
        self.assertEqual(result.residue_count, 3)
        self.assertEqual(result.incomplete_residues, [])

    def test_validate_ca_only_residue(self):
        pdb_text = self._backbone_pdb(2)
        # Drop N/C/O from residue 1 by rewriting ATOM lines.
        lines = []
        for line in pdb_text.splitlines():
            if line.startswith("ATOM") and int(line[22:26]) == 1:
                if line[12:16].strip() != "CA":
                    continue
            lines.append(line)
        incomplete = "\n".join(lines) + "\n"
        result = validate_backbone_pdb(incomplete)
        self.assertFalse(result.ok)
        self.assertEqual(result.status, "ca_only_or_incomplete")
        self.assertTrue(any(res_id == 1 for res_id, _ in result.incomplete_residues))

    def test_reattach_pdb_header(self):
        header = "REMARK   1 TEST\nSEQRES   1 A    2  MET PHE"
        packed = "ATOM      1  N   MET A   1\nEND\n"
        merged = reattach_pdb_header(header, packed)
        self.assertTrue(merged.startswith("REMARK   1 TEST"))
        self.assertIn("SEQRES", merged)
        self.assertIn("ATOM", merged)
        self.assertEqual(extract_pdb_header(merged).splitlines()[0],
                         "REMARK   1 TEST")


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
        structure_string, filetype, chain, structure = _get_template_for_carving(
            alignment, ())
        self.assertEqual(filetype, "mmcif")
        self.assertEqual(chain, "A")
        self.assertGreater(len(structure_string), 1000)
        self.assertEqual(structure_string, self.structure_string)
        self.assertGreater(len(structure), 0)

    def test_compress_carved_structures_writes_tar_gz(self):
        import tarfile

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            carve_dir = temp_path / "carved_pdbs"
            carve_dir.mkdir()
            (carve_dir / "q1.pdb").write_text("ATOM q1\n", encoding="utf-8")
            (carve_dir / "q2.pdb").write_text("ATOM q2\n", encoding="utf-8")
            archive = temp_path / "carved_pdbs.tar.gz"
            compress_carved_structures(carve_dir, archive)

            self.assertTrue(archive.exists())
            self.assertFalse(carve_dir.exists())
            with tarfile.open(archive, "r:gz") as tar:
                names = sorted(tar.getnames())
                self.assertEqual(names, ["q1.pdb", "q2.pdb"])
                self.assertEqual(
                    tar.extractfile("q1.pdb").read().decode("utf-8"),
                    "ATOM q1\n",
                )

    def test_compress_carved_structures_skips_empty_dir(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            carve_dir = temp_path / "carved_pdbs"
            carve_dir.mkdir()
            archive = temp_path / "carved_pdbs.tar.gz"
            compress_carved_structures(carve_dir, archive)
            self.assertFalse(archive.exists())
            self.assertTrue(carve_dir.exists())


class TestBFactorPreservation(unittest.TestCase):
    """
    Biotite drops the B-factor column unless it is requested explicitly, which
    silently replaced template pLDDT with 0.00 in every carved model.
    """
    def setUp(self):
        self.data_dir = Path(__file__).parent / "data"
        self.structure_path = self.data_dir / "AF-A0A3B4WVX2-F1-model_v6.cif"
        self.structure_string = self.structure_path.read_text(encoding="utf-8")

    def test_load_structure_reads_template_plddt(self):
        structure = load_structure(self.structure_string, filetype="mmcif")
        self.assertIn("b_factor", structure.get_annotation_categories())
        # AFDB stores per-residue pLDDT in the B-factor column.
        self.assertTrue((structure.b_factor > 0).all())
        self.assertLessEqual(structure.b_factor.max(), 100.0)

    def test_load_structure_can_skip_b_factor(self):
        structure = load_structure(self.structure_string,
                                   filetype="mmcif",
                                   include_b_factor=False)
        self.assertNotIn("b_factor", structure.get_annotation_categories())

    def test_carved_pdb_keeps_template_b_factors(self):
        structure = load_structure(self.structure_string, filetype="mmcif")
        groups = get_residue_atom_groups(structure, chain="A")[:5]
        mini = concatenate(groups)
        pdb_file = PDBFile()
        pdb_file.set_structure(mini)
        buffer = StringIO()
        pdb_file.write(buffer)
        mini_pdb = buffer.getvalue()

        target_sequence, _ = extract_residues_coordinates(
            self.structure_string, chain="A", filetype="mmcif")
        target_sequence = target_sequence[:5]

        alignment = AlignmentResult(
            query_name="q",
            query_sequence=target_sequence,
            target_name="mini_template",
            target_sequence=target_sequence,
            alignment="M" * 5,
            db_name="afdb_test",
            structure_path=str(self.structure_path),
        )
        alignment.gapped_sequence = target_sequence
        alignment.gapped_target = target_sequence

        carved = carve_aligned_pdb(alignment, mini_pdb, filetype="pdb",
                                   chain="A")
        atom_lines = "\n".join(line for line in carved.splitlines()
                                if line.startswith(("ATOM", "HETATM")))
        carved_atoms = PDBFile.read(StringIO(atom_lines)).get_structure(
            extra_fields=["b_factor"])[0]

        self.assertTrue((carved_atoms.b_factor > 0).all())
        expected = mini[np.isin(mini.atom_name, list(BACKBONE_ATOM_NAMES))]
        self.assertEqual(len(carved_atoms), len(expected))
        np.testing.assert_allclose(sorted(carved_atoms.b_factor),
                                   sorted(expected.b_factor),
                                   atol=0.01)

    def test_carved_header_records_b_factor_provenance(self):
        alignment = AlignmentResult(query_name="q",
                                    query_sequence="A",
                                    target_name="t",
                                    target_sequence="A",
                                    alignment="M",
                                    db_name="afdb_test")
        alignment.gapped_sequence = "A"
        alignment.gapped_target = "A"
        from mDeepFRI.bio_utils import _format_carved_pdb_header
        header = "\n".join(_format_carved_pdb_header(alignment, chain="A"))
        self.assertIn("REMARK   1 TEMPLATE DATABASE: afdb_test", header)
        self.assertIn("B-FACTOR COLUMN HOLDS THE TEMPLATE B-FACTORS", header)

        alignment.db_name = None
        header = "\n".join(_format_carved_pdb_header(alignment, chain="A"))
        self.assertIn("TEMPLATE DATABASE: user-supplied structure file",
                      header)

    def test_missing_b_iso_column_does_not_break_carving(self):
        """An mmCIF without B_iso_or_equiv parses to NaN, and the PDB writer
        rejects NaN outright; it must come out as 0.00 instead of raising."""
        cif = "\n".join([
            "data_test",
            "loop_",
            "_atom_site.group_PDB",
            "_atom_site.id",
            "_atom_site.type_symbol",
            "_atom_site.label_atom_id",
            "_atom_site.label_comp_id",
            "_atom_site.label_asym_id",
            "_atom_site.auth_asym_id",
            "_atom_site.label_seq_id",
            "_atom_site.pdbx_PDB_ins_code",
            "_atom_site.Cartn_x",
            "_atom_site.Cartn_y",
            "_atom_site.Cartn_z",
            "_atom_site.occupancy",
            "_atom_site.pdbx_PDB_model_num",
            "ATOM 1 N N ALA A A 1 . 11.104 6.134 -6.504 1.00 1",
            "ATOM 2 C CA ALA A A 1 . 11.639 6.071 -5.147 1.00 1",
            "",
        ])
        structure = load_structure(cif, filetype="mmcif")
        self.assertTrue((structure.b_factor == 0.0).all())
        # Would raise BadStructureError if the NaN reached the writer.
        pdb_file = PDBFile()
        pdb_file.set_structure(structure)

    def test_sanitise_b_factor_clips_to_pdb_field(self):
        values = np.array([np.nan, -5.0, 1e6, 42.5], dtype=np.float32)
        result = _sanitise_b_factor(values)
        np.testing.assert_allclose(result,
                                   [0.0, 0.0, B_FACTOR_MAX, 42.5],
                                   atol=0.01)

    def test_unreadable_b_factor_column_falls_back(self):
        """A malformed B-factor column must not fail the whole carving job."""
        real = load_structure(self.structure_string,
                              filetype="mmcif",
                              include_b_factor=False)
        calls = []

        def fake_read(structure_string, filetype, extra_fields):
            calls.append(list(extra_fields))
            if extra_fields:
                raise ValueError("could not convert string to float: '?'")
            return real

        with patch("mDeepFRI.bio_utils._read_structure_model", fake_read):
            structure = load_structure(self.structure_string,
                                       filetype="mmcif")

        self.assertEqual(calls, [["b_factor"], []])
        self.assertIn("b_factor", structure.get_annotation_categories())
        self.assertTrue((structure.b_factor == 0.0).all())


if __name__ == "__main__":
    unittest.main()
