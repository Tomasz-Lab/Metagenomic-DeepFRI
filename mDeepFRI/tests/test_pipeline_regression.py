import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from mDeepFRI.database import Database
from mDeepFRI.pipeline import (FINAL_OUTPUT_HEADER, QueryFile,
                               predict_protein_function)


class TestPipelineRegression(unittest.TestCase):
    def setUp(self):
        # Locate the small_query.faa file
        self.test_dir = Path(__file__).parent
        self.data_dir = self.test_dir / "data"
        self.query_file_path = self.data_dir / "small_query.faa"

        # Verify file exists
        if not self.query_file_path.exists():
            # Fallback for when running from a different root
            self.query_file_path = Path(
                "/nfs/cds-peta/exports/biol_micro_cds_gr_sunagawa/scratch/vbezshapkin/Metagenomic-DeepFRI/mDeepFRI/tests/data/small_query.faa"
            )

    @patch("mDeepFRI.pipeline.Pool")
    @patch("mDeepFRI.pipeline.load_deepfri_config")
    @patch("mDeepFRI.pipeline.Predictor")
    @patch("mDeepFRI.pipeline.align_mmseqs_results")
    @patch("mDeepFRI.pipeline.extract_calpha_coords")
    @patch("mDeepFRI.pipeline.build_align_contact_map")
    @patch("mDeepFRI.pipeline.get_json_values")
    def test_predict_protein_function(
        self,
        mock_get_json_values,
        mock_build_align_contact_map,
        mock_extract_calpha_coords,
        mock_align_mmseqs_results,
        mock_predictor_cls,
        mock_load_config,
        mock_pool,
    ):
        # --- Mock setup ---

        # Mock Pool to just execute the function immediately
        mock_pool_instance = mock_pool.return_value
        mock_pool_instance.__enter__.return_value = mock_pool_instance
        mock_pool_instance.map.side_effect = lambda func, iterable: [
            func(i) for i in iterable
        ]

        # 1. Mock Config
        mock_load_config.return_value = {
            "gcn": {
                "ec": "path/to/gcn_ec.onnx",
                "bp": "path/to/gcn_bp.onnx"
            },
            "cnn": {
                "ec": "path/to/cnn_ec.onnx",
                "bp": "path/to/cnn_bp.onnx"
            },
            "version": "1.1"  # Skips 'ec' mode
        }

        # 2. Mock JSON values (GO terms and names)
        # First call gets 'goterms', second call gets 'gonames' (simplified for loop)
        mock_get_json_values.side_effect = lambda path, key: (
            ["GO:001", "GO:002"] if key == "goterms" else ["Term1", "Term2"])

        # 3. Mock MMseqs Alignment Results
        # Simulating one aligned sequence and one unaligned
        mock_aln = MagicMock()
        mock_aln.query_name = "A0A3B4WVX2_Gasdermin_pore_forming"
        mock_aln.target_name = "Target1.1"
        mock_aln.db_name = "TestDB"
        mock_aln.query_identity = 0.9
        mock_aln.query_coverage = 0.8
        mock_aln.target_coverage = 0.8
        mock_aln.query_sequence = "MFSKATANFVRQIDPEGSLIHVSRVNDSQKLVPMALVVKRNRLWFWQRPKYHPTDF"  # Truncated

        mock_align_mmseqs_results.return_value = [mock_aln]

        # 4. Mock C-alpha Coords
        mock_coords = np.zeros((len(mock_aln.query_sequence), 3))
        mock_extract_calpha_coords.return_value = [mock_coords]

        # 5. Mock Contact Map Building
        mock_cmap = np.random.rand(len(mock_aln.query_sequence),
                                   len(mock_aln.query_sequence))
        # build_align_contact_map returns (AlignmentResult, ContactMap)
        mock_build_align_contact_map.return_value = (mock_aln, mock_cmap)

        # 6. Mock Predictor
        mock_predictor_instance = mock_predictor_cls.return_value
        # Prediction output is a probability vector matching GO terms length (2)
        mock_predictor_instance.forward_pass.return_value = np.array(
            [0.95, 0.05], dtype=np.float32)

        # --- Test execution ---

        # Prepare Inputs
        query_file = QueryFile(filepath=str(self.query_file_path))
        query_file.load_sequences()

        mock_db = MagicMock(spec=Database)
        mock_db.name = "TestDB"
        mock_db.mmseqs_result = "path/to/mmseqs_results.tsv"
        mock_db.sequence_db = "path/to/seq_db"

        with tempfile.TemporaryDirectory() as temp_out:
            output_path = Path(temp_out)

            # RUN PIPELINE
            predict_protein_function(
                query_file=query_file,
                databases=(mock_db, ),
                weights="path/to/weights",
                output_path=str(output_path),
                deepfri_processing_modes=[
                    "bp"
                ],  # 'ec' filtered out by 'version' 1.1 logic mocked above
                save_cmaps=True,
                save_structures=True)

            # --- Assertions ---

            # 1. Check Output Files Created
            self.assertTrue((output_path / "alignment_summary.tsv").exists())
            self.assertTrue((output_path / "results.tsv").exists())
            self.assertTrue((output_path / "contact_maps" /
                             "A0A3B4WVX2_Gasdermin_pore_forming.npy").exists())

            # 2. Check Result Content
            with open(output_path / "results.tsv", "r") as f:
                content = f.read()
                # Expect header and at least one prediction line
                self.assertIn(
                    "protein\tnetwork_type\tprediction_mode\tgo_term\tscore",
                    content)
                self.assertIn("A0A3B4WVX2_Gasdermin_pore_forming", content)
                # GCN prediction for the aligned sequence
                self.assertIn("gcn", content)

            # 3. Verify Mock Calls
            mock_align_mmseqs_results.assert_called_once()
            mock_build_align_contact_map.assert_called()
            mock_predictor_instance.forward_pass.assert_called()

    @patch("mDeepFRI.pipeline.Pool")
    @patch("mDeepFRI.pipeline.load_deepfri_config")
    @patch("mDeepFRI.pipeline.Predictor")
    @patch("mDeepFRI.pipeline.extract_residues_coordinates")
    @patch("mDeepFRI.pipeline.build_align_contact_map")
    @patch("mDeepFRI.pipeline.get_json_values")
    def test_custom_mapping_bypass(
        self,
        mock_get_json_values,
        mock_build_align_contact_map,
        mock_extract_residues_coordinates,
        mock_predictor_cls,
        mock_load_config,
        mock_pool,
    ):
        """Test bypassing database search with custom sequence-to-structure mapping.

        Uses real alignment scoring via PyOpal to test that different query sequences
        produce different alignment metrics when aligned to the same target sequence.
        """

        # --- Mock setup ---

        # Mock Pool to just execute the function immediately
        mock_pool_instance = mock_pool.return_value
        mock_pool_instance.__enter__.return_value = mock_pool_instance
        mock_pool_instance.map.side_effect = lambda func, iterable: [
            func(i) for i in iterable
        ]

        # 1. Mock Config
        mock_load_config.return_value = {
            "gcn": {
                "bp": "path/to/gcn_bp.onnx",
                "mf": "path/to/gcn_mf.onnx"
            },
            "cnn": {
                "bp": "path/to/cnn_bp.onnx",
                "mf": "path/to/cnn_mf.onnx"
            },
            "version": "1.1"
        }

        # 2. Mock JSON values (GO terms and names)
        mock_get_json_values.side_effect = lambda path, key: (
            ["GO:001", "GO:002"]
            if key == "goterms" else ["BioProcess1", "MolFunc1"])

        # 3. Create a temporary mapping file
        mapping_file_path = Path(self.data_dir) / "small_mapping.tsv"

        # 4. Mock extract_residues_coordinates
        # Return a dummy sequence and coordinates
        # Use a target sequence similar to gasdermin to allow varied alignments
        mock_target_seq = "MFSKATANFVRQIDPEGSLIHVSRVNDSQKLVPMALVVKRNRLWFWQRPKYHPTDFTLSD"
        mock_coords = np.zeros((len(mock_target_seq), 3))
        mock_extract_residues_coordinates.return_value = (mock_target_seq,
                                                          mock_coords)

        # 5. Mock Contact Map Building
        mock_cmap = np.random.rand(60, 60)  # Match target seq length

        # build_align_contact_map returns (AlignmentResult, ContactMap)
        # We need to return a properly formed tuple
        def build_cmap_side_effect(alignment_result, **kwargs):
            return (alignment_result, mock_cmap)

        mock_build_align_contact_map.side_effect = build_cmap_side_effect

        # 6. Mock Predictor
        mock_predictor_instance = mock_predictor_cls.return_value
        mock_predictor_instance.forward_pass.return_value = np.array(
            [0.92, 0.08], dtype=np.float32)

        # --- Test execution ---

        # Prepare Inputs
        query_file = QueryFile(filepath=str(self.query_file_path))
        query_file.load_sequences()

        # Important: mapping_file_path must exist for the test
        if not mapping_file_path.exists():
            self.skipTest(f"Test data file not found: {mapping_file_path}. "
                          "This test requires the small_mapping.tsv file.")

        with tempfile.TemporaryDirectory() as temp_out:
            output_path = Path(temp_out)

            # RUN PIPELINE with custom mapping
            predict_protein_function(
                query_file=query_file,
                databases=(),  # No databases needed for custom mapping
                weights="path/to/weights",
                output_path=str(output_path),
                deepfri_processing_modes=["bp", "mf"],
                save_cmaps=True,
                custom_mapping_file=str(mapping_file_path))

            # --- Assertions ---

            # 1. Check Output Files Created
            self.assertTrue((output_path / "alignment_summary.tsv").exists(),
                            "alignment_summary.tsv not created")
            self.assertTrue((output_path / "results.tsv").exists(),
                            "results.tsv not created")
            self.assertTrue((output_path / "contact_maps").exists(),
                            "contact_maps directory not created")

            # 2. Check alignment_summary.tsv content for real alignment metrics
            with open(output_path / "alignment_summary.tsv", "r") as f:
                lines = f.readlines()
                # Should have header + at least one alignment
                self.assertGreaterEqual(
                    len(lines), 2,
                    "alignment_summary.tsv should have header and alignments")
                # Header check
                header = lines[0].strip()
                self.assertIn("query_id", header)
                self.assertIn("aligned", header)
                self.assertIn("query_identity", header)

                # Should show alignments with custom mapping
                # At least one line should show aligned=True
                aligned_found = any("True" in line for line in lines[1:])
                self.assertTrue(
                    aligned_found,
                    "At least one alignment should be marked as aligned=True")

                # Parse and verify we have diverse alignment metrics (not all identical)
                # Different query sequences should produce different alignment scores
                identities = []
                coverages = []
                for line in lines[1:]:
                    cols = line.strip().split('\t')
                    if cols[1] == "True":  # Only aligned proteins
                        try:
                            identity = float(cols[4])
                            coverage = float(cols[5])
                            identities.append(identity)
                            coverages.append(coverage)
                        except (ValueError, IndexError):
                            pass

                # With 4 different query sequences aligned to same target,
                # we should see variation in alignment metrics
                if len(identities) > 1:
                    identity_variance = max(identities) - min(identities)
                    # Real alignments should have some variation (>0.01 threshold)
                    self.assertGreater(
                        identity_variance, 0.01,
                        "Different query sequences should produce different alignment identities with real alignment"
                    )

            # 4. Check Result Content
            with open(output_path / "results.tsv", "r") as f:
                content = f.read()
                # Should have predictions
                self.assertIn("protein", content)
                self.assertIn("network_type", content)
                self.assertIn("prediction_mode", content)
                # All custom mapping proteins should use GCN (not CNN)
                # (CNN would be for unaligned sequences)

            # 5. Verify that predictions happened
            # Predictor.forward_pass should have been called for GCN
            mock_predictor_instance = mock_predictor_cls.return_value
            self.assertGreater(mock_predictor_instance.forward_pass.call_count,
                               0, "Predictor.forward_pass should be called")

            # 6. Verify build_align_contact_map was called
            self.assertGreater(mock_build_align_contact_map.call_count, 0,
                               "build_align_contact_map should be called")

    @patch("mDeepFRI.pipeline.load_go_to_cog")
    @patch("mDeepFRI.pipeline.load_deepfri_config")
    @patch("mDeepFRI.pipeline.Predictor")
    @patch("mDeepFRI.pipeline.extract_residues_coordinates")
    @patch("mDeepFRI.pipeline.build_align_contact_map")
    @patch("mDeepFRI.pipeline.get_json_values")
    def test_custom_mapping_cnn_fallback_on_structure_failure(
        self,
        mock_get_json_values,
        mock_build_align_contact_map,
        mock_extract_residues_coordinates,
        mock_predictor_cls,
        mock_load_config,
        mock_load_go_to_cog,
    ):
        """Failed structure loads should fall back to CNN prediction."""

        mock_load_config.return_value = {
            "gcn": {"bp": "path/to/gcn_bp.onnx"},
            "cnn": {"bp": "path/to/cnn_bp.onnx"},
            "version": "1.1",
        }
        mock_get_json_values.side_effect = lambda path, key: (
            ["GO:001", "GO:002"] if key == "goterms" else ["BioProcess1",
                                                           "MolFunc1"])
        mock_load_go_to_cog.return_value = {}

        mapping_file_path = Path(self.data_dir) / "small_mapping.tsv"
        if not mapping_file_path.exists():
            self.skipTest(f"Test data file not found: {mapping_file_path}")

        mock_target_seq = "MFSKATANFVRQIDPEGSLIHVSRVNDSQKLVPMALVVKRNRLWFWQRPKYHPTDFTLSD"
        mock_coords = np.zeros((len(mock_target_seq), 3))
        failed_query = "A0A3B4WVX2_Gasdermin_pore_forming"

        def extract_side_effect(structure_string, chain="A", filetype="mmcif"):
            return (mock_target_seq, mock_coords)

        mock_extract_residues_coordinates.side_effect = extract_side_effect

        mock_cmap = np.random.rand(60, 60)

        def build_cmap_side_effect(alignment_result, **kwargs):
            if alignment_result.query_name == failed_query:
                raise ValueError("broken structure")
            return (alignment_result, mock_cmap)

        mock_build_align_contact_map.side_effect = build_cmap_side_effect

        cnn_queries = []
        gcn_queries = []

        def forward_pass_side_effect(seqres, cmap=None):
            if cmap is None:
                cnn_queries.append(seqres)
            else:
                gcn_queries.append(seqres)
            return np.array([0.92, 0.08], dtype=np.float32)

        mock_predictor_instance = mock_predictor_cls.return_value
        mock_predictor_instance.forward_pass.side_effect = forward_pass_side_effect

        query_file = QueryFile(filepath=str(self.query_file_path))
        query_file.load_sequences()

        with tempfile.TemporaryDirectory() as temp_out:
            output_path = Path(temp_out)
            predict_protein_function(
                query_file=query_file,
                databases=(),
                weights="path/to/weights",
                output_path=str(output_path),
                deepfri_processing_modes=["bp"],
                skip_matrix=True,
                custom_mapping_file=str(mapping_file_path))

            with open(output_path / "alignment_summary.tsv", "r") as f:
                rows = {
                    line.split("\t")[0]: line
                    for line in f.read().strip().splitlines()[1:]
                }
            self.assertIn(failed_query, rows)
            self.assertIn("False", rows[failed_query])
            self.assertGreater(len(gcn_queries), 0)
            self.assertGreater(len(cnn_queries), 0)

            with open(output_path / "results.tsv", "r") as f:
                content = f.read()
            self.assertIn("\tcnn\t", content)

    @patch("mDeepFRI.pipeline.Pool")
    @patch("mDeepFRI.pipeline.load_deepfri_config")
    @patch("mDeepFRI.pipeline.Predictor")
    @patch("mDeepFRI.pipeline.align_mmseqs_results")
    @patch("mDeepFRI.pipeline.extract_calpha_coords")
    @patch("mDeepFRI.pipeline.build_align_contact_map")
    @patch("mDeepFRI.pipeline.get_json_values")
    def test_metadata_preamble_is_opt_in(
        self,
        mock_get_json_values,
        mock_build_align_contact_map,
        mock_extract_calpha_coords,
        mock_align_mmseqs_results,
        mock_predictor_cls,
        mock_load_config,
        mock_pool,
    ):
        """results.tsv starts with the column header unless --write-metadata.

        The '##' provenance preamble breaks plain TSV readers, so it must stay
        opt-in; passing command_str/version alone must not enable it.
        """

        mock_pool_instance = mock_pool.return_value
        mock_pool_instance.__enter__.return_value = mock_pool_instance
        mock_pool_instance.map.side_effect = lambda func, iterable: [
            func(i) for i in iterable
        ]
        mock_load_config.return_value = {
            "gcn": {
                "bp": "path/to/gcn_bp.onnx"
            },
            "cnn": {
                "bp": "path/to/cnn_bp.onnx"
            },
            "version": "1.1"
        }
        mock_get_json_values.side_effect = lambda path, key: (
            ["GO:001", "GO:002"] if key == "goterms" else ["Term1", "Term2"])

        mock_aln = MagicMock()
        mock_aln.query_name = "A0A3B4WVX2_Gasdermin_pore_forming"
        mock_aln.target_name = "Target1.1"
        mock_aln.db_name = "TestDB"
        mock_aln.query_identity = 0.9
        mock_aln.query_coverage = 0.8
        mock_aln.target_coverage = 0.8
        mock_aln.query_sequence = "MFSKATANFVRQIDPEGSLIHVSRVNDSQKLVPMALVVKRNRLWFWQRPKYHPTDF"
        mock_align_mmseqs_results.return_value = [mock_aln]
        mock_extract_calpha_coords.return_value = [
            np.zeros((len(mock_aln.query_sequence), 3))
        ]
        mock_build_align_contact_map.return_value = (mock_aln,
                                                     np.random.rand(
                                                         len(mock_aln.
                                                             query_sequence),
                                                         len(mock_aln.
                                                             query_sequence)))
        mock_predictor_cls.return_value.forward_pass.return_value = np.array(
            [0.95, 0.05], dtype=np.float32)

        query_file = QueryFile(filepath=str(self.query_file_path))
        query_file.load_sequences()

        mock_db = MagicMock(spec=Database)
        mock_db.name = "TestDB"
        mock_db.mmseqs_result = "path/to/mmseqs_results.tsv"
        mock_db.sequence_db = "path/to/seq_db"

        for write_metadata in (False, True):
            with self.subTest(write_metadata=write_metadata), \
                    tempfile.TemporaryDirectory() as temp_out:
                output_path = Path(temp_out)
                predict_protein_function(
                    query_file=query_file,
                    databases=(mock_db, ),
                    weights="path/to/weights",
                    output_path=str(output_path),
                    deepfri_processing_modes=["bp"],
                    skip_matrix=True,
                    command_str="mDeepFRI predict-function -i q.faa",
                    version="1.2.0",
                    write_metadata=write_metadata)

                lines = (output_path /
                         "results.tsv").read_text().splitlines()
                if write_metadata:
                    self.assertTrue(lines[0].startswith("## "))
                    self.assertIn("## mDeepFRI-1.2.0", lines)
                    header_index = 3
                else:
                    self.assertFalse(
                        any(line.startswith("##") for line in lines),
                        "results.tsv must not contain a '##' preamble by default"
                    )
                    header_index = 0
                self.assertEqual(lines[header_index].split("\t"),
                                 FINAL_OUTPUT_HEADER)


if __name__ == '__main__':
    unittest.main()
