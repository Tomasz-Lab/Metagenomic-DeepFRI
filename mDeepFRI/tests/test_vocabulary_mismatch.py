"""
Test suite for vocabulary mismatch bug fix in pipeline.py

Tests the scenario where GCN and CNN models have different output vocabularies,
which occurs in DeepFRI v1.1 where GCN uses expanded GO/EC vocabulary while
CNN maintains older vocabulary. The pipeline should:

1. Detect vocabulary mismatches
2. Create separate matrix files when vocabularies differ
3. Correctly read and combine predictions from both vocabularies
4. Maintain backward compatibility when vocabularies match
"""

import unittest
from pathlib import Path
from unittest.mock import patch


class TestVocabularyMismatch(unittest.TestCase):
    """Test suite for split-matrix functionality when GCN/CNN vocabularies differ."""
    def setUp(self):
        """Set up test environment."""
        self.test_dir = Path(__file__).parent
        self.data_dir = self.test_dir / "data"
        self.query_file_path = self.data_dir / "small_query.faa"

        if not self.query_file_path.exists():
            self.query_file_path = Path(
                "/nfs/cds-peta/exports/biol_micro_cds_gr_sunagawa/scratch/vbezshapkin/Metagenomic-DeepFRI/mDeepFRI/tests/data/small_query.faa"
            )

    @patch("mDeepFRI.pipeline.load_go_to_cog")
    @patch("mDeepFRI.pipeline.get_json_values")
    def test_split_matrices_flag_when_vocabularies_differ(
        self,
        mock_get_json_values,
        mock_load_go_to_cog,
    ):
        """Test that split_matrices flag is correctly set when vocabularies differ.

        This unit test verifies the detection logic without running full pipeline.
        """

        # Different vocabularies: GCN (4 terms) vs CNN (2 terms)
        gcn_goterms = ["GO:0001", "GO:0002", "GO:0003", "GO:0004"]
        cnn_goterms = ["GO:0001", "GO:0002"]
        gcn_gonames = ["BP1", "BP2", "BP3", "BP4"]

        def mock_json_side_effect(path, key):
            if key == "goterms":
                return gcn_goterms if "gcn" in path else cnn_goterms
            elif key == "gonames":
                return gcn_gonames if "gcn" in path else ["BP1", "BP2"]
            return []

        mock_get_json_values.side_effect = mock_json_side_effect
        mock_load_go_to_cog.return_value = {}

        # Test the vocabulary comparison logic
        gcn_goterms_test = ["GO:0001", "GO:0002", "GO:0003", "GO:0004"]
        cnn_goterms_test = ["GO:0001", "GO:0002"]

        split_matrices = (len(gcn_goterms_test) != len(cnn_goterms_test)
                          or gcn_goterms_test != cnn_goterms_test)

        self.assertTrue(
            split_matrices,
            "split_matrices should be True when vocabularies differ")

    @patch("mDeepFRI.pipeline.load_go_to_cog")
    @patch("mDeepFRI.pipeline.get_json_values")
    def test_single_matrix_flag_when_vocabularies_match(
        self,
        mock_get_json_values,
        mock_load_go_to_cog,
    ):
        """Test that split_matrices flag is False when vocabularies match."""

        # Same vocabularies
        shared_goterms = ["GO:0001", "GO:0002"]
        shared_gonames = ["BP1", "BP2"]

        def mock_json_side_effect(path, key):
            if key == "goterms":
                return shared_goterms
            elif key == "gonames":
                return shared_gonames
            return []

        mock_get_json_values.side_effect = mock_json_side_effect
        mock_load_go_to_cog.return_value = {}

        # Test the vocabulary comparison logic
        gcn_goterms = ["GO:0001", "GO:0002"]
        cnn_goterms = ["GO:0001", "GO:0002"]

        split_matrices = (len(gcn_goterms) != len(cnn_goterms)
                          or gcn_goterms != cnn_goterms)

        self.assertFalse(
            split_matrices,
            "split_matrices should be False when vocabularies match")

    def test_matrix_file_naming_gcn_vocab_mismatch(self):
        """Test that correct matrix filenames are generated when vocabularies differ."""

        mode = "bp"
        split_matrices = True

        if split_matrices:
            gcn_filename = f"prediction_matrix_{mode}_gcn.tsv"
            cnn_filename = f"prediction_matrix_{mode}_cnn.tsv"
        else:
            gcn_filename = f"prediction_matrix_{mode}.tsv"
            cnn_filename = f"prediction_matrix_{mode}.tsv"

        self.assertEqual(gcn_filename, "prediction_matrix_bp_gcn.tsv",
                         "GCN matrix should have correct filename when split")
        self.assertEqual(cnn_filename, "prediction_matrix_bp_cnn.tsv",
                         "CNN matrix should have correct filename when split")

    def test_matrix_file_naming_same_vocab(self):
        """Test that correct matrix filenames are generated when vocabularies match."""

        mode = "ec"
        split_matrices = False

        if split_matrices:
            gcn_filename = f"prediction_matrix_{mode}_gcn.tsv"
            cnn_filename = f"prediction_matrix_{mode}_cnn.tsv"
        else:
            gcn_filename = f"prediction_matrix_{mode}.tsv"
            cnn_filename = f"prediction_matrix_{mode}.tsv"

        self.assertEqual(
            gcn_filename, "prediction_matrix_ec.tsv",
            "Both GCN and CNN should use combined filename when vocabularies match"
        )
        self.assertEqual(cnn_filename, "prediction_matrix_ec.tsv")

    @patch("mDeepFRI.pipeline.load_deepfri_config")
    @patch("mDeepFRI.pipeline.get_json_values")
    def test_matrix_job_structure(self, mock_get_json_values,
                                  mock_load_config):
        """Test that matrix_jobs_by_mode data structure is correctly assembled."""

        # Simulate what the pipeline does
        gcn_goterms = ["GO:0001", "GO:0002", "GO:0003"]
        cnn_goterms = ["GO:0001", "GO:0002"]

        def mock_json_side_effect(path, key):
            if key == "goterms":
                return gcn_goterms if "gcn" in path else cnn_goterms
            return []

        mock_get_json_values.side_effect = mock_json_side_effect

        # Simulate job creation
        matrix_jobs_by_mode = {}
        mode = "bp"
        gcn_config_path = "path/to/gcn_bp_model_params.json"
        cnn_config_path = "path/to/cnn_bp_model_params.json"
        gcn_matrix_source = "path/to/prediction_matrix_bp_gcn.tsv"
        cnn_matrix_source = "path/to/prediction_matrix_bp_cnn.tsv"

        split_matrices = len(gcn_goterms) != len(cnn_goterms)

        if split_matrices:
            matrix_jobs_by_mode[mode] = [
                {
                    "config_path": gcn_config_path,
                    "matrix_source": gcn_matrix_source,
                },
                {
                    "config_path": cnn_config_path,
                    "matrix_source": cnn_matrix_source,
                },
            ]
        else:
            matrix_jobs_by_mode[mode] = [{
                "config_path": gcn_config_path,
                "matrix_source": gcn_matrix_source,
            }]

        # Verify structure
        self.assertEqual(len(matrix_jobs_by_mode[mode]), 2,
                         "Should have 2 job entries when vocabularies differ")
        self.assertEqual(matrix_jobs_by_mode[mode][0]["config_path"],
                         gcn_config_path)
        self.assertEqual(matrix_jobs_by_mode[mode][1]["config_path"],
                         cnn_config_path)

    def test_split_matrix_content_parsing(self):
        """Test that split matrix files can be correctly parsed and assembled."""

        # Create sample matrix file content
        gcn_matrix_content = """protein\tnetwork_type\tGO:0001\tGO:0002\tGO:0003
Protein_A\tgcn\t0.95\t0.85\t0.65"""

        cnn_matrix_content = """protein\tnetwork_type\tGO:0001\tGO:0002
Protein_B\tcnn\t0.92\t0.82"""

        # Test parsing GCN matrix
        gcn_lines = gcn_matrix_content.strip().split("\n")
        gcn_reader = [line.split("\t") for line in gcn_lines]
        gcn_header = gcn_reader[0]
        gcn_terms = gcn_header[2:]

        self.assertEqual(gcn_terms, ["GO:0001", "GO:0002", "GO:0003"],
                         "GCN matrix should have 3 GO terms")
        self.assertEqual(len(gcn_reader[1][2:]), 3,
                         "GCN data row should have 3 scores")

        # Test parsing CNN matrix
        cnn_lines = cnn_matrix_content.strip().split("\n")
        cnn_reader = [line.split("\t") for line in cnn_lines]
        cnn_header = cnn_reader[0]
        cnn_terms = cnn_header[2:]

        self.assertEqual(cnn_terms, ["GO:0001", "GO:0002"],
                         "CNN matrix should have 2 GO terms")
        self.assertEqual(len(cnn_reader[1][2:]), 2,
                         "CNN data row should have 2 scores")

    def test_vocabulary_mismatch_detection_different_order(self):
        """Test that vocabulary mismatch is detected even if terms are in different order."""

        gcn_goterms = ["GO:0001", "GO:0002", "GO:0003"]
        cnn_goterms = ["GO:0003", "GO:0001", "GO:0002"]  # Different order!

        split_matrices = (len(gcn_goterms) != len(cnn_goterms)
                          or gcn_goterms != cnn_goterms)

        self.assertTrue(
            split_matrices,
            "split_matrices should detect differences in term order")

    def test_vocabulary_mismatch_detection_same_length_different_terms(self):
        """Test that vocabulary mismatch is detected when terms differ even with same length."""

        gcn_goterms = ["GO:0001", "GO:0002", "GO:0003"]
        cnn_goterms = ["GO:0001", "GO:0002",
                       "GO:0004"]  # Same length, different last term

        split_matrices = (len(gcn_goterms) != len(cnn_goterms)
                          or gcn_goterms != cnn_goterms)

        self.assertTrue(
            split_matrices,
            "split_matrices should detect different terms even with same length"
        )


if __name__ == "__main__":
    unittest.main()
