import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

# Import your CLI
from mDeepFRI.cli import main  # adjust path if CLI is in a different file


class TestCLI(unittest.TestCase):
    def setUp(self):
        self.runner = CliRunner()

    def test_cli_version(self):
        """Test that version flag works."""
        result = self.runner.invoke(main, ["--version"])
        self.assertEqual(result.exit_code, 0)
        # grab version
        version_pattern = r"\d+\.\d+\.\d+"
        self.assertRegex(result.output.strip(), version_pattern)

    def test_cli_help(self):
        """Test that help flag shows usage info."""
        result = self.runner.invoke(main, ["--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Usage", result.output)
        self.assertIn("mDeepFRI", result.output)

    def test_predict_function_custom_mapping_option(self):
        """Test that predict-function exposes --custom-mapping."""
        result = self.runner.invoke(main, ["predict-function", "--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("--custom-mapping", result.output)

    def test_compress_structures_requires_carve_pdbs(self):
        input_file = Path(__file__).parent / "data" / "small_query.faa"
        result = self.runner.invoke(main, [
            "predict-function",
            "-i",
            str(input_file),
            "-o",
            "out",
            "--compress-structures",
        ])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("--compress-structures requires --carve-pdbs", result.output)

    def test_predict_function_exposes_skip_and_compress_flags(self):
        result = self.runner.invoke(main, ["predict-function", "--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("--skip-prediction", result.output)
        self.assertIn("--compress-structures", result.output)
        self.assertIn("--pippack-dir", result.output)
        self.assertIn("--pippack-device", result.output)
        self.assertIn("--pippack-n-recycle", result.output)
        self.assertIn("--pippack-temperature", result.output)
        self.assertIn("--pippack-use-resample", result.output)
        self.assertIn("none", result.output)

    def test_carve_pdbs_requires_pippack_dir(self):
        input_file = Path(__file__).parent / "data" / "small_query.faa"
        with patch.dict("os.environ", {}, clear=False):
            # Ensure PIPPACK_DIR is unset for this invocation.
            env = {key: value for key, value in __import__("os").environ.items()
                   if key != "PIPPACK_DIR"}
            result = self.runner.invoke(
                main,
                [
                    "predict-function",
                    "-i",
                    str(input_file),
                    "-o",
                    "out",
                    "--custom-mapping",
                    str(Path(__file__).parent / "data" / "small_mapping.tsv"),
                    "--carve-pdbs",
                    "--skip-prediction",
                ],
                env=env,
            )
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("PIPPack", result.output)

    @patch("mDeepFRI.cli.predict_protein_function")
    @patch("mDeepFRI.cli.load_query_file")
    def test_predict_function_passes_threads(
            self, mock_load_query_file, mock_predict_protein_function):
        input_file = Path(__file__).parent / "data" / "small_query.faa"
        mapping_file = Path(__file__).parent / "data" / "small_mapping.tsv"
        mock_load_query_file.return_value.sequences = {}

        result = self.runner.invoke(main, [
            "predict-function",
            "-i",
            str(input_file),
            "-o",
            "out",
            "--custom-mapping",
            str(mapping_file),
            "--skip-prediction",
            "--threads",
            "12",
        ])

        self.assertEqual(result.exit_code, 0, result.output)
        mock_predict_protein_function.assert_called_once()
        self.assertEqual(mock_predict_protein_function.call_args.kwargs["threads"],
                         12)

    @patch("mDeepFRI.pippack.resolve_pippack_dir")
    @patch("mDeepFRI.cli.predict_protein_function")
    @patch("mDeepFRI.cli.load_query_file")
    def test_predict_function_passes_pippack_options(
            self, mock_load_query_file, mock_predict_protein_function,
            mock_resolve):
        input_file = Path(__file__).parent / "data" / "small_query.faa"
        mapping_file = Path(__file__).parent / "data" / "small_mapping.tsv"
        mock_load_query_file.return_value.sequences = {}

        with tempfile.TemporaryDirectory() as temp_dir:
            pippack_dir = Path(temp_dir)
            mock_resolve.return_value = pippack_dir

            result = self.runner.invoke(main, [
                "predict-function",
                "-i",
                str(input_file),
                "-o",
                "out",
                "--custom-mapping",
                str(mapping_file),
                "--carve-pdbs",
                "--skip-prediction",
                "--pippack-dir",
                str(pippack_dir),
                "--pippack-device",
                "gpu",
                "--pippack-workers",
                "4",
                "--pippack-n-recycle",
                "5",
                "--pippack-temperature",
                "0.1",
                "--pippack-use-resample",
                "--pippack-seed",
                "7",
            ])

        self.assertEqual(result.exit_code, 0, result.output)
        kwargs = mock_predict_protein_function.call_args.kwargs
        self.assertTrue(kwargs["carve_pdbs"])
        self.assertEqual(kwargs["pippack_device"], "gpu")
        self.assertEqual(kwargs["pippack_workers"], 4)
        self.assertEqual(kwargs["pippack_dir"], str(pippack_dir))
        self.assertEqual(kwargs["pippack_n_recycle"], 5)
        self.assertEqual(kwargs["pippack_temperature"], 0.1)
        self.assertTrue(kwargs["pippack_use_resample"])
        self.assertEqual(kwargs["pippack_seed"], 7)
