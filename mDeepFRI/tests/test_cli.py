import unittest
from pathlib import Path

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
        self.assertIn("none", result.output)
