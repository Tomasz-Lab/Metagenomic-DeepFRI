import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

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

    def test_search_databases(self):
        """Test that search-databases runs a PDB100 search."""
        query = Path(__file__).parent / "data" / "small_query.faa"
        with TemporaryDirectory() as output:
            result = self.runner.invoke(
                main, ["search-databases", "-i",
                       str(query), "-o", output])
            self.assertEqual(result.exit_code, 0, result.exception)
            self.assertTrue(
                (Path(output) / "pdb100_230517_results.tsv").exists())
