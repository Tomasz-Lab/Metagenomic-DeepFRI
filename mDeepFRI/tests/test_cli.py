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

    def test_benchmark_without_weights(self):
        """Test that benchmark builds contact maps for every bin and setting."""
        query = Path(__file__).parent / "data" / "small_query.faa"
        args = ["benchmark", "-i", str(query), "--save-cmaps"]
        with TemporaryDirectory() as tmp:
            output = Path(tmp) / "benchmark"
            result = self.runner.invoke(
                main, args + [
                    "-o",
                    str(output), "--identity-bin", "0", "0.5",
                    "--identity-bin", "0.5", "1.01", "--generate-contacts",
                    "0", "--generate-contacts", "2"
                ])
            self.assertEqual(result.exit_code, 0, result.exception)

            bin_path = output / "identity_bin_0.50-1.01"
            for path in [
                    output / "timing.tsv",
                    output / "database_search" / "pdb100_230517_results.tsv",
                    bin_path / "hits" / "pdb100_230517.tsv",
                    bin_path / "alignments" / "pdb100_230517.tsv",
                    output / "identity_bin_0.00-0.50" / "generate_contacts_0" /
                    "alignment_summary.tsv"
            ]:
                self.assertTrue(path.exists(), path)
            self.assertTrue(
                list((bin_path / "generate_contacts_2" /
                      "contact_maps").glob("*.npy")))

            rerun = Path(tmp) / "rerun"
            result = self.runner.invoke(
                main, args + [
                    "-o",
                    str(rerun), "--identity-bin", "0.5", "1.01",
                    "--search-results",
                    str(output / "database_search")
                ])
            self.assertEqual(result.exit_code, 0, result.exception)
            hits = Path(
                "identity_bin_0.50-1.01") / "hits" / "pdb100_230517.tsv"
            self.assertEqual((output / hits).read_text(),
                             (rerun / hits).read_text())
