import tempfile
import unittest
from pathlib import Path

from mDeepFRI.pipeline import _load_custom_mapping_file


class TestCustomMappingLoader(unittest.TestCase):
    def test_comma_csv_with_structure_path_column(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            structures_dir = temp_path / "structures"
            structures_dir.mkdir()
            pdb_file = structures_dir / "template.pdb"
            pdb_file.write_text(
                "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N\n",
                encoding="utf-8",
            )

            mapping_file = temp_path / "mapping.csv"
            mapping_file.write_text(
                "query,target,structure_path\n"
                f"protein1,AFDB:AF-TEST,/missing/part_01/{pdb_file.name}\n",
                encoding="utf-8",
            )

            mapping = _load_custom_mapping_file(mapping_file)
            self.assertEqual(mapping["protein1"], str(pdb_file))

    def test_tab_two_column_mapping(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            pdb_file = temp_path / "template.pdb"
            pdb_file.write_text(
                "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N\n",
                encoding="utf-8",
            )

            mapping_file = temp_path / "mapping.tsv"
            mapping_file.write_text(
                "protein_id\tstructure_path\n"
                f"protein1\t{pdb_file}\n",
                encoding="utf-8",
            )

            mapping = _load_custom_mapping_file(mapping_file)
            self.assertEqual(mapping["protein1"], str(pdb_file))


if __name__ == "__main__":
    unittest.main()
