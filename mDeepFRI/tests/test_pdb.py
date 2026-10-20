import pathlib
import unittest
from unittest import mock

import requests

from mDeepFRI.pdb import create_pdb_mmseqs, get_pdb_seq_coords

STRUCTURE = pathlib.Path(
    __file__).parent / "data" / "AF-A0A3B4WVX2-F1-model_v6.cif"


class TestPDB(unittest.TestCase):
    def test_create_pdb_mmseqs(self):
        pdb100 = create_pdb_mmseqs()

        self.assertTrue(pathlib.Path(pdb100.sequence_db).exists())
        self.assertTrue(pathlib.Path(pdb100.mmseqs_db).exists())


class TestGetPDBSeqCoords(unittest.TestCase):
    def setUp(self):
        self.structure = STRUCTURE.read_text()

    def test_extracts_chain(self):
        with mock.patch("mDeepFRI.pdb.get_pdb_structure",
                        return_value=self.structure):
            sequence, coords = get_pdb_seq_coords("1abc_A", "query")

        self.assertEqual(len(sequence), len(coords))

    def test_missing_chain_is_skipped(self):
        with mock.patch("mDeepFRI.pdb.get_pdb_structure",
                        return_value=self.structure):
            with self.assertWarns(UserWarning):
                result = get_pdb_seq_coords("1abc_Z", "query")

        self.assertEqual(result, (None, None))

    def test_failed_download_is_skipped(self):
        response = mock.Mock()
        response.raise_for_status.side_effect = requests.HTTPError(
            "404 Client Error")
        with mock.patch("mDeepFRI.pdb.requests.get", return_value=response):
            with self.assertWarns(UserWarning):
                result = get_pdb_seq_coords("0zzz_A", "query")

        self.assertEqual(result, (None, None))
