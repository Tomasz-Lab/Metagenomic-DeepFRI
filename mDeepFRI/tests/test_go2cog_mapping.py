import pathlib
import tempfile
import unittest

import numpy as np

from mDeepFRI.pipeline import _format_ic, _parse_ic, load_go_to_cog


class TestGo2CogMapping(unittest.TestCase):

    def test_parse_ic_missing_values(self):
        self.assertIsNone(_parse_ic(None))
        self.assertIsNone(_parse_ic(""))
        self.assertIsNone(_parse_ic("   "))
        self.assertIsNone(_parse_ic("nan"))
        self.assertIsNone(_parse_ic(np.nan))

    def test_parse_ic_numeric_values(self):
        self.assertEqual(_parse_ic("10.8064"), 10.8064)
        self.assertEqual(_parse_ic(9.51), 9.51)

    def test_format_ic(self):
        self.assertEqual(_format_ic(None), "")
        self.assertEqual(_format_ic(10.8064), "10.81")
        self.assertEqual(_format_ic(9.5), "9.50")

    def test_load_go_to_cog_parses_ic(self):
        with tempfile.NamedTemporaryFile("w",
                                         encoding="utf-8",
                                         delete=False) as handle:
            handle.write(
                "GO term\tCOGs\tGO term name\tNumber of COGs\tIC\tSuperCOGs\tNumber of SuperCOGs\n"
                "GO:0000002\t{'R'}\tmitochondrial genome maintenance\t1\t10.8064\t{'general function'}\t1\n"
                "GO:0000003\t{'D'}\treproduction\t1\t\t{'supercog1'}\t1\n")
            mapping_path = pathlib.Path(handle.name)

        try:
            mapping = load_go_to_cog(mapping_path)
            self.assertEqual(mapping["GO:0000002"][0], 10.8064)
            self.assertIsNone(mapping["GO:0000003"][0])
            self.assertEqual(_format_ic(mapping["GO:0000002"][0]), "10.81")
            self.assertEqual(_format_ic(mapping["GO:0000003"][0]), "")
        finally:
            mapping_path.unlink()


if __name__ == "__main__":
    unittest.main()
