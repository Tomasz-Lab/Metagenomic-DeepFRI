from . import (test_alignment, test_bio_utils, test_cli, test_conctact_map,
               test_contact_map_utils, test_go2cog_mapping, test_install,
               test_mmseqs, test_pdb, test_pipeline_regression, test_predict,
               test_utils, test_vocabulary_mismatch)


def load_tests(loader, suite, pattern):
    suite.addTests(loader.loadTestsFromModule(test_alignment))
    suite.addTests(loader.loadTestsFromModule(test_bio_utils))
    suite.addTests(loader.loadTestsFromModule(test_cli))
    suite.addTests(loader.loadTestsFromModule(test_conctact_map))
    suite.addTests(loader.loadTestsFromModule(test_contact_map_utils))
    suite.addTests(loader.loadTestsFromModule(test_go2cog_mapping))
    suite.addTests(loader.loadTestsFromModule(test_install))
    suite.addTests(loader.loadTestsFromModule(test_mmseqs))
    suite.addTests(loader.loadTestsFromModule(test_pdb))
    suite.addTests(loader.loadTestsFromModule(test_pipeline_regression))
    suite.addTests(loader.loadTestsFromModule(test_predict))
    suite.addTests(loader.loadTestsFromModule(test_utils))
    suite.addTests(loader.loadTestsFromModule(test_vocabulary_mismatch))
    return suite
