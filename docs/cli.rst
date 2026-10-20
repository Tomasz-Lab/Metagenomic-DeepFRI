Commands (CLI)
==============

The ``mDeepFRI`` command-line interface provides three main commands:

- ``get-models`` - Download pre-trained DeepFRI models (v1.0 or v1.1)
- ``predict-function`` - Run the prediction pipeline on protein sequences
- ``benchmark`` - Evaluate structure retrieval and prediction across identity bins of database hits

Key Features
------------

**Prediction Modes**

The GO ontology contains three subontologies, plus EC number prediction:

- Molecular Function (mf)
- Biological Process (bp)
- Cellular Component (cc)
- Enzyme Commission numbers (ec)

By default, predictions are made in all 4 categories. Use ``-p`` or ``--processing-modes``
to select specific modes.

**Hierarchical Database Search**

Different databases have different levels of evidence. For example, PDB structures
are experimental and considered highest quality. Use ``-d`` or ``--databases``
multiple times to search databases hierarchically.

**Performance Options**

- ``--skip-matrix`` - Skip writing large prediction matrix files to save disk space
- ``--threads`` - Parallelize alignment, contact map alignment, and annotation
- GPU acceleration is automatically used if CUDA is available

Command Reference
-----------------

.. click:: mDeepFRI.cli:main
    :prog: mDeepFRI
    :nested: full
