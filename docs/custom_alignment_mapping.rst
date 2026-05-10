Custom Alignment Mapping
========================

Overview
--------

Metagenomic-DeepFRI's custom alignment mapping feature enables you to **bypass the hierarchical
database search** and provide pre-computed sequence-to-structure mappings. This is useful for:

- **Testing and validation**: Predict function for a small, known set of proteins with specific structures
- **Benchmarking**: Evaluate prediction performance with pre-determined alignments
- **Pre-computed alignments**: Use existing structural annotations or alignments from other tools
- **Non-standard databases**: Integrate custom structure databases that are not in standard formats

Instead of searching multiple databases and selecting best matches, you provide a simple TSV file
that maps each protein to its corresponding structure file. Metagenomic-DeepFRI then:

1. Loads the provided structures
2. **Performs real pairwise alignments** using PyOpal (same as database search)
3. Calculates actual alignment metrics (identity, coverage)
4. Extracts coordinates and predicts function

All proteins with mappings are processed with **GCN** (structure-aware predictions) since structures
are provided.

Requirements
------------

- **Mapping file**: Tab-separated values (TSV) with format: ``protein_id<tab>structure_path``
- **Structure files**: Must be in CIF (mmcif) or PDB format
- **Query file**: FASTA file with protein sequences to predict

Mapping File Format
-------------------

The mapping file must be a tab-separated text file with two columns:

.. code-block:: text

    protein_id  structure_path
    A0A3B4WVX2_Gasdermin_pore_forming    structures/AF-A0A3B4WVX2-F1-model_v6.cif
    G3VTC1_Gasdermin_E    structures/AF-G3VTC1-F1-model_v2.cif
    A2RUC4_TYW5_HUMAN    /absolute/path/to/structure.pdb

**Requirements:**

- First line is treated as header (skipped automatically)
- Structure paths can be relative or absolute
- Relative paths are resolved relative to the mapping file's directory
- Supported file formats: ``.cif``, ``.mmcif``, ``.pdb``
- Proteins not in the mapping will be skipped with a warning

Usage
-----

Command-line Interface
~~~~~~~~~~~~~~~~~~~~~~

Use the ``--custom-mapping`` flag to provide your mapping file:

.. code-block:: console

    python -m mDeepFRI.cli \\
        --input proteins.faa \\
        --output results/ \\
        --weights path/to/model/weights \\
        --custom-mapping protein_structures.tsv

Python API
~~~~~~~~~~

Use the ``custom_mapping_file`` parameter in ``predict_protein_function()``:

.. code-block:: python

    from mDeepFRI.pipeline import QueryFile, predict_protein_function

    # Load query sequences
    query_file = QueryFile(filepath="proteins.faa")
    query_file.load_sequences()

    # Predict using custom mapping
    predict_protein_function(
        query_file=query_file,
        databases=(),  # No databases needed
        weights="path/to/model/weights",
        output_path="results/",
        custom_mapping_file="protein_structures.tsv"
    )

Or use with CLI (add to your CLI configuration or scripts):

.. code-block:: python

    from mDeepFRI.cli import main

    # Build custom argument list
    args = [
        "--input", "proteins.faa",
        "--output", "results/",
        "--weights", "path/to/weights",
        "--custom-mapping", "protein_structures.tsv"
    ]
    main(args)

Example
-------

Let's say you have:

**proteins.faa**:

.. code-block:: text

    >protein_A
    MFSKATANFVRQIDPEGSLIHVSRVNDSQKLVPMALVVKRNRLWFWQRPKYHPTDF
    >protein_B
    MFAKATRNFLRDTDPGGDLIPVSSLNDSDTLQLLSLVVKKKKFWCWQRPKYQFLSV

**protein_structures.tsv**:

.. code-block:: text

    protein_id  structure_path
    protein_A   structures/model_A.cif
    protein_B   structures/model_B.pdb

**Directory structure**:

.. code-block:: text

    .
    ├── proteins.faa
    ├── protein_structures.tsv
    ├── structures/
    │   ├── model_A.cif
    │   └── model_B.pdb
    └── ... (other files)

Then run:

.. code-block:: console

    python -m mDeepFRI.cli \\
        --input proteins.faa \\
        --output predictions/ \\
        --weights path/to/weights \\
        --custom-mapping protein_structures.tsv

Output Files
~~~~~~~~~~~~

Metagenomic-DeepFRI generates the standard output files in the ``predictions/`` directory:

- **alignment_summary.tsv**: Shows alignment metrics for each protein
- **results.tsv**: Final functional predictions (GO terms with scores)
- **prediction_matrix_bp.tsv**, **prediction_matrix_mf.tsv**, etc.: Full prediction matrices per mode
- **contact_maps/**: (Optional, with ``--save-cmaps``) Aligned contact maps

Alignment Metrics in Results
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Even with custom mappings, Metagenomic-DeepFRI performs **real sequence alignments** between
query and structure sequences using PyOpal. The output includes:

- **query_identity**: Sequence identity of the alignment (0.0-1.0)
- **query_coverage**: Fraction of query covered by alignment (0.0-1.0)
- **target_coverage**: Fraction of target structure sequence covered (0.0-1.0)

Example **alignment_summary.tsv**:

.. code-block:: text

    query_id    aligned target_id   db_name query_identity  query_coverage  target_coverage
    protein_A   True    model_A     custom_mapping  0.85    0.92    0.88
    protein_B   True    model_B     custom_mapping  0.78    0.95    0.85

Advanced Usage
--------------

Skipping Proteins
~~~~~~~~~~~~~~~~~

Proteins in your query FASTA that are **not** in the mapping file will be skipped with a warning:

.. code-block:: text

    WARNING: Protein unmapped_protein not found in mapping; skipping.

If you want to predict unmapped proteins using sequence-only mode, use the standard pipeline
with databases instead of custom mapping.


See Also
--------

- :doc:`cli` - Command-line interface documentation
- :doc:`api/pipeline` - Python API documentation for ``predict_protein_function``
