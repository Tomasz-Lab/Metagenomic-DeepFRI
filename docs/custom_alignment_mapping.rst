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
4. Extracts coordinates and predicts function with **GCN** when the structure is usable
5. Falls back to **CNN** (sequence-only) when no usable structure is available

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

Usage
-----

Command-line Interface
~~~~~~~~~~~~~~~~~~~~~~

Use the ``--custom-mapping`` flag with ``predict-function``:

.. code-block:: console

    mDeepFRI predict-function \\
        --input proteins.faa \\
        --output results/ \\
        --weights path/to/model/weights \\
        --custom-mapping protein_structures.tsv

When ``--custom-mapping`` is provided, ``--db-path`` is not required and database
search is skipped.

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

    mDeepFRI predict-function \\
        --input proteins.faa \\
        --output predictions/ \\
        --weights path/to/weights \\
        --custom-mapping protein_structures.tsv

CNN Fallback
------------

Custom mapping uses the same GCN/CNN split as the standard database-search pipeline:

+-------------------------------+---------------------------+
| Condition                     | Network used              |
+===============================+===========================+
| Structure loaded and aligned  | GCN (structure-aware)     |
| Missing from mapping file     | CNN (sequence-only)       |
| Structure file not found      | CNN (sequence-only)       |
| Structure parse/load error    | CNN (sequence-only)       |
+-------------------------------+---------------------------+

When a structure cannot be used, mDeepFRI logs a warning such as:

.. code-block:: text

    WARNING: protein_B: structure file not found: structures/missing.cif. Falling back to CNN (sequence-only) prediction.

Those proteins appear in ``alignment_summary.tsv`` with ``aligned=False`` and are
annotated in ``results.tsv`` with ``network_type=cnn``.

Output Files
~~~~~~~~~~~~

Metagenomic-DeepFRI generates the standard output files in the ``predictions/`` directory:

- **alignment_summary.tsv**: Shows alignment metrics for each protein
- **results.tsv**: Final functional predictions (GO terms with scores)
- **prediction_matrix_bp.tsv**, **prediction_matrix_mf.tsv**, etc.: Full prediction matrices per mode
- **contact_maps/**: (Optional, with ``--save-cmaps``) Aligned contact maps for GCN proteins only

Alignment Metrics in Results
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Even with custom mappings, Metagenomic-DeepFRI performs **real sequence alignments** between
query and structure sequences using PyOpal for proteins with loadable structures. The output includes:

- **query_identity**: Sequence identity of the alignment (0.0-1.0)
- **query_coverage**: Fraction of query covered by alignment (0.0-1.0)
- **target_coverage**: Fraction of target structure sequence covered (0.0-1.0)

Example **alignment_summary.tsv**:

.. code-block:: text

    query_id    aligned target_id   db_name         query_identity  query_coverage  target_coverage
    protein_A   True    model_A     custom_mapping  0.85            0.92            0.88
    protein_B   False                               nan             nan             nan

``protein_B`` failed structure loading and was predicted with CNN instead.

See Also
--------

- :doc:`cli` - Command-line interface documentation
- :doc:`api/pipeline` - Python API documentation for ``predict_protein_function``
