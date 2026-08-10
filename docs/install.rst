Installation
============

.. hint::

    Windows OS is not supported. Consider using a Python
    inside `Windows Subsystem for Linux <https://learn.microsoft.com/en-us/windows/wsl/install>`_.

Requirements
------------

- **Python**: >= 3.9, < 3.13 (tested with 3.11)
- **Dependencies**: Automatically installed via pip

Installation from PyPI
----------------------

Install using pip (recommended):

.. code:: console

    pip install mdeepfri

Installation from Source
-------------------------

For development or the latest features:

.. code:: console

    git clone https://github.com/Tomasz-Lab/Metagenomic-DeepFRI.git
    cd Metagenomic-DeepFRI
    pip install -e .

Download Models
---------------

Two versions of models are available:

- **v1.0**: Original version from DeepFRI publication
- **v1.1**: Finetuned on AlphaFold models and machine-generated Gene Ontology Uniprot annotations

To download models run command:

.. code:: console

    mDeepFRI get-models --output path/to/weights/folder -v {1.0 or 1.1}

Prepare Structural Database
----------------------------

The PDB database will be automatically downloaded and installed during the first run of ``mDeepFRI``.

You can download additional FoldComp databases from the
`FoldComp website <https://foldcomp.steineggerlab.workers.dev/>`_.
During first run, FASTA sequences will be extracted from FoldComp database
and MMseqs2 database will be created and indexed.

Tested databases:

- ``afdb_swissprot``
- ``afdb_swissprot_v4``
- ``afdb_rep_v4``
- ``afdb_rep_dark_v4``
- ``afdb_uniprot_v4``
- ``esmatlas``
- ``esmatlas_v2023_02``
- ``highquality_clust30``

.. warning::

    Do not rename downloaded databases. FoldComp has certain inconsistencies
    in the way FASTA sequences are extracted, therefore the pipeline was
    tweaked for each database.

For custom FoldComp databases, use the FoldComp executable:

.. code:: console

    foldcomp compress [-t number] <dir|tar(.gz)> [<dir|tar|db>]

Verification
------------

Verify installation:

.. code:: console

    mDeepFRI --help
    python -c "import mDeepFRI; print(mDeepFRI.__version__)"
