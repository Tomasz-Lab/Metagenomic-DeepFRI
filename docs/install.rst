Installation
============

.. hint::

    Windows OS is not supported. Consider using a Python
    inside `Windows Subsystem for Linux <https://learn.microsoft.com/en-us/windows/wsl/install>`_.

Requirements
------------

- **Python**: >= 3.11, tested with 3.11 - 3.14 on Linux x86-64 and macOS (Apple Silicon)
- **Dependencies**: Automatically installed via pip
- **Wheels**: Linux x86-64 and macOS 14+ on Apple Silicon. On Intel Macs pip builds from source,
  which needs a C++ compiler with OpenMP (``brew install libomp``). Linux aarch64 is not supported,
  as the ``foldcomp`` dependency publishes no aarch64 wheels and its source distribution does not build.
  Linux x86-64 wheels bundle the AVX2 build of MMseqs2; on CPUs without AVX2 install with
  ``pip install --no-binary mdeepfri mdeepfri``.

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
    python -m unittest mDeepFRI.tests
