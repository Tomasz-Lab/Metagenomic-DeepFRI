Welcome to Metagenomic-DeepFRI's documentation! |Stars|
=======================================================

.. |Stars| image:: https://img.shields.io/github/stars/Tomasz-Lab/Metagenomic-DeepFRI.svg?style=social&maxAge=3600&label=Star
   :target: https://github.com/Tomasz-Lab/Metagenomic-DeepFRI/stargazers


Overview
--------

Metagenomic-DeepFRI is a high-performance pipeline for annotating protein sequences
with Gene Ontology (GO) terms using `DeepFRI <https://github.com/flatironinstitute/DeepFRI>`_,
a deep learning model for functional protein annotation.

Protein function prediction is increasingly important as sequencing technologies
generate vast numbers of novel sequences. Metagenomic-DeepFRI combines:

- **Structure information** from FoldComp databases (AlphaFold, ESMFold, PDB, etc.)
- **Sequence-based predictions** using DeepFRI's neural networks
- **Fast searches** with MMseqs2 for database alignment
- **Significant speedup** of 2-12× compared to standard DeepFRI implementation

Pipeline Stages
^^^^^^^^^^^^^^^

1. Search proteins similar to query in PDB and supply FoldComp databases with MMseqs2.
2. Find the best alignment among MMseqs2 hits using PyOpal.
3. Align target protein contact map to query protein with unknown structure.
4. Run DeepFRI with the structure if found in the database, otherwise run DeepFRI
   with sequence only.

Built With
^^^^^^^^^^

- `MMseqs2 <https://github.com/soedinglab/MMseqs2>`_ - Fast sequence search
- `pyOpal <https://github.com/althonos/pyOpal>`_ - SIMD-accelerated pairwise alignment
- `DeepFRI <https://github.com/flatironinstitute/DeepFRI>`_ - Deep learning protein function prediction
- `FoldComp <https://github.com/steineggerlab/foldcomp>`_ - Protein structure compression
- `ONNX <https://github.com/onnx/onnx>`_ - Neural network inference

Setup
-----

.. code:: console

    pip install mdeepfri



Contents
--------

.. toctree::
   :maxdepth: 2

    Installation <install>
    Commands <cli>
    API Reference <api/index>
    Examples <examples/index>
    Result Interpretation <results>
    Custom Alignment Mapping <custom_alignment_mapping>
    Changelog <changelog>


.. list-table::

    * - .. image :: _images/sano_logo.png
          :align: center
          :width: 400
      - .. image :: _images/flatiron-institute-bloc-ii.jpg
          :align: center
          :width: 400
    * - .. image :: _images/full_col2x-100.jpg
          :align: center
          :width: 400
      -
