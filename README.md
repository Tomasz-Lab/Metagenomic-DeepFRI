<!-- markdownlint-disable-file MD024 MD013 -->
# 🍳 Metagenomic-DeepFRI [![Stars](https://img.shields.io/github/stars/bioinf-MCB/Metagenomic-DeepFRI.svg?style=social&maxAge=3600&label=Star)](https://github.com/bioinf-MCB/Metagenomic-DeepFRI/stargazers)

[![License](https://img.shields.io/badge/license-BSD%203--Clause-blue.svg?style=flat-square&maxAge=2678400)](https://choosealicense.com/licenses/bsd-3-clause/)
[![PyPI](https://img.shields.io/pypi/v/mdeepfri.svg?style=flat-square&maxAge=3600&logo=PyPI)](https://pypi.org/project/mdeepfri)
[![Wheel](https://img.shields.io/pypi/wheel/mdeepfri.svg?style=flat-square&maxAge=3600)](https://pypi.org/project/mdeepfri/#files)
[![Python Versions](https://img.shields.io/pypi/pyversions/mdeepfri.svg?style=flat-square&maxAge=600&logo=python)](https://pypi.org/project/mdeepfri/#files)
[![Python Implementations](https://img.shields.io/pypi/implementation/mdeepfri.svg?style=flat-square&maxAge=600&label=impl)](https://pypi.org/project/mdeepfri/#files)
[![Source](https://img.shields.io/badge/source-GitHub-303030.svg?maxAge=2678400&style=flat-square)](https://github.com/bioinf-MCB/Metagenomic-DeepFRI/)
[![GitHub issues](https://img.shields.io/github/issues/bioinf-MCB/Metagenomic-DeepFRI.svg?style=flat-square&maxAge=600)](https://github.com/bioinf-MCB/Metagenomic-DeepFRI//issues)
[![Docs](https://img.shields.io/readthedocs/metagenomic-deepfri/latest?style=flat-square&maxAge=600)](https://metagenomic-deepfri.readthedocs.io/)
[![Changelog](https://img.shields.io/badge/keep%20a-changelog-8A0707.svg?maxAge=2678400&style=flat-square)](https://github.com/bioinf-MCB/Metagenomic-DeepFRI/blob/main/CHANGELOG.md)
[![Downloads](https://img.shields.io/pypi/dm/mdeepfri?style=flat-square&color=303f9f&maxAge=86400&label=downloads)](https://pepy.tech/project/mdeepfri)

*A pipeline for annotation of genes with [DeepFRI](https://github.com/flatironinstitute/DeepFRI),
a deep learning model for functional protein annotation with
[Gene Ontology (GO) terms](https://geneontology.org/docs/go-annotations/) and
mapping to [Cluster of Orthologous Groups (COG)](https://www.ncbi.nlm.nih.gov/research/cog) categories.
It incorporates [FoldComp](https://github.com/steineggerlab/foldcomp)
databases of predicted protein structures for fast annotation of
metagenomic gene catalogues.*

## 🔍 Overview

Metagenomic-DeepFRI is a high-performance pipeline for
annotating protein sequences with Gene Ontology (GO) terms using
[DeepFRI](https://github.com/flatironinstitute/DeepFRI),
a deep learning model for functional protein annotation.

Protein function prediction is increasingly important
as sequencing technologies generate vast numbers of novel sequences.
Metagenomic-DeepFRI combines:

- **Structure information** from FoldComp databases (AlphaFold, ESMFold, PDB, etc.)
- **Sequence-based predictions** using DeepFRI's neural networks
- **Fast searches** with MMseqs2 for database alignment
- **Significant speedup** of
[2-12×](https://github.com/bioinf-mcb/Metagenomic-DeepFRI/blob/main/weight_convert/onnx_vs_tf2.png)
compared to standard DeepFRI implementation.

### 📋 Pipeline stages

1. Search proteins similar to query in PDB and supply `FoldComp` databases
with `MMseqs2`.
2. Find the best alignment among `MMseqs2` hits using `PyOpal`.
3. Align target protein contact map to query protein with unknown structure.
4. Run `DeepFRI` with the structure if found in the database, otherwise run
`DeepFRI` with sequence only.

![image.png](https://github.com/bioinf-mcb/Metagenomic-DeepFRI/blob/main/docs/_images/mdeepfri-design.drawio.png)

### 🛠️ Built With

- [MMseqs2](https://github.com/soedinglab/MMseqs2)
- [pyOpal](https://github.com/althonos/pyOpal)
- [DeepFRI](https://github.com/flatironinstitute/DeepFRI)
- [FoldComp](https://github.com/steineggerlab/foldcomp)
- [ONNX](https://github.com/onnx/onnx)

## 📦 Requirements

- **Python:** >= 3.11, < 3.13 (tested with 3.11)
- **Dependencies:** Automatically installed via pip

## 🔧 Installation

1\. Install from PyPI. Installation might take a few minutes
due to download of MMseqs2 binaries.

```{code-block} bash
pip install mdeepfri
```

The default install uses CPU ONNX Runtime and does not require CUDA.
On Linux x86_64, GPU acceleration may still be used automatically when a
compatible CUDA stack is already present on your machine.

For a self-contained GPU install, use:

```{code-block} bash
pip install mdeepfri[gpu]
```

This pins ONNX Runtime 1.26 with CUDA 12 libraries bundled via pip
(`nvidia-cuda-runtime-cu12`, `nvidia-cudnn-cu12`, etc.). No system CUDA
toolkit upgrade is required, but an NVIDIA driver new enough for CUDA 12
is still needed.

2\. Run and view the help message.

```{code-block} bash
mDeepFRI --help
```

## 💡 Usage

### 1. Prepare structural database

#### 1.1 Existing `FoldComp` databases

The PDB database will be automatically downloaded and installed
during the first run of `mDeepFRI`.
The PDB suffers from formatting inconsistencies, therefore during PDB alignment
around 10% will fail and will be reported via `WARNING`.
We suggest coupling PDB search with predicted databases, as it massively
improves the structural coverage of the protein universe.
A good protein structure allows DeepFRI to annotate the function in more detail.
However, the sequence branch of the model has the largest weight,
thus even if the predicted structure is erroneous,
it will have a minor effect on the prediction.
The details can be found in
[the original manuscript, fig. 2A](https://www.nature.com/articles/s41467-021-23303-9/figures/2).

You can download additional databases
from [website](https://foldcomp.steineggerlab.workers.dev/).
During a first run, FASTA sequences will be extracted
from `FoldComp` database and `MMseqs2` database will be created and indexed.
You can use different databases, but be mindful
that computation time might increase exponentially with the size of the database.

Tested databases:

- `afdb_swissprot`
- `afdb_swissprot_v4`
- `afdb_rep_v4`
- `afdb_rep_dark_v4`
- `afdb_uniprot_v4`
- `esmatlas`
- `esmatlas_v2023_02`
- `highquality_clust30`

`ATTENTION:` Please, do not rename downloaded databases.
`FoldComp` has certain inconsistencies in the way FASTA sequences are extracted ([example](https://github.com/steineggerlab/foldcomp/issues/51)),
therefore pipeline was tweaked for each database.
If database you need does not work, please report in
[issues](https://github.com/bioinf-mcb/Metagenomic-DeepFRI/issues).

`ATTENTION:` database creation is a very sensitive step which
relies on external software.
If pipeline is interrupted during this step, the databases might be corrupted.
If you are not sure about your database,
rerun the pipeline with `--overwrite` flag -
it will rerun database creation process.

#### 1.2. Custom `FoldComp` database

In order to use personal database of structures,
you will have to create a custom FoldComp database.
For that, download a `FoldComp` executable and run  the following command:

```{code-block} bash
foldcomp compress [-t number] <dir|tar(.gz)> [<dir|tar|db>]
```

### 2. Download models

Two versions of models available:

- `v1.0` - is the original version from DeepFRI publication.
- `v1.1` - is a version finetuned on AlphaFold models and machine-generated Gene Ontology Uniprot annotations ([details](https://www.nature.com/articles/s41467-025-63250-3)).

To download models run command:

```{code-block} bash
mDeepFRI get-models --output path/to/weights/folder -v {1.0 or 1.1}
```

### 3. Predict protein function & capture log

```{code-block} bash
mDeepFRI predict-function -i /path/to/protein/sequences \
-d /path/to/foldcomp/database/ \
-w /path/to/deepfri/weights/folder \
-o /output_path > log.txt
```

Other available parameters can be found upon command `mDeepFRI --help`.

## ✅ Results

The output folder will contain several files from different stages of the pipeline:

### Main Output Files

1. **`results.tsv`** - Primary output file containing
all functional predictions from the DeepFRI model.

2. **`alignment_summary.tsv`** - Summary of alignment statistics for
   each query protein, showing which queries were
   successfully aligned to database structures.

3. **`database_search/`** - Directory containing individual search results for
   each database queried:
   - `{database_name}_results.tsv` - One file per database searched
     (e.g., `pdb100_230517_results.tsv`, `afdb_swissprot_v4_results.tsv`)

4. **`prediction_matrix_*.tsv`** - Detailed prediction matrices for each
   ontology mode:
   - `prediction_matrix_bp.tsv` - Biological Process predictions
   - `prediction_matrix_cc.tsv` - Cellular Component predictions
   - `prediction_matrix_ec.tsv` - Enzyme Commission predictions
   - `prediction_matrix_mf.tsv` - Molecular Function predictions

   These files contain raw prediction scores for every protein × GO term
   combination and can be very large (>50MB).

5. **`query.mmseqsDB`** + associated index files - MMseqs2 database created
   from input query sequences.

### Primary Output Format (`results.tsv`)

The main output file contains the following columns:

- **protein** - Name of the protein from the input FASTA file.
- **network_type** - Type of neural network used for prediction:
  - `gcn` (Graph Convolutional Network) - Used when structural information
    is available from database alignment, providing more confident
    predictions.
  - `cnn` (Convolutional Neural Network) - Used when no proteins above
    similarity cutoff (50% identity by default) are found.
- **prediction_mode** - Ontology category: `mf` (Molecular Function),
  `bp` (Biological Process), `cc` (Cellular Component), or
  `ec` (Enzyme Commission).
- **go_term** - Predicted GO term identifier or EC number.
- **score** - DeepFRI confidence score for the prediction. Higher scores
  indicate greater confidence. See the
  [DeepFRI publication](https://www.nature.com/articles/s41467-021-23303-9)
  for details.
- **go_name** - Human-readable annotation from the Gene Ontology or EC
  nomenclature.
- **aligned** - Boolean indicating whether the query was successfully aligned
  to a database structure (`True`/`False`).
- **target_id** - Identifier of the matched database entry (e.g., `3al6_D` for
  PDB chain). Empty if no hit was found.
- **db_name** - Name of the database where the match was found
  (e.g., `pdb100_230517`, `afdb_swissprot_v4`).
- **query_identity** - Sequence identity percentage between query and hit
  (0.0-1.0 scale). Empty if no hit was found.
- **query_coverage** - Proportion of query sequence covered by the alignment
  (0.0-1.0 scale).
- **target_coverage** - Proportion of target sequence covered by the alignment
  (0.0-1.0 scale).

## ⚙️Features

### 1. Prediction modes

The GO ontology contains three subontologies, defined by their root nodes:

- Molecular Function (MF)
- Biological Process (BP)
- Cellular Component (CC)
- Additionally, Metagenomic-DeepFRI v1.0 can predict Enzyme Commission
  (EC) numbers.

By default, the tool makes predictions in all 4 categories. To select only a
few pass the parameter `-p` or `--processing-modes` few times, i.e.:

```{code-block} bash
mDeepFRI predict-function -i /path/to/protein/sequences \
  -d /path/to/foldcomp/database/ -w /path/to/deepfri/weights/folder \
  -o /output_path -p mf -p bp
```

### 2. Hierarchical database search

Different databases have different levels of evidence. For example, PDB
structures are real experimental structures and are considered highest quality
data. New proteins are first queried against PDB. Computational predictions
differ by quality (e.g., AlphaFold predictions are often more accurate than
ESMFold). You can search multiple databases hierarchically for flexibility.
For example, to search AlphaFold first, then ESMFold, pass the parameter `-d`
or `--databases` multiple times:

```{code-block} bash
mDeepFRI predict-function -i /path/to/protein/sequences \
  -d /path/to/alphafold/database/ -d /path/to/another/esmcomp/database/ \
  -w /path/to/deepfri/weights/folder -o /output_path
```

### 3. Temporary files

The first run of `mDeepFRI` with the database will create temporary files
needed for the pipeline. If you don't want to keep them for the next run, add
flag `--remove-intermediate`.

### 4. Skipping prediction matrices

By default, `mDeepFRI` writes detailed prediction matrix files
(`prediction_matrix_*.tsv`) containing raw scores for every protein × GO term
combination. These files can be very large (>50MB each). If you only need the
final `results.tsv` file and want to save disk space, use the `--skip-matrix`
flag:

```{code-block} bash
mDeepFRI predict-function -i /path/to/protein/sequences \
  -d /path/to/foldcomp/database/ -w /path/to/deepfri/weights/folder \
  -o /output_path --skip-matrix
```

### 4b. Raw PyOpal alignments

Use `--save-raw-alignments` to export all pairwise PyOpal alignments into a
single file `{output_path}/raw_alignments.fasta` in this format (one block per
query, concatenated):

```text
>query_id|target=target_id|identity=0.8589|coverage=1.0000|score=2227
QUERYSEQUENCE
>target_id|query=query_id|identity=0.8589|coverage=1.0000|score=2227
TARGETSEQUENCE
#alignment_string: MMXMMMM...
```

### 5. Carved PDB structures

Use `--carve-pdbs` to write query-aligned PDB files under
`{output_path}/carved_pdbs/{query_id}.pdb`. Each file is carved from the
template structure selected by MMseqs2/PyOpal alignment (or from
`--custom-mapping`):

- **SEQRES** always reflects the full query sequence.
- **ATOM** records are written only for query residues mapped to template
  coordinates (query insertions appear in SEQRES only).
- **Residue numbers** follow the query sequence (1-based).
- **Backbone only:** N, CA, C, and O coordinates are transferred from the
  template. Residue names follow the query sequence.
- **Side chains** are rebuilt with [PIPPack](https://github.com/Kuhlman-Lab/PIPPack)
  after carving. Packing is always part of `--carve-pdbs`.
- Templates with CA-only / incomplete backbone residues are still carved
  (available atoms kept) with a warning; packing is **skipped** and the
  partial backbone PDB is kept.

This is different from `--save-structures`, which saves the raw template
structures used as alignment targets.

#### PIPPack setup (required for `--carve-pdbs`)

PIPPack is an external dependency (not installed into the mDeepFRI env):

1. Clone PIPPack and create its conda env from `env/pippack_env.yaml`.
2. Ensure `model_weights/` is present in the checkout.
3. Pass `--pippack-dir /path/to/PIPPack` or set `PIPPACK_DIR`.
4. Pass `--pippack-python /path/to/pippack/env/bin/python` (or set
   `PIPPACK_PYTHON`). Without this, workers may pick the mDeepFRI/pixi
   interpreter, which does **not** have PyTorch and packing will fail.

Packing runs many **single-threaded** worker processes (best CPU throughput
in our benchmarks):

- `--pippack-device cpu` (default): worker count defaults to `--threads`
- `--pippack-device gpu`: worker count defaults to 1 (override with
  `--pippack-workers`)
- `--pippack-model` selects the checkpoint (default `pippack_model_1`)

Inference knobs (PIPPack defaults; safe to leave unchanged):

| Flag | Default | Meaning |
|------|---------|---------|
| `--pippack-n-recycle` | `3` | Recycling iterations |
| `--pippack-temperature` | `0.0` | Chi sampling temperature (`0` = argmax) |
| `--pippack-use-resample` | off | Post-hoc clash/proline steric adjustment |
| `--pippack-seed` | `42` | RNG seed for packing workers |

**Note:** post-hoc steric clash adjustment (`--pippack-use-resample`) is **not**
enabled by default. It can improve packing quality but is slower; enable it
explicitly when you want PIPPack's resampling pass.

If unknown residues "X" are present in the query sequence, normally PiPPack would drop them.
In Metagenomic-DeepFRI those are stashed and reintroduced after side-chain packing
as backbone only positions. 

Please also note, that due to unfortunate atom masking in PIPPack, CA coordinates summing to 0
are dropped - this sometimes happens by chance. The offending code comes from inference.py:
`atom_mask = (np.sum(atom_positions, axis=-1) != 0.0)`
You can apply a fix changing this line to:
`atom_mask = atom37_atom_exists[i].detach().cpu().numpy().astype(np.int32)`

Carving itself also respects `--threads`. Use `--compress-structures` to
archive carved/packed PDBs as `{output_path}/carved_pdbs.tar.gz`. Individual
PDB files are deleted after compression succeeds. (FoldComp is not used for
carved outputs: discontinuous residue numbering from alignment gaps breaks
FoldComp decompress.)

To run alignment and carving only (no DeepFRI inference), use
`--skip-prediction` or pass `-p none`. Weights are not required in that mode.

```{code-block} bash
mDeepFRI predict-function -i /path/to/protein/sequences \
  --custom-mapping /path/to/mapping.tsv \
  -o /output_path --carve-pdbs --skip-prediction --threads 8 \
  --pippack-dir /path/to/PIPPack \
  --pippack-python /path/to/pippack/env/bin/python
```

```{code-block} bash
mDeepFRI predict-function -i /path/to/protein/sequences \
  -d /path/to/foldcomp/database/ -w /path/to/deepfri/weights/folder \
  -o /output_path --carve-pdbs --compress-structures --threads 8 \
  --pippack-dir /path/to/PIPPack --pippack-device gpu
```

#### GPU packing tip

For large batches on GPU, start with `--pippack-device gpu --pippack-workers 1`.
On multi-GPU hosts you can raise `--pippack-workers` and assign devices via
`CUDA_VISIBLE_DEVICES` outside mDeepFRI.

### 6. CPU / GPU utilization

If argument `threads` is provided, the app will parallelize certain steps
(alignment, contact map alignment, PDB carving, PIPPack side-chain packing on
CPU, structure compression, and functional annotation). GPU is often used
to speed up neural networks. By default, `mDeepFRI` runs on CPU. During
prediction it tries the CUDA execution provider first and falls back to CPU
if CUDA is unavailable. PIPPack packing can use `--pippack-device gpu`
independently of DeepFRI prediction.

**Technical tip:** Single instance of DeepFRI on GPU requires 2GB VRAM.

| Install | When to use |
| --- | --- |
| `pip install mdeepfri` | Default. Works without CUDA. |
| `pip install mdeepfri[gpu]` | Linux x86_64 with NVIDIA GPU; bundles CUDA 12 libs (ORT 1.26). |

**Troubleshooting GPU usage:**
If CUDA libraries are not found at runtime, you might see:

```bash
[W:onnxruntime:Default, onnxruntime_pybind_state.cc:1013 CreateExecutionProviderFactoryInstance]
Failed to create CUDAExecutionProvider. Require cuDNN 9.* and CUDA 12.*.
```

Try `pip install mdeepfri[gpu]` first. If the error persists, add library
paths to `LD_LIBRARY_PATH`:

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(python -c 'import os, nvidia.cudnn, nvidia.cublas, nvidia.cuda_runtime; libs=[nvidia.cudnn, nvidia.cublas, nvidia.cuda_runtime]; print(":".join([os.path.join(m.__path__[0], "lib") for m in libs]))')
```

## 🔖 Citations

Metagenomic-DeepFRI is a scientific software. If you use it
in an academic work, please cite the papers behind it:

- Gligorijević et al. "Structure-based protein function prediction using
  graph convolutional networks" Nat. Comms. (2021).
  <https://doi.org/10.1038/s41467-021-23303-9>
- Steinegger & Söding "MMseqs2 enables sensitive protein
  sequence searching for the analysis of massive data sets"
  Nat. Biotechnol. (2017) <https://doi.org/10.1038/nbt.3988>
- Kim, Midrita & Steinegger "Foldcomp: a library and format for compressing
  and indexing large protein structure sets" Bioinformatics (2023)
  <https://doi.org/10.1093/bioinformatics/btad153>
- Maranga et al. "Comprehensive Functional Annotation of Metagenomes and
  Microbial Genomes Using a Deep Learning-Based Method" mSystems (2023)
  <https://doi.org/10.1128/msystems.01178-22>
- Szczerbiak P et al. "Large protein databases reveal structural complementarity
  and functional locality". Nat. Commun. (2025).
  <https://doi.org/10.1038/s41467-025-63250-3>

## 💭 Feedback

### ⚠️ Issue Tracker

Found a bug? Have an enhancement request? Head over to the
[GitHub issue tracker](https://github.com/bioinf-mcb/Metagenomic-DeepFRI/issues)
if you need to report or ask something. If you are filing a bug, please
include as much information as you can about the issue, and try to recreate
the same bug in a simple, easily reproducible situation.

### 🏗️ Contributing

Contributions are more than welcome! See
[`CONTRIBUTING.md`](https://github.com/bioinf-mcb/Metagenomic-DeepFRI/blob/main/CONTRIBUTING.md)
for more details.

## 📋 Changelog

This project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html)
and provides a [changelog](https://github.com/bioinf-mcb/Metagenomic-DeepFRI/blob/main/CONTRIBUTING.md)
in the [Keep a Changelog](http://keepachangelog.com/en/1.0.0/) format.

## ⚖️ License

This library is provided under the GPL-3.0 license.
