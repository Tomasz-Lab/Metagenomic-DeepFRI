<!-- markdownlint-disable-file MD024 MD013 -->
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

[Unreleased]: https://github.com/bioinf-MCB/Metagenomic-DeepFRI/compare/v1.1.10...HEAD

### Added

- GO term mapping to COG categories, SuperCOGs and Information Content (IC) in output results.
- `go2cog_USECLO_ALL.tsv` mapping file in assets from [GO2COG](https://github.com/Tomasz-Lab/go2cog)

## [1.1.10] - 2026-01-16

[1.1.10]: https://github.com/bioinf-MCB/Metagenomic-DeepFRI/compare/v1.1.9...v1.1.10

### Fixed

- error with processing `UniProt` headers ([#101](https://github.com/bioinf-MCB/Metagenomic-DeepFRI/issues/101))
- error for no alignment result with `MMseqs`
- critical error for selenoproteins is now an `info` ([#102](https://github.com/bioinf-MCB/Metagenomic-DeepFRI/issues/102))
- uppercase sequences before `pyopal` to avoid alphabet errors

## [1.1.9] - 2026-01-06

[1.1.9]: https://github.com/bioinf-MCB/Metagenomic-DeepFRI/compare/v1.1.8...v1.1.9

### Added

- `--alignment-min-coverage` parameter to filter low coverage alignments
- `generate-config` CLI command - generate config file in case models are downloaded manually
- `calculate-maps` CLI command - calculates all contact maps from PDB/MMCIF in directory ([#100](https://github.com/bioinf-MCB/Metagenomic-DeepFRI/issues/100))
- prediction matrices - predicted vectors allow to compare proteins with the help of distance analysis ([#79](https://github.com/bioinf-MCB/Metagenomic-DeepFRI/issues/79))
- custom scoring matrices - default `VTML80`
- support for python < 3.13

### Fixed

- Skipping extraction of broken structures from `highquality_clust30`,
which caused segmentation fault (<https://github.com/steineggerlab/foldcomp/issues/56>)
- all output from the software log to `stdout`
- `np.genfromtext` bug ([#96](https://github.com/bioinf-MCB/Metagenomic-DeepFRI/issues/96))
- contact map wrapping ([#99](https://github.com/bioinf-MCB/Metagenomic-DeepFRI/issues/99))
- query file losing sequences ([#92](https://github.com/bioinf-MCB/Metagenomic-DeepFRI/issues/92))
- `MMseqs` convertalis unlimited thread usage

## [1.1.8] - 2024-06-26

[1.1.8]: https://github.com/bioinf-MCB/Metagenomic-DeepFRI/compare/v1.1.7...v1.1.8

### Added

- fixed tests and added to CI/CD
- added function to download MMseqs2 binaries for full CI/CD runs
- removed Conda install - fully installable from PyPI

### Fixed

- hetero atom bug introduced by Ruff (c044a158)
- MacOS CI/CD bug connected to Foldcomp binary

## [1.1.7] - 2024-06-10

[1.1.7]: https://github.com/bioinf-MCB/Metagenomic-DeepFRI/compare/v1.1.6...v1.1.7

### Fixed

- resolved old naming issues ([#84](https://github.com/bioinf-MCB/Metagenomic-DeepFRI/issues/84))

## [1.1.6] - 2024-06-04

[1.1.6]: https://github.com/bioinf-MCB/Metagenomic-DeepFRI/compare/v1.1.5...v1.1.6

### Added

- MMseqs2 API + tests
- rudimentary documentation website
- fixed bugs of v1.1.5

## [1.1.5] - 2024-04-01

[1.1.5]: https://github.com/bioinf-MCB/Metagenomic-DeepFRI/compare/v1.1.4...v1.1.5

### Added

- improved logging
- support for ESM databases ([#80](https://github.com/bioinf-MCB/Metagenomic-DeepFRI/issues/80))
- corrected structures retrieval from PDB ([#81](https://github.com/bioinf-MCB/Metagenomic-DeepFRI/issues/81), [#83](https://github.com/bioinf-MCB/Metagenomic-DeepFRI/issues/83))
  - correct non-standard amino acids
  - remove base pairs

### Changed

- retrieval of structures from PDB & FoldComp databases
- optimized contact map alignment via Cython -
improved search in large databases (ESM)

## [v1.1.4] - 2024-03-23

[v1.1.4]: https://github.com/bioinf-MCB/Metagenomic-DeepFRI/compare/v1.1.3...v1.1.4

### Added

- `--skip-pdb` flag in case of unexpected PDB bugs
- user parameter for allowed protein lengths `--min-length` and `--max-length`

### Fixes

- installation of PDB database
- alignment of faulty PDB structures
  - cases are reported via `logging`, require case-by-case investigation
  - faulty structures are then aligned to a predicted database, which are consistent
  - around 130/1300 alignments to PDB fail (10%)

## [v1.1.3] - 2024-03-13

[v1.1.3]: https://github.com/bioinf-MCB/Metagenomic-DeepFRI/compare/v1.1.2...v1.1.3

### Fixes

- bug where MMseqs2 returned empty results search [#73](https://github.com/bioinf-mcb/Metagenomic-DeepFRI/issues/73)

## [v1.1.2] - 2024-02-22

[v1.1.2]: https://github.com/bioinf-MCB/Metagenomic-DeepFRI/compare/v1.1.1...v1.1.2

### Added

- chunked loading of MMseqs2 preselected targets - improves runtime

### Fixed

- C++17 lowered to C++14 for compatibility
- pyopal API v0.5 integrated
- removed `scipy` distance calculation with function
- replaced `biopython` with `biotite` for structure manipulation
better runtime and easier deploy
- tests fixed & added to CI/CD

## [v1.1.1] - 2024-03-01

[v1.1.1]: https://github.com/bioinf-MCB/Metagenomic-DeepFRI/compare/v1.0.0...v1.1.1

### Added

- Automatic setup of the PDB database during the first run
- Automatic parsing of both mmCIF and PDB files
- Removed `torch` dependency - significant improvement in startup times
- added models of v1.1 - available for the download via `get-models` subcommand
- CI/CD
