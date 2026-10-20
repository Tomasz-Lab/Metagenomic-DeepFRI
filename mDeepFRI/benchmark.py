"""
Benchmarking of structure retrieval, contact map transfer and prediction.

Every database is searched once with every query. For each identity bin one
MMseqs2 hit is kept per query and database, aligned globally and used as a
structural template. Contact maps are built for each ``generate_contacts``
value and, if weights are given, GO terms are predicted.

Output layout::

    database_search/<db>_results.tsv     all MMseqs2 hits
    <bin>/hits/<db>.tsv                  selected hit per query
    <bin>/alignments/<db>.tsv            global alignments of the hits
    <bin>/structures/<db>/               template structures (optional)
    <bin>/generate_contacts_<gc>/        templates, contact maps, predictions
    timing.tsv                           wall time of each stage
"""

import csv
import dataclasses
import logging
import re
import time
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from typing import Iterable, List, Literal, Optional, Tuple

import numpy as np
import numpy.lib.recfunctions as rfn

from mDeepFRI.alignment import AlignmentResult
from mDeepFRI.mmseqs import MMSEQS_COLUMNS, MMseqsResult, QueryFile
from mDeepFRI.pipeline import (_initialize_processing_modes,
                               align_database_hits, build_contact_maps,
                               load_databases, predict_from_contact_maps,
                               save_contact_maps, select_templates,
                               write_alignment_summary)
from mDeepFRI.utils import load_deepfri_config

logger = logging.getLogger(__name__)

ALIGNMENTS_HEADER = [
    "query_id", "target_id", "query_identity", "query_coverage",
    "target_coverage", "query_alignment", "target_alignment"
]
TIMING_HEADER = [
    "stage", "identity_bin", "database", "generate_contacts", "seconds"
]
MODEL_VERSION = re.compile(r"-model_v\d+$")


def is_self_hit(query: str, target: str) -> bool:
    """
    Check whether a hit is the query itself, ignoring AlphaFold model versions.

    Args:
        query (str): Query identifier, e.g. `P12345` or `AF-P12345-F1-model_v6`.
        target (str): Target identifier, e.g. `AF-P12345-F1-model_v4`.

    Returns:
        bool: True for a self-hit.
    """
    return (MODEL_VERSION.sub("", query) == MODEL_VERSION.sub("", target)
            or target.startswith(f"AF-{query}-"))


def select_hits(hits: np.ndarray,
                identity_bin: Tuple[float, float],
                selection: Literal["bitscore", "random"] = "bitscore",
                drop_self_hits: bool = True,
                seed: int = 0) -> np.ndarray:
    """
    Keep one MMseqs2 hit per query within an identity bin.

    Args:
        hits (np.ndarray): MMseqs2 hits.
        identity_bin (Tuple[float, float]): Inclusive lower and exclusive
            upper bound of the MMseqs2 identity.
        selection (str, optional): Keep the hit with the highest bitscore,
            ties broken by identity, or a random hit. Defaults to "bitscore".
        drop_self_hits (bool, optional): Discard hits to the query itself.
            Defaults to True.
        seed (int, optional): Random seed for random selection.
            Defaults to 0.

    Returns:
        np.ndarray: One hit per query, sorted by query.
    """
    low, high = identity_bin
    candidates = hits[(hits["fident"] >= low) & (hits["fident"] < high)]
    if drop_self_hits:
        self_hits = [
            is_self_hit(query, target)
            for query, target in zip(candidates["query"], candidates["target"])
        ]
        candidates = candidates[~np.array(self_hits, dtype=bool)]

    if selection == "random":
        order = np.argsort(candidates["query"], kind="stable")
        _, starts, counts = np.unique(candidates["query"][order],
                                      return_index=True,
                                      return_counts=True)
        rng = np.random.default_rng(seed)
        picks = [
            start + rng.integers(0, count)
            for start, count in zip(starts, counts)
        ]
        return candidates[order[picks]]

    order = np.argsort(candidates, order=["query", "bits", "fident"])[::-1]
    _, first = np.unique(candidates["query"][order], return_index=True)
    return candidates[order[first]]


def write_alignments(path: Path, alignments: List[AlignmentResult]) -> None:
    """
    Write global alignments with their gapped sequences.

    Args:
        path (Path): Output TSV file.
        alignments (List[AlignmentResult]): Alignments to write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(ALIGNMENTS_HEADER)
        for aln in alignments:
            writer.writerow([
                aln.query_name, aln.target_name, aln.query_identity,
                aln.query_coverage, aln.target_coverage, aln.gapped_sequence,
                aln.gapped_target
            ])


@contextmanager
def _timed(timing_file: Path,
           stage: str,
           identity_bin: Optional[str] = None,
           database: Optional[str] = None,
           generate_contacts: Optional[int] = None):
    start = time.perf_counter()
    yield
    seconds = round(time.perf_counter() - start, 2)
    logger.info("%s finished in %.2f s.", stage, seconds)
    with open(timing_file, "a", encoding="utf-8", newline="") as f:
        csv.writer(f, delimiter="\t").writerow(
            [stage, identity_bin, database, generate_contacts, seconds])


def run_benchmark(query_file: QueryFile,
                  output_path: Path,
                  databases: Iterable[str] = (),
                  identity_bins: Iterable[Tuple[float,
                                                float]] = ((0.0, 1.01), ),
                  generate_contacts: Iterable[int] = (2, ),
                  weights: Optional[str] = None,
                  deepfri_processing_modes: Iterable[str] = ("bp", "cc", "ec",
                                                             "mf"),
                  search_results: Optional[Path] = None,
                  skip_pdb: bool = False,
                  overwrite: bool = False,
                  mmseqs_sensitivity: float = 5.7,
                  min_bits: float = 0,
                  max_eval: float = 1e-3,
                  min_ident: float = 0.0,
                  min_coverage: float = 0.9,
                  selection: Literal["bitscore", "random"] = "bitscore",
                  drop_self_hits: bool = True,
                  seed: int = 0,
                  alignment_gap_open: int = 10,
                  alignment_gap_extend: int = 1,
                  scoring_matrix: str = "VTML80",
                  alignment_min_identity: float = 0.0,
                  alignment_min_coverage: float = 0.0,
                  angstrom_contact_threshold: float = 6,
                  save_structures: bool = False,
                  save_cmaps: bool = False,
                  skip_matrix: bool = False,
                  tmpdir: Optional[Path] = None,
                  threads: int = 1) -> None:
    """
    Run the pipeline for every combination of identity bin and
    `generate_contacts` value, reusing one database search.

    Args:
        query_file (QueryFile): Query sequences.
        output_path (Path): Output directory.
        databases (Iterable[str], optional): FoldComp databases searched
            after PDB100, in order of preference.
        identity_bins (Iterable[Tuple[float, float]], optional): Bounds of
            the MMseqs2 identity, lower inclusive and upper exclusive.
            Defaults to all hits.
        generate_contacts (Iterable[int], optional): Numbers of neighbours
            connected to residues inserted in the query. Defaults to (2,).
        weights (str, optional): DeepFRI weights folder. Without weights,
            only contact maps are built.
        deepfri_processing_modes (Iterable[str], optional): Modes to predict.
        search_results (Path, optional): Directory with `<db>_results.tsv`
            files of a previous search, used instead of searching.
        skip_pdb (bool, optional): Skip PDB100. Defaults to False.
        overwrite (bool, optional): Rebuild database files. Defaults to False.
        mmseqs_sensitivity (float, optional): MMseqs2 sensitivity.
        min_bits (float, optional): Minimum MMseqs2 bitscore.
        max_eval (float, optional): Maximum MMseqs2 e-value.
        min_ident (float, optional): Minimum MMseqs2 identity.
        min_coverage (float, optional): Minimum MMseqs2 query and target
            coverage.
        selection (str, optional): Per-query hit selection, see
            `select_hits`. Defaults to "bitscore".
        drop_self_hits (bool, optional): Discard hits to the query itself.
            Defaults to True.
        seed (int, optional): Random seed for random selection.
        alignment_gap_open (int, optional): Gap open penalty.
        alignment_gap_extend (int, optional): Gap extension penalty.
        scoring_matrix (str, optional): Alignment scoring matrix.
        alignment_min_identity (float, optional): Minimum identity of the
            global alignment for a template. Defaults to 0.0.
        alignment_min_coverage (float, optional): Minimum query coverage of
            the global alignment for a template. Defaults to 0.0.
        angstrom_contact_threshold (float, optional): Contact distance
            threshold. Defaults to 6.
        save_structures (bool, optional): Save template structures.
        save_cmaps (bool, optional): Save contact maps.
        skip_matrix (bool, optional): Skip writing prediction matrices.
        tmpdir (Path, optional): Temporary directory for MMseqs2.
        threads (int, optional): Number of threads. Defaults to 1.
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    timing_file = output_path / "timing.tsv"
    timing_file.write_text("\t".join(TIMING_HEADER) + "\n")
    timed = partial(_timed, timing_file)

    if weights:
        deepfri_models_config = load_deepfri_config(weights)
        deepfri_processing_modes = _initialize_processing_modes(
            deepfri_processing_modes, deepfri_models_config)

    dbs = load_databases(databases=databases,
                         skip_pdb=skip_pdb,
                         overwrite=overwrite,
                         threads=threads)

    for db in dbs:
        if search_results:
            db.mmseqs_result = Path(search_results) / f"{db.name}_results.tsv"
            if not db.mmseqs_result.exists():
                raise FileNotFoundError(
                    f"No search results for {db.name}: {db.mmseqs_result}")
            continue

        search_path = output_path / "database_search"
        search_path.mkdir(exist_ok=True)
        db.mmseqs_result = search_path / f"{db.name}_results.tsv"
        with timed("search", database=db.name):
            hits = query_file.search(db.mmseqs_db,
                                     mmseqs_sensitivity=mmseqs_sensitivity,
                                     eval=max_eval,
                                     tmpdir=tmpdir,
                                     threads=threads)
            hits = hits.apply_filters(min_cov=min_coverage,
                                      min_bits=min_bits,
                                      min_ident=min_ident)
            hits.find_best_matches(k=None).save(db.mmseqs_result)

    identity_bins = list(identity_bins)
    bin_names = [
        f"identity_bin_{low:.2f}-{high:.2f}" for low, high in identity_bins
    ]
    for db in dbs:
        with timed("hit_selection", database=db.name):
            hits = MMseqsResult.from_best_matches(db.mmseqs_result).result_arr
            for bin_name, identity_bin in zip(bin_names, identity_bins):
                selected = select_hits(hits,
                                       identity_bin,
                                       selection=selection,
                                       drop_self_hits=drop_self_hits,
                                       seed=seed)
                hits_path = output_path / bin_name / "hits" / f"{db.name}.tsv"
                hits_path.parent.mkdir(parents=True, exist_ok=True)
                MMseqsResult(rfn.repack_fields(selected[MMSEQS_COLUMNS]),
                             query_file.filepath, db.mmseqs_db).save(hits_path)

    for bin_name in bin_names:
        bin_path = output_path / bin_name
        logger.info("Processing %s.", bin_name)
        bin_dbs = [
            dataclasses.replace(db,
                                mmseqs_result=bin_path / "hits" /
                                f"{db.name}.tsv") for db in dbs
        ]

        with timed("alignment", identity_bin=bin_name):
            alignments = align_database_hits(
                bin_dbs,
                alignment_gap_open=alignment_gap_open,
                alignment_gap_extend=alignment_gap_extend,
                scoring_matrix=scoring_matrix,
                threads=threads)
        for db in bin_dbs:
            write_alignments(bin_path / "alignments" / f"{db.name}.tsv",
                             alignments[db.name])

        with timed("templates", identity_bin=bin_name):
            templates = select_templates(
                query_file,
                bin_dbs,
                alignments,
                min_identity=alignment_min_identity,
                min_coverage=alignment_min_coverage,
                structures_path=(bin_path /
                                 "structures" if save_structures else None),
                threads=threads)

        for gc in generate_contacts:
            gc_path = bin_path / f"generate_contacts_{gc}"
            gc_path.mkdir(parents=True, exist_ok=True)
            with timed("contact_maps",
                       identity_bin=bin_name,
                       generate_contacts=gc):
                aligned_cmaps = build_contact_maps(
                    templates,
                    angstrom_contact_threshold=angstrom_contact_threshold,
                    generate_contacts=gc,
                    threads=threads)
            if save_cmaps:
                save_contact_maps(aligned_cmaps, gc_path / "contact_maps")

            if not weights:
                aligned = {aln.query_name for aln, _ in aligned_cmaps}
                write_alignment_summary(
                    gc_path / "alignment_summary.tsv", aligned_cmaps,
                    [q for q in query_file.sequences if q not in aligned])
                continue

            with timed("prediction",
                       identity_bin=bin_name,
                       generate_contacts=gc):
                predict_from_contact_maps(query_file,
                                          aligned_cmaps,
                                          deepfri_models_config,
                                          deepfri_processing_modes,
                                          gc_path,
                                          threads=threads,
                                          skip_matrix=skip_matrix)

    logger.info("Benchmark finished successfully.")
