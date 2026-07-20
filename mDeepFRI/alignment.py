"""
Sequence alignment module using PyOpal for protein alignments.

This module provides utilities for performing pairwise protein sequence alignments
using PyOpal, a Python wrapper for the SIMD-accelerated Opal alignment library.
It handles alignment of query sequences to database hits from MMseqs2 searches
and provides alignment statistics including identity, coverage, and gaps.

The module optimizes alignment by batching sequences and using parallel processing,
making it suitable for aligning large numbers of protein sequences.

Classes:
    AlignmentResult: Container for pairwise alignment results and statistics.

Functions:
    insert_gaps: Insert gaps into sequences based on alignment string.
    _pyopal_align: Internal function for PyOpal alignment.
    align_batch: Align batch of sequences in parallel.
    align_mmseqs_results: Align all MMseqs2 hits to their targets.
"""

import warnings
from functools import partial
from multiprocessing import Pool
from typing import Optional, Tuple

import numpy as np
import pyopal
from scoring_matrices import ScoringMatrix

from mDeepFRI.mmseqs import MMseqsResult
from mDeepFRI.utils import (load_fasta_as_dict, retrieve_fasta_entries_as_dict,
                            stdout_warn)

warnings.showwarning = stdout_warn


def insert_gaps(sequence: str, reference: str,
                alignment_string: str) -> Tuple[str, str]:
    """
    Inserts gaps into query and target sequences.

    Args:
        sequence (str): Query sequence.
        reference (str): Target sequence.
        alignment_string (str): Alignment string.

    Returns:
        gapped_sequence (str): Query sequence with gaps.
        gapped_target (str): Target sequence with gaps.
    """

    sequence_list: list[str] = list(sequence)
    reference_list: list[str] = list(reference)
    alignment_list: list[str] = list(alignment_string)

    for i, a in enumerate(alignment_list):
        if a == "I":
            sequence_list.insert(i, "-")
        elif a == "D":
            reference_list.insert(i, "-")
    return "".join(sequence_list), "".join(reference_list)


class AlignmentResult:
    """
    Container for pairwise protein alignment results and statistics.

    This class stores the results of a protein sequence alignment, including
    the aligned sequences with gaps, alignment statistics (identity, coverage),
    and optional structural information (coordinates, contact maps).

    Attributes:
        query_name (str): Identifier of the query sequence.
        query_sequence (str): Ungapped query protein sequence.
        target_name (str): Identifier of the target (reference) sequence.
        target_sequence (str): Ungapped target protein sequence.
        alignment (str): Alignment string in CIGAR-like format.
            'M' = match/mismatch, 'I' = insertion, 'D' = deletion.
        query_identity (float): Sequence identity as fraction (0.0-1.0) of
            matching residues to alignment length.
        query_coverage (float): Fraction (0.0-1.0) of query sequence covered
            by the alignment.
        target_coverage (float): Fraction (0.0-1.0) of target sequence covered
            by the alignment.
        db_name (str): Name of the database from which target was retrieved.
        gapped_sequence (str): Query sequence with gaps ('-') inserted for alignment.
        gapped_target (str): Target sequence with gaps ('-') inserted for alignment.
        target_coords (np.ndarray, optional): C-alpha atom coordinates from target structure.
        cmap (np.ndarray, optional): Contact map of target structure.
        aligned_cmap (np.ndarray, optional): Contact map aligned to query sequence.

    Example:
        >>> result = AlignmentResult(
        ...     query_name="protein1",
        ...     query_sequence="MSKGEELFT",
        ...     target_name="1GFL_A",
        ...     target_sequence="MSKGEELFTGV",
        ...     alignment="MMMMMMMMMM",
        ...     query_identity=0.90,
        ...     query_coverage=0.82
        ... )
        >>> print(result.gapped_sequence)
        'MSKGEELFT'
    """
    def __init__(self,
                 query_name: str = "",
                 query_sequence: str = "",
                 target_name: str = "",
                 target_sequence: str = "",
                 alignment: str = "",
                 query_identity: Optional[float] = None,
                 query_coverage: Optional[float] = None,
                 target_coverage: Optional[float] = None,
                 db_name: Optional[str] = None,
                 coords: Optional[np.ndarray] = None,
                 structure_path: Optional[str] = None,
                 structure_string: Optional[str] = None,
                 score: Optional[float] = None):

        self.query_name = query_name
        self.query_sequence = query_sequence
        self.target_name = target_name
        self.target_sequence = target_sequence
        self.alignment = alignment
        self.query_identity = query_identity
        self.query_coverage = query_coverage
        self.target_coverage = target_coverage
        self.insert_gaps()
        self.db_name = db_name
        self.coords = coords
        self.structure_path = structure_path
        self.structure_string = structure_string
        self.score = score
        self.target_coords = None
        self.cmap = None
        self.aligned_cmap = None

    def __str__(self):
        return f"AlignmentResult(query_name={self.query_name}, target_name={self.target_name}, " \
               f"query_identity={self.query_identity}, query_coverage={self.query_coverage})"

    def __repr__(self):
        return f"AlignmentResult(query_name={self.query_name}, target_name={self.target_name}, " \
               f"query_identity={self.query_identity}, query_coverage={self.query_coverage})"

    def insert_gaps(self):
        """
        Inserts gaps into query and target sequences.

        Returns:
            AlignmentResult: The object with gapped sequences.
        """

        self.gapped_sequence, self.gapped_target = insert_gaps(
            self.query_sequence, self.target_sequence, self.alignment)


def format_raw_alignment_fasta(alignment: AlignmentResult) -> str:
    """
    Format a PyOpal alignment as a two-record FASTA plus alignment string.

    Output layout::

        >query|target=...|identity=...|coverage=...|score=...
        QUERYSEQ
        >target|query=...|identity=...|coverage=...|score=...
        TARGETSEQ
        #alignment_string: MMX...
    """
    identity = float(alignment.query_identity or 0.0)
    coverage = float(alignment.query_coverage or 0.0)
    score = alignment.score
    if score is None:
        score_str = "nan"
    else:
        score_str = str(int(round(float(score))))

    query_header = (
        f">{alignment.query_name}|target={alignment.target_name}"
        f"|identity={identity:.4f}|coverage={coverage:.4f}|score={score_str}")
    target_header = (
        f">{alignment.target_name}|query={alignment.query_name}"
        f"|identity={identity:.4f}|coverage={coverage:.4f}|score={score_str}")
    return (
        f"{query_header}\n{alignment.query_sequence}\n"
        f"{target_header}\n{alignment.target_sequence}\n"
        f"#alignment_string: {alignment.alignment}\n")


def write_raw_alignments(
        alignments: list[AlignmentResult],
        output_dir,
) -> int:
    """
    Write one raw-alignment FASTA file per query under ``output_dir``.

    Returns:
        Number of files written.
    """
    from pathlib import Path

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = 0
    for alignment in alignments:
        path = out / f"{alignment.query_name}.fasta"
        path.write_text(format_raw_alignment_fasta(alignment), encoding="utf-8")
        written += 1
    return written


def _uppercase_sequence(seq: str) -> str:
    """Return an uppercased sequence (safe for None)."""
    return seq.upper() if seq else seq


def _uppercase_sequences_dict(seqs: dict[str, str]) -> dict[str, str]:
    """Return a new dict with all sequence values uppercased."""
    return {k: _uppercase_sequence(v) for k, v in seqs.items()}


def best_hit_database(query,
                      target_sequences,
                      gap_open: int = 10,
                      gap_extend: int = 1,
                      scoring_matrix: str = "VTML80"):
    """
    Find the best hit in the database and return index.
    """
    query = _uppercase_sequence(query)
    target_sequences = _uppercase_sequences_dict(target_sequences)

    custom_scoring = ScoringMatrix.from_name(scoring_matrix)
    custom_alphabet = custom_scoring.alphabet
    aligner = pyopal.Aligner(scoring_matrix=custom_scoring,
                             gap_open=gap_open,
                             gap_extend=gap_extend)
    target_database = pyopal.Database(target_sequences.values(),
                                      alphabet=custom_alphabet)

    # Retrieve the best hit
    results = aligner.align(query,
                            target_database,
                            mode="score",
                            overflow="buckets",
                            algorithm="nw")
    best_hit = max(results, key=lambda x: x.score)
    best_index = best_hit.target_index
    best_idx = list(target_sequences.keys())[best_index]
    best_seq = target_sequences[best_idx]

    return best_idx, best_seq


def align_pairwise(query,
                   target,
                   gap_open: int = 10,
                   gap_extend: int = 1,
                   scoring_matrix: str = "VTML80"):
    """
    Aligns the query against the target and returns the alignment.
    """
    query = _uppercase_sequence(query)
    target = _uppercase_sequence(target)

    custom_scoring = ScoringMatrix.from_name(scoring_matrix)
    custom_alphabet = custom_scoring.alphabet
    aligner = pyopal.Aligner(scoring_matrix=custom_scoring,
                             gap_open=gap_open,
                             gap_extend=gap_extend)
    database = pyopal.Database([target], alphabet=custom_alphabet)
    # Align the sequences
    alignment = aligner.align(query, database, algorithm="nw", mode="full")
    alignment_string = alignment[0].alignment
    identity = alignment[0].identity()
    query_coverage = alignment[0].coverage(reference="query")
    target_coverage = alignment[0].coverage(reference="target")
    score = float(alignment[0].score)

    return alignment_string, identity, query_coverage, target_coverage, score


def pairwise_against_database(query_id,
                              query_sequence,
                              target_sequences,
                              gap_open: int = 10,
                              gap_extend: int = 1,
                              scoring_matrix: str = "VTML80"):
    """
    Finds the best alignment of the query against the target.
    """
    query_sequence = _uppercase_sequence(query_sequence)

    best_idx, best_target = best_hit_database(query_sequence, target_sequences,
                                              gap_open, gap_extend,
                                              scoring_matrix)

    # align the query against the best hit
    alignment, identity, query_coverage, target_coverage, score = align_pairwise(
        query_sequence, best_target, gap_open, gap_extend, scoring_matrix)
    # create an alignment object
    alignment_result = AlignmentResult(query_id,
                                       query_sequence,
                                       best_idx,
                                       best_target,
                                       alignment,
                                       identity,
                                       query_coverage=query_coverage,
                                       target_coverage=target_coverage,
                                       score=score)
    return alignment_result


def create_partial_database(top_matches, target_sequences):
    """
    Create a partial database from the target sequences.

    Args:

    """

    partial_db = {k: target_sequences[k] for k in top_matches}

    return partial_db


def align_mmseqs_results(best_matches_filepath: str,
                         sequence_db: str,
                         alignment_gap_open: int = 10,
                         alignment_gap_extend: int = 1,
                         threads: int = 1,
                         scoring_matrix: str = "VTML80"):
    """
    Aligns MMseqs2 search results sequence-wise.
    """
    best_matches = MMseqsResult.from_best_matches(best_matches_filepath)
    # check if there are any best matches
    if best_matches.size == 0:
        return []

    query_dict = load_fasta_as_dict(best_matches.query_fasta)
    query_dict = _uppercase_sequences_dict(query_dict)

    # if UniProt header is used, extract second part as sequence ID
    for qid in list(query_dict.keys()):
        if "|" in qid:
            new_qid = qid.split("|")[1]
            query_dict[new_qid] = query_dict.pop(qid)

    unique_queries = {
        query: best_matches.get_query_targets(query)
        for query in best_matches.get_queries()
    }

    target_ids = best_matches.get_targets()
    target_seqs = retrieve_fasta_entries_as_dict(sequence_db, target_ids)
    target_seqs = _uppercase_sequences_dict(target_seqs)

    # create partial databases
    partial_databases = {
        k: create_partial_database(v, target_seqs)
        for k, v in unique_queries.items()
    }

    align_partial = partial(pairwise_against_database,
                            gap_open=alignment_gap_open,
                            gap_extend=alignment_gap_extend,
                            scoring_matrix=scoring_matrix)

    query_ids = list(unique_queries.keys())
    query_sequences = [query_dict[qid] for qid in query_ids]
    target_sequences = [partial_databases[qid] for qid in query_ids]

    # align the query against the best hit
    with Pool(threads) as pool:
        alignments = pool.starmap(
            align_partial, [(query_id, query_sequence, target_sequence)
                            for query_id, query_sequence, target_sequence in
                            zip(query_ids, query_sequences, target_sequences)])

    return alignments
