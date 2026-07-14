"""
Pipeline module for protein function prediction using DeepFRI.

This module orchestrates the complete Metagenomic-DeepFRI pipeline, including:
1. Hierarchical database searches using MMseqs2
2. Alignment of query sequences to database hits using PyOpal
3. Contact map alignment for structure-based predictions
4. DeepFRI-based functional annotation

The pipeline can process proteins with or without structural information,
using Graph Convolutional Networks (GCN) when structures are available,
and Convolutional Neural Networks (CNN) when only sequences are available.

Attributes:
    ALIGNMENT_HEADER (list): Column names for alignment results TSV file.
    FINAL_OUTPUT_HEADER (list): Column names for final prediction results.
    NAN_ALIGNMENT_INFO (list): Default values for missing alignment information.
"""

import csv
import datetime
import io
import logging
import multiprocessing
import pathlib
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from mDeepFRI import DEEPFRI_MODES
from mDeepFRI.alignment import (AlignmentResult, align_mmseqs_results,
                                align_pairwise)
from mDeepFRI.bio_utils import (build_align_contact_map,
                                compress_carved_structures,
                                extract_residue_sequence,
                                extract_residues_coordinates,
                                write_carved_pdbs, _init_parallel_worker)
from mDeepFRI.database import Database, build_database
from mDeepFRI.mmseqs import MMseqsResult, QueryFile
from mDeepFRI.pdb import create_pdb_mmseqs, extract_calpha_coords
from mDeepFRI.predict import Predictor
from mDeepFRI.utils import (get_json_values, load_deepfri_config,
                            remove_intermediate_files)

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.propagate = False
formatter = logging.Formatter(
    '[%(asctime)s] %(module)s.%(funcName)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

ALIGNMENT_HEADER = [
    "query_id", "aligned", "target_id", "db_name", "query_identity",
    "query_coverage", "target_coverage"
]
FINAL_OUTPUT_HEADER = [
    "protein", "network_type", "prediction_mode", "go_term", "score",
    "go_name", "aligned", "target_id", "db_name", "query_identity",
    "query_coverage", "target_coverage", "ic", "cogs", "supercogs"
]

NAN_ALIGNMENT_INFO = [np.nan] * 6
MISSING_TSV = ""


def _parse_ic(raw) -> Optional[float]:
    """Parse information content from a mapping field."""
    if raw is None:
        return None
    if isinstance(raw, float) and np.isnan(raw):
        return None
    text = str(raw).strip()
    if not text or text.lower() == "nan":
        return None
    return float(text)


def _format_ic(ic: Optional[float]) -> str:
    """Format IC for TSV output (two decimals, empty if missing)."""
    if ic is None:
        return MISSING_TSV
    return f"{ic:.2f}"


def load_go_to_cog(path: pathlib.Path):
    mapping = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for row in reader:
            go_term = row[0]
            cogs = row[1].replace("{", "").replace("}", "").replace("'", "")
            ic = _parse_ic(row[4] if len(row) > 4 else None)
            supercogs = row[5].replace("{", "").replace("}",
                                                        "").replace("'", "")
            mapping[go_term] = (ic, cogs, supercogs)
    return mapping


def load_query_file(query_file: str,
                    min_length: Optional[int] = None,
                    max_length: Optional[int] = None) -> QueryFile:
    """
    Load and filter protein sequences from a FASTA file.

    This function loads protein sequences from a FASTA file and optionally filters
    them based on sequence length constraints.

    Args:
        query_file (str): Path to input FASTA file containing protein sequences.
        min_length (int, optional): Minimum protein length in amino acids.
            Sequences shorter than this will be filtered out. Defaults to None.
        max_length (int, optional): Maximum protein length in amino acids.
            Sequences longer than this will be filtered out. Defaults to None.

    Returns:
        QueryFile: QueryFile object containing loaded and filtered sequences.

    Example:
        >>> qf = load_query_file("proteins.fasta", min_length=30, max_length=5000)
        >>> len(qf.sequences)
        42

    Raises:
        FileNotFoundError: If the query_file does not exist.
    """
    query_file = QueryFile(filepath=query_file)
    query_file.load_sequences()
    removed_seleno = query_file.remove_selenocysteine()
    if removed_seleno:
        logger.info("Removed %d selenoproteins (U residues): %s",
                    len(removed_seleno), ", ".join(removed_seleno))
    # filter out sequences
    if min_length or max_length:
        query_file.filter_sequences(
            lambda x: min_length <= len(x) <= max_length)

    return query_file


def hierarchical_database_search(query_file: QueryFile,
                                 output_path: str,
                                 databases: Iterable[str] = [],
                                 mmseqs_sensitivity: float = 5.7,
                                 min_bits: float = 0,
                                 max_eval: float = 1e-5,
                                 min_ident: float = 0.5,
                                 min_coverage: float = 0.9,
                                 top_k: int = 5,
                                 skip_pdb: bool = False,
                                 overwrite: bool = False,
                                 tmpdir: Optional[str] = None,
                                 threads: int = 1) -> List[Database]:
    """
    Perform hierarchical database searches for protein homologs.

    Searches query sequences against multiple databases in a hierarchical manner,
    starting with PDB100 (unless skipped), followed by user-specified databases.
    Results are filtered and the best matches are retained for structure-based
    annotation.

    Args:
        query_file (QueryFile): Object containing query sequences to search.
        output_path (str): Path to directory for saving search results.
        databases (Iterable[str], optional): List of paths to FoldComp databases
            to search (in order). Common databases include afdb_swissprot,
            esmatlas, etc. Defaults to empty list (only PDB if not skipped).
        mmseqs_sensitivity (float, optional): Sensitivity for MMseqs2 search.
            Range: 1.0-7.5, higher values are more sensitive but slower.
            Defaults to 5.7.
        min_bits (float, optional): Minimum bitscore threshold for hits.
            Defaults to 0.
        max_eval (float, optional): Maximum E-value threshold for hits.
            Defaults to 1e-5.
        min_ident (float, optional): Minimum sequence identity for alignment.
            Range: 0.0-1.0. Defaults to 0.5 (50%).
        min_coverage (float, optional): Minimum query/target coverage.
            Range: 0.0-1.0. Defaults to 0.9 (90%).
        top_k (int, optional): Maximum number of top hits to retain per sequence.
            Defaults to 5.
        skip_pdb (bool, optional): Skip searching against PDB100 database.
            Defaults to False.
        overwrite (bool, optional): Overwrite existing database files.
            Defaults to False.
        tmpdir (str, optional): Temporary directory for intermediate files.
            If None, system default temp directory is used. Defaults to None.
        threads (int, optional): Number of threads for parallel processing.
            Defaults to 1.

    Returns:
        Tuple[Dict, Set]: Tuple containing:
            - Dictionary mapping query IDs to list of alignment information
            - Set of PDB hits (for tracking unique structures)

    Raises:
        FileNotFoundError: If database paths do not exist.
        ValueError: If parameters are out of valid ranges.

    Note:
        The function creates intermediate files including MMseqs2 databases
        and search results. These can be removed after prediction with
        the remove_intermediate_files() function if storage is a concern.

    Example:
        >>> qf = load_query_file("proteins.fasta")
        >>> alignments, pdb_hits = hierarchical_database_search(
        ...     qf,
        ...     output_path="./results",
        ...     databases=["path/to/afdb_swissprot"],
        ...     threads=4
        ... )
    """

    output_path = pathlib.Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # logging variable
    sequence_num_start = len(query_file.sequences)

    for idx, seq in query_file.filtered_out.items():
        logger.info(f"Skipping {idx}; sequence length {len(seq)} aa.")

    dbs = []
    # PDB100 database
    if not skip_pdb:
        logger.info("Creating PDB100 database.")
        pdb100 = create_pdb_mmseqs(threads=threads)
        dbs.append(pdb100)
        logger.info("PDB100 database created.")

    for database in databases:
        database = pathlib.Path(database)
        db = build_database(
            input_path=database,
            output_path=database.parent,
            overwrite=overwrite,
            threads=threads,
        )
        dbs.append(db)

    aligned_total = 0
    pdb_hits = set()

    for db in dbs:
        results = query_file.search(db.mmseqs_db,
                                    mmseqs_sensitivity=mmseqs_sensitivity,
                                    eval=max_eval,
                                    threads=threads,
                                    tmpdir=tmpdir)

        filtered = results.apply_filters(min_cov=min_coverage,
                                         min_bits=min_bits,
                                         min_ident=min_ident)

        try:
            best_matches = filtered.find_best_matches(top_k, threads=threads)
        except ValueError:
            best_matches = MMseqsResult([], results.query_fasta,
                                        db.sequence_db)

        mmseqs_results_path = output_path / f"{db.name}_results.tsv"
        # save intermediate results
        best_matches.save(mmseqs_results_path)
        # store the location of the result for the next step
        db.mmseqs_result = mmseqs_results_path

        # catch error if no matches to database
        # a case from phage proteins
        try:
            all_hits = np.unique(best_matches["query"])
        except IndexError:
            all_hits = np.array([])
        # cover skip_pdb case
        unique_hits = all_hits

        if "pdb100" in db.name:
            pdb_hits.update(all_hits)
        elif not skip_pdb:
            unique_hits = [hit for hit in all_hits if hit not in pdb_hits]

        aligned_db = len(unique_hits)
        aligned_total += aligned_db

        aligned_perc = round(aligned_db / sequence_num_start * 100, 2)
        total_perc = round(aligned_total / sequence_num_start * 100, 2)

        logger.info(f"Aligned {aligned_db}/{sequence_num_start} "
                    f"({aligned_perc:.2f}%) proteins against {db.name}.")
        logger.info(
            f"Aligned {aligned_total}/{sequence_num_start} ({total_perc:.2f}%) proteins in total."
        )

        # this mechanism decreases the amount of sequences
        # on each iteration. Drastically improves execution times
        # for large datasets.
        # PDB100 hits are aligned second time to experimental
        # structures in order to save failed contact map alignemnts.
        if 'pdb100' not in db.name:
            query_file.remove_sequences(all_hits)

    return dbs


def _log_cnn_fallback(query_id: str, reason: str) -> None:
    logger.warning(
        "%s: %s. Falling back to CNN (sequence-only) prediction.",
        query_id,
        reason,
    )


def _detect_mapping_delimiter(header_line: str) -> str:
    tab_count = header_line.count("\t")
    comma_count = header_line.count(",")
    if tab_count > comma_count:
        return "\t"
    return ","


def _resolve_structure_path(structure_ref: str,
                            mapping_path: pathlib.Path) -> pathlib.Path:
    struct_path = pathlib.Path(structure_ref)
    candidates = []

    if struct_path.is_absolute():
        candidates.append(struct_path)
    else:
        candidates.append(mapping_path.parent / struct_path)

    candidates.append(mapping_path.parent / "structures" / struct_path.name)

    if struct_path.suffix == "":
        for ext in (".pdb", ".cif", ".mmcif"):
            candidates.append(
                mapping_path.parent / "structures" / f"{structure_ref}{ext}")

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return candidates[0]


def _load_custom_mapping_file(mapping_path: pathlib.Path) -> Dict[str, str]:
    mapping: Dict[str, str] = {}

    with open(mapping_path, "r", encoding="utf-8") as f:
        header_line = f.readline()
        if not header_line.strip():
            return mapping

        delimiter = _detect_mapping_delimiter(header_line)
        headers = [
            column.strip().lower()
            for column in header_line.strip().split(delimiter)
        ]

        if "structure_path" in headers:
            query_col = headers.index("query") if "query" in headers else 0
            path_col = headers.index("structure_path")
        else:
            query_col = 0
            path_col = 1

        reader = csv.reader(f, delimiter=delimiter)
        for row in reader:
            if len(row) <= max(query_col, path_col):
                logger.warning(f"Skipping malformed mapping line: {row}")
                continue
            protein_id = row[query_col].strip()
            structure_ref = row[path_col].strip()
            mapping[protein_id] = str(
                _resolve_structure_path(structure_ref, mapping_path))

    return mapping


def _align_one_custom_mapping(
        query_id: str,
        query_sequence: str,
        structure_path: str,
        alignment_gap_open: int,
        alignment_gap_extend: int,
        scoring_matrix: str,
        build_contact_maps: bool,
        angstrom_contact_threshold: float,
        generate_contacts: int) -> Tuple[str, Optional[Tuple[AlignmentResult,
                                                              Optional[np.ndarray]]],
                                         Optional[str]]:
    struct_path_obj = pathlib.Path(structure_path)
    if not struct_path_obj.exists():
        return query_id, None, f"structure file not found: {structure_path}"

    try:
        with open(structure_path, "r", encoding="utf-8") as structure_file:
            structure_string = structure_file.read()

        if struct_path_obj.suffix.lower() in {".cif", ".mmcif"}:
            filetype = "mmcif"
        elif struct_path_obj.suffix.lower() == ".pdb":
            filetype = "pdb"
        else:
            filetype = "mmcif"

        if not build_contact_maps:
            target_sequence = extract_residue_sequence(
                structure_string,
                chain="A",
                filetype=filetype,
                structure_path=structure_path)
            coords = None
        else:
            target_sequence, coords = extract_residues_coordinates(
                structure_string,
                chain="A",
                filetype=filetype,
                structure_path=structure_path)
            if target_sequence is None or coords is None:
                return query_id, None, (
                    f"failed to extract structure information from {structure_path}"
                )

        if target_sequence is None:
            return query_id, None, (
                f"failed to extract structure information from {structure_path}")

        alignment_string, identity, query_coverage, target_coverage = align_pairwise(
            query_sequence,
            target_sequence,
            gap_open=alignment_gap_open,
            gap_extend=alignment_gap_extend,
            scoring_matrix=scoring_matrix)

        alignment_result = AlignmentResult(
            query_name=query_id,
            query_sequence=query_sequence,
            target_name=struct_path_obj.stem,
            target_sequence=target_sequence,
            alignment=alignment_string,
            query_identity=identity,
            query_coverage=query_coverage,
            target_coverage=target_coverage,
            db_name="custom_mapping",
            coords=coords,
            structure_path=structure_path)

        if build_contact_maps:
            aligned_cmap = build_align_contact_map(
                alignment_result,
                threshold=angstrom_contact_threshold,
                generated_contacts=generate_contacts)
        else:
            aligned_cmap = (alignment_result, None)

        return query_id, aligned_cmap, None
    except Exception as exc:
        return query_id, None, str(exc)


def _align_custom_mapping_shard(
        work_items: List[Tuple[str, str, str]],
        alignment_gap_open: int,
        alignment_gap_extend: int,
        scoring_matrix: str,
        build_contact_maps: bool,
        angstrom_contact_threshold: float,
        generate_contacts: int,
) -> List[Tuple[str, Optional[Tuple[AlignmentResult, Optional[np.ndarray]]],
                Optional[str]]]:
    return [
        _align_one_custom_mapping(
            query_id,
            query_sequence,
            structure_path,
            alignment_gap_open,
            alignment_gap_extend,
            scoring_matrix,
            build_contact_maps,
            angstrom_contact_threshold,
            generate_contacts,
        )
        for query_id, query_sequence, structure_path in work_items
    ]


def load_custom_alignments_from_mapping(
        mapping_file: str,
        query_file: QueryFile,
        alignment_gap_open: float = 10,
        alignment_gap_extend: float = 1,
        scoring_matrix: str = "VTML80",
        angstrom_contact_threshold: float = 6,
        generate_contacts: int = 2,
        threads: int = 1,
        build_contact_maps: bool = True) -> List[Tuple[AlignmentResult, np.ndarray]]:
    """
    Load custom alignments from a mapping file that associates proteins with structures.

    This function enables bypassing the hierarchical database search by providing pre-computed
    sequence-to-structure mappings. It performs pairwise alignment between query sequences
    and structure sequences, extracts coordinates, and generates contact maps.

    Args:
        mapping_file (str): Path to TSV file with format: protein_id<tab>structure_path
            Skip the header line. Structure paths can be relative or absolute.
            Supported formats: CIF (mmcif), PDB.
        query_file (QueryFile): QueryFile object containing query sequences.
        alignment_gap_open (float, optional): Gap open penalty for sequence alignment.
            Defaults to 10.
        alignment_gap_extend (float, optional): Gap extension penalty for sequence alignment.
            Defaults to 1.
        scoring_matrix (str, optional): Scoring matrix for sequence alignment.
            Defaults to "VTML80".
        angstrom_contact_threshold (float, optional): Distance threshold in Angstroms
            for generating contact maps. Defaults to 6.
        generate_contacts (int, optional): Gap for generating contacts in gapped regions.
            Defaults to 2.
        threads (int, optional): Number of threads for parallel processing.
            Defaults to 1.

    Returns:
        List[Tuple[AlignmentResult, np.ndarray]]: List of tuples containing:
            - AlignmentResult: Pairwise alignment with real metrics (identity, coverage)
            - np.ndarray: Aligned contact map

    Example:
        >>> qf = QueryFile("proteins.faa")
        >>> qf.load_sequences()
        >>> alignments = load_custom_alignments_from_mapping(
        ...     "mapping.tsv",
        ...     qf,
        ...     alignment_gap_open=10,
        ...     alignment_gap_extend=1
        ... )

    Notes:
        - Proteins with a successful mapping are processed with GCN
        - Proteins missing from the mapping file or with unloadable structures
          are logged and predicted with CNN (sequence-only)
        - Structure files must be in CIF (mmcif) or PDB format
    """

    # Load the mapping file
    mapping = {}
    mapping_path = pathlib.Path(mapping_file)

    if not mapping_path.exists():
        raise FileNotFoundError(f"Mapping file not found: {mapping_file}")

    mapping = _load_custom_mapping_file(mapping_path)

    logger.info(f"Loaded {len(mapping)} protein-to-structure mappings.")

    aligned_cmaps = []
    work_items = [
        (query_id, query_sequence, mapping[query_id])
        for query_id, query_sequence in query_file.sequences.items()
        if query_id in mapping
    ]

    for query_id in query_file.sequences:
        if query_id not in mapping:
            _log_cnn_fallback(query_id,
                              "not found in custom mapping file")

    worker_count = max(1, min(threads, len(work_items))) if work_items else 1
    if work_items:
        logger.info(
            "Aligning %d mapped structure(s) using %d parallel worker process(es).",
            len(work_items), worker_count)

    alignment_kwargs = dict(
        alignment_gap_open=int(alignment_gap_open),
        alignment_gap_extend=int(alignment_gap_extend),
        scoring_matrix=scoring_matrix,
        build_contact_maps=build_contact_maps,
        angstrom_contact_threshold=angstrom_contact_threshold,
        generate_contacts=generate_contacts,
    )

    if worker_count == 1:
        shard_results = _align_custom_mapping_shard(work_items,
                                                    **alignment_kwargs)
        for query_id, aligned_cmap, error in shard_results:
            if error is not None:
                _log_cnn_fallback(query_id,
                                  f"error loading structure ({error})")
                continue
            aligned_cmaps.append(aligned_cmap)
    else:
        ctx = multiprocessing.get_context("fork")
        shards = [work_items[i::worker_count] for i in range(worker_count)]
        shards = [shard for shard in shards if shard]
        align_shard = partial(_align_custom_mapping_shard, **alignment_kwargs)
        with ProcessPoolExecutor(max_workers=worker_count,
                                 mp_context=ctx,
                                 initializer=_init_parallel_worker) as executor:
            for shard in executor.map(align_shard, shards):
                for query_id, aligned_cmap, error in shard:
                    if error is not None:
                        _log_cnn_fallback(query_id,
                                          f"error loading structure ({error})")
                        continue
                    aligned_cmaps.append(aligned_cmap)

    logger.info(
        "Loaded %d structure alignment(s) from custom mapping; "
        "%d protein(s) queued for CNN fallback.",
        len(aligned_cmaps),
        len(query_file.sequences) - len(aligned_cmaps),
    )
    return aligned_cmaps


def _initialize_processing_modes(modes: List[str],
                                 config: Dict[str, Any]) -> List[str]:
    """
    Filters processing modes based on the model config version.
    """
    filtered_modes = list(modes)
    # version 1.1 drops support for ec
    if config.get("version") == "1.1":
        if "ec" in filtered_modes:
            filtered_modes.remove("ec")
            logger.info(
                "EC number prediction is not supported in version 1.1.")

    if len(filtered_modes) == 0:
        raise ValueError("No processing modes selected.")
    return filtered_modes


def _build_contact_map_shard(
        shard_alignments: List[AlignmentResult],
        partial_map_align,
) -> List[Tuple[AlignmentResult, np.ndarray]]:
    return [partial_map_align(aln) for aln in shard_alignments]


def _run_prediction_loop(predictor, data_iterable: iter, data_len: int,
                         net_type: str, tsv_writer: csv.writer,
                         description: str):
    """
    A helper function to run a prediction loop for either GCN or CNN.
    """
    BAR_FORMAT = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}], {rate_fmt}{postfix}"
    # Assuming BAR_FORMAT and sys.stdout are available in scope
    # or passed as arguments if this is in a different module.
    for item in tqdm(data_iterable,
                     total=data_len,
                     desc=description,
                     bar_format=BAR_FORMAT,
                     file=sys.stdout,
                     mininterval=10):
        if net_type == "gcn":
            # item is (aln, aligned_cmap)
            aln, aligned_cmap = item
            query_id = aln.query_name
            pred_vector = predictor.forward_pass(seqres=aln.query_sequence,
                                                 cmap=aligned_cmap)
        else:  # net_type == "cnn"
            # item is (query_id, sequence)
            query_id, sequence = item
            pred_vector = predictor.forward_pass(seqres=sequence)

        out_row = [query_id, net_type] + pred_vector.tolist()
        tsv_writer.writerow(out_row)


def predict_protein_function(
        query_file: QueryFile,
        databases: Tuple[Database],
        weights: Optional[str],
        output_path: str,
        deepfri_processing_modes: List[str] = ["ec", "bp", "mf", "cc"],
        angstrom_contact_threshold: float = 6,
        generate_contacts: int = 2,
        alignment_gap_open: float = 10,
        alignment_gap_continuation: float = 1,
        remove_intermediate=False,
        threads: int = 1,
        save_structures: bool = False,
        save_cmaps: bool = False,
        carve_pdbs: bool = False,
        compress_structures: bool = False,
        skip_prediction: bool = False,
        skip_matrix: bool = False,
        scoring_matrix: str = "VTML80",
        command_str: Optional[str] = None,
        version: Optional[str] = None,
        custom_mapping_file: Optional[str] = None,
        propagate_go_terms: bool = False,
        obo_path: Optional[str] = None):
    """
    Predict protein function using DeepFRI.

    This function is the main entry point for the prediction pipeline. It aligns
    query sequences to databases, generates contact maps, and runs DeepFRI
    predictions for specified functional categories.

    Can operate in two modes:
    1. Hierarchical database search mode (default): searches query_file against databases
    2. Custom mapping mode: uses pre-computed sequence-to-structure mappings

    Args:
        query_file (QueryFile): Object containing query sequences.
        databases (Tuple[Database], optional): Tuple of database objects to search against.
            Only used if custom_mapping_file is not provided.
            Defaults to empty tuple.
        weights (str, optional): Path to folder containing DeepFRI model weights.
            Required unless ``skip_prediction`` is True.
        output_path (str): Path to directory for saving results.
        deepfri_processing_modes (List[str], optional): List of modes to predict.
            Options: "ec", "bp", "mf", "cc", or "none" to skip prediction.
            Defaults to ["ec", "bp", "mf", "cc"].
        angstrom_contact_threshold (float, optional): Distance threshold for contact maps.
            Defaults to 6.
        generate_contacts (int, optional): Gap for generating contact maps.
            Defaults to 2.
        alignment_gap_open (float, optional): Gap open penalty for alignment.
            Defaults to 10.
        alignment_gap_continuation (float, optional): Gap extension penalty.
            Defaults to 1.
        remove_intermediate (bool, optional): Remove intermediate files.
            Defaults to False.
        threads (int, optional): Number of threads for parallel processing.
            Defaults to 1.
        save_structures (bool, optional): Save aligned structures to disk.
            Defaults to False.
        save_cmaps (bool, optional): Save generated contact maps to disk.
            Defaults to False.
        carve_pdbs (bool, optional): Write query-aligned PDB files carved from
            template structures using PyOpal alignment. Defaults to False.
        compress_structures (bool, optional): Compress carved PDB files into a
            FoldComp database after carving. Requires ``carve_pdbs``.
            Defaults to False.
        skip_prediction (bool, optional): Run alignment and carving only; skip
            DeepFRI inference. Defaults to False.
        skip_matrix (bool, optional): Skip writing full prediction matrices.
            Defaults to False.
        scoring_matrix (str, optional): Scoring matrix for alignment.
            Defaults to "VTML80".
        command_str (str, optional): The original command-line invocation.
            Defaults to None.
        version (str, optional): The version of mDeepFRI.
            Defaults to None.
        custom_mapping_file (str, optional): Path to TSV file with protein-to-structure mappings.
            Format: protein_id<tab>structure_path (skip header).
            When provided, bypasses hierarchical database search. Proteins with a
            loadable structure mapping are processed with GCN; others use CNN.
            Defaults to None (use databases).
        propagate_go_terms (bool, optional): Propagate GO terms up the
            ontology DAG using the true-path rule (is_a and part_of).
            Downloads go-basic.obo automatically if needed.
            Defaults to False.
        obo_path (str, optional): Path to GO OBO file (go-basic.obo).
            If None and propagate_go_terms is True, the file is downloaded
            to the output directory automatically.

    Returns:
        None: Results are written to files in output_path.

    See Also:
        hierarchical_database_search: For the initial search step.
        load_custom_alignments_from_mapping: For custom mapping implementation.
    """

    skip_prediction = skip_prediction or "none" in deepfri_processing_modes
    deepfri_processing_modes = [
        mode for mode in deepfri_processing_modes if mode != "none"
    ]

    output_path = pathlib.Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    aligned_cmaps = []

    # Use custom mapping if provided, otherwise use hierarchical database search
    if custom_mapping_file:
        logger.info(
            f"Using custom sequence-to-structure mapping from {custom_mapping_file}"
        )
        aligned_cmaps = load_custom_alignments_from_mapping(
            mapping_file=custom_mapping_file,
            query_file=query_file,
            alignment_gap_open=alignment_gap_open,
            alignment_gap_extend=alignment_gap_continuation,
            scoring_matrix=scoring_matrix,
            angstrom_contact_threshold=angstrom_contact_threshold,
            generate_contacts=generate_contacts,
            threads=threads,
            build_contact_maps=not skip_prediction)
    else:
        # Standard hierarchical database search
        for db in databases:
            # SEQUENCE ALIGNMENT
            # calculate already aligned sequences
            alignments = align_mmseqs_results(
                best_matches_filepath=db.mmseqs_result,
                sequence_db=db.sequence_db,
                alignment_gap_open=alignment_gap_open,
                alignment_gap_extend=alignment_gap_continuation,
                threads=threads,
                scoring_matrix=scoring_matrix)

            try:
                # set a db name for alignments
                for aln in alignments:
                    aln.db_name = db.name

                aligned_queries = [aln[0].query_name for aln in aligned_cmaps]
                new_alignments = {
                    aln.query_name: aln
                    for aln in alignments
                    if aln.query_name not in aligned_queries
                    and aln.query_name in query_file.sequences
                }

                # CONTACT MAP ALIGNMENT
                # initially designed as a separate step
                # some protein structures in PDB are not formatted correctly
                # so contact map alignment fails for them
                # for this cases we replace closest experimental structure with
                # closest predicted structure if available
                # if no alignments were found - report

                # remove broken structures
                if db.name == "highquality_clust30":
                    data_path = pathlib.Path(__file__).parent / "assets"
                    # convert to abspath
                    data_path = data_path.resolve()
                    with open(data_path / "highquality_clust30_error_ids.pkl",
                              "rb") as f:
                        error_ids = pickle.load(f)
                    # filter out broken structures
                    new_alignments = {
                        query_name: aln
                        for query_name, aln in new_alignments.items()
                        if aln.target_name not in error_ids
                    }

                query_ids = [aln.query_name for aln in new_alignments.values()]
                target_ids = [
                    aln.target_name.rsplit(".", 1)[0]
                    for aln in new_alignments.values()
                ]

                # extract structural information
                # in form of C-alpha coordinates
                if save_structures:
                    save_dir = output_path / "structures" / db.name
                    save_dir.mkdir(parents=True, exist_ok=True)
                else:
                    save_dir = None

                coords = extract_calpha_coords(db,
                                               target_ids,
                                               query_ids,
                                               save_directory=save_dir,
                                               threads=threads)

                for aln, coord in zip(new_alignments.values(), coords):
                    aln.coords = coord

            # troubleshoot cases where alignments are empty
            except IndexError:
                logger.info("No alignments found for %s.", db.name)
                new_alignments = {}
                continue

            # if new alignments are empty - result is empty as well
            partial_map_align = partial(build_align_contact_map,
                                        threshold=angstrom_contact_threshold,
                                        generated_contacts=generate_contacts)
            alignment_values = list(new_alignments.values())
            contact_worker_count = max(1, min(threads, len(alignment_values)))
            if contact_worker_count == 1:
                cmaps = [partial_map_align(aln) for aln in alignment_values]
            else:
                ctx = multiprocessing.get_context("fork")
                contact_shards = [
                    alignment_values[i::contact_worker_count]
                    for i in range(contact_worker_count)
                ]
                contact_shards = [shard for shard in contact_shards if shard]
                build_shard = partial(_build_contact_map_shard,
                                      partial_map_align=partial_map_align)

                with ProcessPoolExecutor(max_workers=contact_worker_count,
                                         mp_context=ctx,
                                         initializer=_init_parallel_worker) as executor:
                    cmaps = []
                    for shard in executor.map(build_shard, contact_shards):
                        cmaps.extend(shard)

            # filter errored contact maps
            # returned as Tuple[AlignmentResult, None] from `retrieve_align_contact_map`
            partial_cmaps = [cmap for cmap in cmaps if cmap[1] is not None]
            aligned_cmaps.extend(partial_cmaps)
            aligned_database = round(
                len(partial_cmaps) / len(query_file.sequences) * 100, 2)
            aligned_total = round(
                len(aligned_cmaps) / len(query_file.sequences) * 100, 2)
            logger.info(
                f"Aligned {len(partial_cmaps)}/{len(query_file.sequences)} ({aligned_database}%) "
                f"proteins against {db.name} [without length ivalid].")
            logger.info(
                f"Aligned {len(aligned_cmaps)}/{len(query_file.sequences)} ({aligned_total}%) "
                "proteins in total [without length invalid].")

    if save_cmaps:
        cmap_dir = output_path / "contact_maps"
        cmap_dir.mkdir(parents=True, exist_ok=True)
        for i, (aln, cmap) in enumerate(aligned_cmaps):
            cmap_file = cmap_dir / f"{aln.query_name}.npy"
            np.save(cmap_file, cmap)

    aligned_queries = [aln.query_name for aln, _ in aligned_cmaps]
    unaligned_queries = {
        query_id: seq
        for query_id, seq in query_file.sequences.items()
        if query_id not in aligned_queries
    }

    if custom_mapping_file and unaligned_queries:
        logger.info(
            "Predicting %d protein(s) with CNN (sequence-only): %s",
            len(unaligned_queries),
            ", ".join(unaligned_queries.keys()),
        )

    alignment_results_file = output_path / "alignment_summary.tsv"
    with open(alignment_results_file, "w", encoding="utf-8") as aln_output:
        tsv_writer = csv.writer(aln_output, delimiter="\t")
        tsv_writer.writerow(ALIGNMENT_HEADER)
        for aln, _ in aligned_cmaps:
            tsv_writer.writerow([
                aln.query_name, True, aln.target_name, aln.db_name,
                aln.query_identity, aln.query_coverage, aln.target_coverage
            ])
        for query_id in unaligned_queries:
            tsv_writer.writerow(
                [query_id, False, np.nan, np.nan, np.nan, np.nan, np.nan])

    if carve_pdbs:
        carve_dir = output_path / "carved_pdbs"
        carved_count = write_carved_pdbs(aligned_cmaps,
                                         databases,
                                         carve_dir,
                                         threads=threads,
                                         release_contact_maps=skip_prediction)
        if compress_structures:
            if carved_count == 0:
                logger.warning(
                    "No carved PDB files to compress; skipping FoldComp compression."
                )
            else:
                compress_carved_structures(
                    carve_dir,
                    output_path / "carved_pdbs.foldcomp",
                    threads=threads)

    if skip_prediction:
        logger.info("Skipping DeepFRI prediction (--skip-prediction).")
        if remove_intermediate:
            for db in databases:
                remove_intermediate_files([db.sequence_db, db.mmseqs_db])
        logger.info("meta-DeepFRI finished successfully.")
        return

    if not weights:
        raise ValueError(
            "Model weights path is required unless --skip-prediction is set.")

    # load DeepFRI model
    deepfri_models_config = load_deepfri_config(weights)
    deepfri_processing_modes = _initialize_processing_modes(
        deepfri_processing_modes, deepfri_models_config)

    weights = pathlib.Path(weights)

    ### FUNCTION PREDICTION ###
    # sort cmaps by length of query sequence
    aligned_cmaps = sorted(aligned_cmaps,
                           key=lambda x: len(x[0].query_sequence))
    # sort unaligned queries by length
    unaligned_queries = dict(
        sorted(unaligned_queries.items(), key=lambda x: len(x[1])))

    # output_file_name = output_path / "results.tsv"
    # output_buffer = open(output_file_name, "w", encoding="utf-8")
    # csv_writer = csv.writer(output_buffer, delimiter="\t")
    # csv_writer.writerow(OUTPUT_HEADER)

    # Per-mode list of {config_path, matrix_source (Path|StringIO), net_label}
    # v1.1 GCN weights use an expanded GO/EC vocabulary while CNN still uses the
    # older MERGED head sizes; a single TSV cannot mix both. When term lists
    # differ, write separate matrices per network (see assembly loop below).

    matrix_jobs_by_mode: Dict[str, List[Dict[str, Any]]] = {}
    for i, mode in enumerate(deepfri_processing_modes):
        gcn_model_path = deepfri_models_config["gcn"][mode]
        cnn_model_path = deepfri_models_config["cnn"][mode]
        gcn_config_path = gcn_model_path.rsplit(".", 1)[0] + "_model_params.json"
        cnn_config_path = cnn_model_path.rsplit(".", 1)[0] + "_model_params.json"
        goterms_gcn = get_json_values(gcn_config_path, "goterms")
        goterms_cnn = get_json_values(cnn_config_path, "goterms")
        split_matrices = (len(goterms_gcn) != len(goterms_cnn)
                          or goterms_gcn != goterms_cnn)

        if split_matrices:
            logger.info(
                "GCN and CNN use different output vocabularies for mode %s "
                "(%d vs %d labels). Writing separate prediction_matrix_%s_*.tsv "
                "files.", mode, len(goterms_gcn), len(goterms_cnn), mode)

        gcn_prots = len(aligned_cmaps)
        cnn_prots = len(unaligned_queries)
        matrix_jobs_by_mode[mode] = []

        def _open_matrix_sink(filename_suffix: str):
            if skip_matrix:
                buf = io.StringIO()
                return buf, buf
            out_path = output_path / filename_suffix
            fh = open(out_path, "w", encoding="utf-8")
            return fh, out_path

        logger.info("Processing mode: %s; %i/%i", DEEPFRI_MODES[mode], i + 1,
                    len(deepfri_processing_modes))

        if split_matrices:
            if gcn_prots > 0:
                fh, src = _open_matrix_sink(
                    f"prediction_matrix_{mode}_gcn.tsv")
                tsv_writer = csv.writer(fh, delimiter="\t")
                tsv_writer.writerow(["protein", "network_type"] + goterms_gcn)
                gcn_path = deepfri_models_config["gcn"][mode]
                gcn = Predictor(gcn_path, threads=threads)
                _run_prediction_loop(
                    predictor=gcn,
                    data_iterable=aligned_cmaps,
                    data_len=len(aligned_cmaps),
                    net_type="gcn",
                    tsv_writer=tsv_writer,
                    description=f"Predicting with GCN ({DEEPFRI_MODES[mode]})")
                del gcn
                if not skip_matrix:
                    fh.close()
                matrix_jobs_by_mode[mode].append({
                    "config_path": gcn_config_path,
                    "matrix_source": src,
                })

            if cnn_prots > 0:
                fh, src = _open_matrix_sink(
                    f"prediction_matrix_{mode}_cnn.tsv")
                tsv_writer = csv.writer(fh, delimiter="\t")
                tsv_writer.writerow(["protein", "network_type"] + goterms_cnn)
                cnn_path = deepfri_models_config["cnn"][mode]
                cnn = Predictor(cnn_path, threads=threads)
                _run_prediction_loop(
                    predictor=cnn,
                    data_iterable=unaligned_queries.items(),
                    data_len=len(unaligned_queries),
                    net_type="cnn",
                    tsv_writer=tsv_writer,
                    description=f"Predicting with CNN ({DEEPFRI_MODES[mode]})")
                del cnn
                if not skip_matrix:
                    fh.close()
                matrix_jobs_by_mode[mode].append({
                    "config_path": cnn_config_path,
                    "matrix_source": src,
                })
        else:
            fh, src = _open_matrix_sink(f"prediction_matrix_{mode}.tsv")
            tsv_writer = csv.writer(fh, delimiter="\t")
            tsv_writer.writerow(["protein", "network_type"] + goterms_gcn)

            if gcn_prots > 0:
                gcn_path = deepfri_models_config["gcn"][mode]
                gcn = Predictor(gcn_path, threads=threads)
                _run_prediction_loop(
                    predictor=gcn,
                    data_iterable=aligned_cmaps,
                    data_len=len(aligned_cmaps),
                    net_type="gcn",
                    tsv_writer=tsv_writer,
                    description=f"Predicting with GCN ({DEEPFRI_MODES[mode]})")
                del gcn

            if cnn_prots > 0:
                cnn_path = deepfri_models_config["cnn"][mode]
                cnn = Predictor(cnn_path, threads=threads)
                _run_prediction_loop(
                    predictor=cnn,
                    data_iterable=unaligned_queries.items(),
                    data_len=len(unaligned_queries),
                    net_type="cnn",
                    tsv_writer=tsv_writer,
                    description=f"Predicting with CNN ({DEEPFRI_MODES[mode]})")
                del cnn

            if not skip_matrix:
                fh.close()
            matrix_jobs_by_mode[mode].append({
                "config_path": gcn_config_path,
                "matrix_source": src,
            })

    ### FORMAT AND CREATE FINAL OUTPUT FILES ###
    # combine mode-specific matrices into a single file
    # open and load alignment data
    with open(alignment_results_file, "r", encoding="utf-8") as aln_input:
        tsv_reader = csv.reader(aln_input, delimiter="\t")
        next(tsv_reader)  # skip header
        alignment_data = {row[0]: row[1:] for row in tsv_reader}

    go2cog_path = pathlib.Path(
        __file__).parent / "assets" / "go2cog_USECLO_ALL.tsv"
    if go2cog_path.exists():
        go2cog_mapping = load_go_to_cog(go2cog_path)
    else:
        logger.warning(f"GO to COG mapping file not found at {go2cog_path}")
        go2cog_mapping = {}

    final_output = output_path / "results.tsv"
    with open(final_output, "w", encoding="utf-8") as fout:
        # Write metadata at the beginning
        if command_str is not None or version is not None:
            # Write timestamp
            timestamp = datetime.datetime.now().strftime(
                "%a %b %d %H:%M:%S %Y")
            fout.write(f"## {timestamp}\n")
            # Write version
            if version is not None:
                fout.write(f"## mDeepFRI-{version}\n")
            # Write command string
            if command_str is not None:
                fout.write(f"## {command_str}\n")
        fout.write("\t".join(FINAL_OUTPUT_HEADER) + "\n")
        for mode, jobs in matrix_jobs_by_mode.items():
            for job in jobs:
                json_path = job["config_path"]
                matrix_source = job["matrix_source"]
                GONAMES = get_json_values(json_path, "gonames")

                if isinstance(matrix_source, io.StringIO):
                    matrix_source.seek(0)
                    matrix_content = matrix_source.getvalue()
                    matrix_lines = matrix_content.strip().split("\n")
                    tsv_reader = csv.reader(matrix_lines, delimiter="\t")
                else:
                    with open(matrix_source, "r",
                              encoding="utf-8") as matrix_input:
                        tsv_reader = csv.reader(matrix_input, delimiter="\t")
                        header = next(tsv_reader)
                        terms = header[2:]
                        term_to_name = {
                            term: name
                            for term, name in zip(terms, GONAMES)
                        }
                        for row in tsv_reader:
                            query_id = row[0]
                            net_type = row[1]
                            scores = row[2:]
                            if len(scores) != len(terms):
                                raise ValueError(
                                    f"Row length mismatch for mode {mode}: "
                                    f"{len(scores)} scores vs {len(terms)} terms "
                                    f"(config {json_path}).")
                            term_score = {
                                terms[i]: float(scores[i])
                                for i in range(len(terms))
                                if float(scores[i]) >= 0.1
                            }
                            sorted_term_score = dict(
                                sorted(term_score.items(),
                                       key=lambda item: item[1],
                                       reverse=True))
                            for term, score in sorted_term_score.items():
                                go_name = term_to_name.get(term, "Unknown")
                                aln_info = alignment_data.get(
                                    query_id, [np.nan] * 6)
                                aligned, target_id, database, target_identity, query_cov, target_cov = aln_info
                                ic, cogs, supercogs = go2cog_mapping.get(
                                    term, (None, MISSING_TSV, MISSING_TSV))
                                fout.write(
                                    f"{query_id}\t{net_type}\t{DEEPFRI_MODES[mode]}\t{term}\t{score:.4f}\t{go_name}"
                                    f"\t{aligned}\t{target_id}\t{database}\t{target_identity}\t{query_cov}\t{target_cov}\t{_format_ic(ic)}\t{cogs}\t{supercogs}\n"
                                )
                    continue

                header = next(tsv_reader)
                terms = header[2:]
                term_to_name = {
                    term: name
                    for term, name in zip(terms, GONAMES)
                }
                for row in tsv_reader:
                    query_id = row[0]
                    net_type = row[1]
                    scores = row[2:]
                    if len(scores) != len(terms):
                        raise ValueError(
                            f"Row length mismatch for mode {mode}: "
                            f"{len(scores)} scores vs {len(terms)} terms "
                            f"(config {json_path}).")
                    term_score = {
                        terms[i]: float(scores[i])
                        for i in range(len(terms)) if float(scores[i]) >= 0.1
                    }
                    sorted_term_score = dict(
                        sorted(term_score.items(),
                               key=lambda item: item[1],
                               reverse=True))
                    for term, score in sorted_term_score.items():
                        go_name = term_to_name.get(term, "Unknown")
                        aln_info = alignment_data.get(query_id, [np.nan] * 6)
                        aligned, target_id, database, target_identity, query_cov, target_cov = aln_info
                        ic, cogs, supercogs = go2cog_mapping.get(
                            term, (None, MISSING_TSV, MISSING_TSV))
                        fout.write(
                            f"{query_id}\t{net_type}\t{DEEPFRI_MODES[mode]}\t{term}\t{score:.4f}\t{go_name}"
                            f"\t{aligned}\t{target_id}\t{database}\t{target_identity}\t{query_cov}\t{target_cov}\t{_format_ic(ic)}\t{cogs}\t{supercogs}\n"
                        )

    # GO-term propagation (true-path rule)
    if propagate_go_terms:
        from mDeepFRI.go_propagation import download_obo, propagate_results

        if obo_path is None:
            obo_file = output_path / "go-basic.obo"
        else:
            obo_file = pathlib.Path(obo_path)

        download_obo(obo_file)

        propagated_output = output_path / "results_propagated.tsv"
        propagate_results(
            results_path=final_output,
            output_path=propagated_output,
            obo_path=obo_file,
        )

    if remove_intermediate:
        for db in databases:
            remove_intermediate_files([db.sequence_db, db.mmseqs_db])

    logger.info("meta-DeepFRI finished successfully.")
