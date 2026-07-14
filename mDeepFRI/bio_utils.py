"""
Biological utilities for structure processing and contact map generation.

This module provides utilities for:
- Loading protein structures from PDB/CIF files and FoldComp databases
- Extracting C-alpha coordinates
- Generating contact maps from structures
- Aligning contact maps between query and target sequences
- Handling non-standard amino acid residues

The module integrates with Biotite for structure parsing and FoldComp for
compressed structure databases, enabling efficient processing of large-scale
structural annotations.

Functions:
    build_align_contact_map: Complete pipeline for contact map alignment.
    decompress_and_decode: Decompress FoldComp structure to Biotite object.
    get_calpha_coordinates: Extract C-alpha atom coordinates.
    construct_contact_map: Generate contact map from coordinates.
    align_coordinates: Align coordinates based on sequence alignment.
"""

import gc
import logging
import multiprocessing
import os
import pathlib
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from io import StringIO
from typing import Dict, List, Literal, NamedTuple, Optional, Tuple

import foldcomp
import numpy as np
from biotite.sequence import ProteinSequence
from biotite.structure import concatenate, get_chains
from biotite.structure.io.pdb import PDBFile
from biotite.structure.io.pdbx import CIFFile, get_structure

from mDeepFRI.alignment import AlignmentResult
from mDeepFRI.contact_map_utils import align_contact_map, pairwise_sqeuclidean
from mDeepFRI.mmseqs import FOLDCOMP_PATH

logger = logging.getLogger(__name__)


def _pin_blas_threads_per_process() -> None:
    """Avoid oversubscribing CPU when using one process per worker."""
    for env_var in (
            "OPENBLAS_NUM_THREADS",
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[env_var] = "1"


def _init_parallel_worker() -> None:
    _pin_blas_threads_per_process()
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(
    '[%(asctime)s] %(module)s.%(funcName)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# https://github.com/openmm/pdbfixer/blob/master/pdbfixer/pdbfixer.py
substitutions = {
    '2AS': 'ASP',
    '3AH': 'HIS',
    '5HP': 'GLU',
    '5OW': 'LYS',
    'ACL': 'ARG',
    'AGM': 'ARG',
    'AIB': 'ALA',
    'ALM': 'ALA',
    'ALO': 'THR',
    'ALY': 'LYS',
    'ARM': 'ARG',
    'ASA': 'ASP',
    'ASB': 'ASP',
    'ASK': 'ASP',
    'ASL': 'ASP',
    'ASQ': 'ASP',
    'AYA': 'ALA',
    'BCS': 'CYS',
    'BHD': 'ASP',
    'BMT': 'THR',
    'BNN': 'ALA',
    'BUC': 'CYS',
    'BUG': 'LEU',
    'C5C': 'CYS',
    'C6C': 'CYS',
    'CAS': 'CYS',
    'CCS': 'CYS',
    'CEA': 'CYS',
    'CGU': 'GLU',
    'CHG': 'ALA',
    'CLE': 'LEU',
    'CME': 'CYS',
    'CSD': 'ALA',
    'CSO': 'CYS',
    'CSP': 'CYS',
    'CSS': 'CYS',
    'CSW': 'CYS',
    'CSX': 'CYS',
    'CXM': 'MET',
    'CY1': 'CYS',
    'CY3': 'CYS',
    'CYG': 'CYS',
    'CYM': 'CYS',
    'CYQ': 'CYS',
    'DAH': 'PHE',
    'DAL': 'ALA',
    'DAR': 'ARG',
    'DAS': 'ASP',
    'DCY': 'CYS',
    'DGL': 'GLU',
    'DGN': 'GLN',
    'DHA': 'ALA',
    'DHI': 'HIS',
    'DIL': 'ILE',
    'DIV': 'VAL',
    'DLE': 'LEU',
    'DLY': 'LYS',
    'DNP': 'ALA',
    'DPN': 'PHE',
    'DPR': 'PRO',
    'DSN': 'SER',
    'DSP': 'ASP',
    'DTH': 'THR',
    'DTR': 'TRP',
    'DTY': 'TYR',
    'DVA': 'VAL',
    'EFC': 'CYS',
    'FLA': 'ALA',
    'FME': 'MET',
    'GGL': 'GLU',
    'GL3': 'GLY',
    'GLZ': 'GLY',
    'GMA': 'GLU',
    'GSC': 'GLY',
    'HAC': 'ALA',
    'HAR': 'ARG',
    'HIC': 'HIS',
    'HIP': 'HIS',
    'HMR': 'ARG',
    'HPQ': 'PHE',
    'HTR': 'TRP',
    'HYP': 'PRO',
    'IAS': 'ASP',
    'IIL': 'ILE',
    'IYR': 'TYR',
    'KCX': 'LYS',
    'LLP': 'LYS',
    'LLY': 'LYS',
    'LTR': 'TRP',
    'LYM': 'LYS',
    'LYZ': 'LYS',
    'MAA': 'ALA',
    'MEN': 'ASN',
    'MHS': 'HIS',
    'MIS': 'SER',
    'MK8': 'LEU',
    'MLE': 'LEU',
    'MPQ': 'GLY',
    'MSA': 'GLY',
    'MSE': 'MET',
    'MVA': 'VAL',
    'NEM': 'HIS',
    'NEP': 'HIS',
    'NLE': 'LEU',
    'NLN': 'LEU',
    'NLP': 'LEU',
    'NMC': 'GLY',
    'OAS': 'SER',
    'OCS': 'CYS',
    'OMT': 'MET',
    'PAQ': 'TYR',
    'PCA': 'GLU',
    'PEC': 'CYS',
    'PHI': 'PHE',
    'PHL': 'PHE',
    'PR3': 'CYS',
    'PRR': 'ALA',
    'PTR': 'TYR',
    'PYX': 'CYS',
    'SAC': 'SER',
    'SAR': 'GLY',
    'SCH': 'CYS',
    'SCS': 'CYS',
    'SCY': 'CYS',
    'SEL': 'SER',
    'SEP': 'SER',
    'SET': 'SER',
    'SHC': 'CYS',
    'SHR': 'LYS',
    'SMC': 'CYS',
    'SOC': 'CYS',
    'STY': 'TYR',
    'SVA': 'SER',
    'TIH': 'ALA',
    'TPL': 'TRP',
    'TPO': 'THR',
    'TPQ': 'ALA',
    'TRG': 'LYS',
    'TRO': 'TRP',
    'TYB': 'TYR',
    'TYI': 'TYR',
    'TYQ': 'TYR',
    'TYS': 'TYR',
    'TYY': 'TYR'
}

ONE_TO_THREE = {
    "A": "ALA",
    "R": "ARG",
    "N": "ASN",
    "D": "ASP",
    "C": "CYS",
    "Q": "GLN",
    "E": "GLU",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "L": "LEU",
    "K": "LYS",
    "M": "MET",
    "F": "PHE",
    "P": "PRO",
    "S": "SER",
    "T": "THR",
    "W": "TRP",
    "Y": "TYR",
    "V": "VAL",
    "U": "SEC",
    "O": "PYL",
    "X": "UNK",
}


def build_target_to_query_map(gapped_query: str,
                              gapped_target: str) -> list[int]:
    """
    Map template residue indices to query indices from a gapped alignment.

    Mirrors the column-wise walk in ``align_contact_map`` without synthetic
    contacts. Template residues aligned to query gaps receive ``-1``.

    Args:
        gapped_query: Query sequence with gap characters.
        gapped_target: Target sequence with gap characters.

    Returns:
        List where index is the template residue index (0-based CA order)
        and value is the query residue index or ``-1``.
    """
    if len(gapped_query) != len(gapped_target):
        raise ValueError("Gapped query and target must have equal length.")

    target_to_query: list[int] = []
    query_idx = 0
    target_idx = 0

    for q_char, t_char in zip(gapped_query, gapped_target):
        if q_char == "-":
            target_to_query.append(-1)
            target_idx += 1
        elif t_char == "-":
            query_idx += 1
        else:
            target_to_query.append(query_idx)
            query_idx += 1
            target_idx += 1

    return target_to_query


def chain_id_from_filename(filename: str) -> Optional[str]:
    """
    Infer a chain identifier from a PDB-style template filename suffix.

    Filenames like ``5aa0_BZ.pdb`` or ``6sxu_BBB.pdb`` encode the chain as the
    last alphanumeric character of the part after the PDB ID
    (``Z`` and ``B`` respectively). AlphaFold and other non-PDB names are ignored.
    """
    stem = pathlib.Path(filename).stem
    if "_" not in stem:
        return None
    pdb_id, suffix = stem.rsplit("_", 1)
    if len(pdb_id) != 4 or not pdb_id.isalnum():
        return None
    if not suffix:
        return None
    chain_char = suffix[-1]
    if chain_char.isalnum():
        return chain_char
    return None


def resolve_structure_chain(
        structure: np.ndarray,
        structure_path: Optional[str] = None,
        chain: str = "A") -> str:
    """
    Resolve the chain ID to use for a template structure.

    Prefers ``chain`` when present, otherwise parses ``structure_path`` using
    :func:`chain_id_from_filename`, trying case variants for letter chains.
    """
    chains = list(get_chains(structure))
    if chain in chains:
        return chain

    if structure_path:
        candidate = chain_id_from_filename(structure_path)
        if candidate is not None:
            for variant in (candidate, candidate.lower(), candidate.upper()):
                if variant in chains:
                    return variant

    if chain != "A":
        raise ValueError(
            f"Chain {chain!r} not found in structure (available: {chains}).")

    if len(chains) == 1:
        return chains[0]

    best_chain = None
    best_count = -1
    for chain_id in chains:
        ca_count = np.sum((structure.chain_id == chain_id)
                          & (structure.atom_name == "CA")
                          & (structure.hetero == False))  # noqa
        if ca_count > best_count:
            best_count = ca_count
            best_chain = chain_id

    if best_chain is not None and best_count > 0:
        return best_chain

    raise ValueError(
        f"Chain {chain!r} not found in structure (available: {chains}).")


def get_residue_atom_groups(structure: np.ndarray,
                            chain: str = "A") -> list[np.ndarray]:
    """
    Group all atoms by residue in CA-atom order.

    Uses the same CA ordering as :func:`get_residues_coordinates`.

    Args:
        structure: Biotite atom array for the structure.
        chain: Chain identifier to extract.

    Returns:
        List of atom arrays, one per residue in CA order.
    """
    chains = get_chains(structure)
    if chain not in chains:
        raise ValueError(f"Chain {chain} not found in structure.")

    protein_chain = structure[structure.chain_id == chain]
    ca_atoms = protein_chain[(protein_chain.atom_name == "CA")
                             & (protein_chain.hetero == False)]  # noqa

    groups: list[np.ndarray] = []
    for ca_atom in ca_atoms:
        residue_mask = ((protein_chain.chain_id == ca_atom.chain_id)
                        & (protein_chain.res_id == ca_atom.res_id)
                        & (protein_chain.ins_code == ca_atom.ins_code))
        groups.append(protein_chain[residue_mask])

    return groups


def _query_residue_to_three_letter(residue: str) -> str:
    return ONE_TO_THREE.get(residue.upper(), "UNK")


def _format_seqres_records(query_sequence: str, chain: str = "A") -> list[str]:
    three_letter = [
        _query_residue_to_three_letter(residue)
        for residue in query_sequence
    ]
    total = len(three_letter)
    lines: list[str] = []
    for start in range(0, total, 13):
        chunk = three_letter[start:start + 13]
        line_num = start // 13 + 1
        residue_names = " ".join(f"{name:>3}" for name in chunk)
        lines.append(
            f"SEQRES{line_num:4d} {chain} {total:4d}  {residue_names}")
    return lines


def _format_carved_pdb_header(alignment: AlignmentResult,
                              chain: str = "A") -> list[str]:
    template_id = alignment.target_name
    query_id = alignment.query_name
    remarks = [
        "REMARK   1 CARVED BY MDEEPFRI",
        f"REMARK   1 TEMPLATE STRUCTURE: {template_id}",
        f"REMARK   1 QUERY SEQUENCE: {query_id}",
        "REMARK   1 ALIGNMENT METHOD: PYOPAL GLOBAL NEEDLEMAN-WUNSCH",
        "REMARK   1 RESIDUE NAMES FOLLOW THE QUERY SEQUENCE; ATOM",
        "REMARK   1 COORDINATES AND ATOM TYPES ARE TRANSFERRED FROM THE",
        "REMARK   1 TEMPLATE. RESIDUE IDENTITY MAY NOT MATCH ATOM GEOMETRY.",
        "REMARK   1 QUERY INSERTIONS (GAPS IN TEMPLATE ALIGNMENT) APPEAR IN",
        "REMARK   1 SEQRES ONLY AND HAVE NO ATOM COORDINATES.",
    ]
    return remarks + _format_seqres_records(alignment.query_sequence, chain)


def carve_aligned_pdb(alignment: AlignmentResult,
                      structure_string: str,
                      filetype: Literal["mmcif", "pdb"],
                      chain: str = "A",
                      output_chain: str = "A",
                      structure: Optional[np.ndarray] = None) -> str:
    """
    Carve a query-aligned PDB from a template structure and PyOpal alignment.

    Coordinates are transferred from mapped template residues. Residue names
    and numbering follow the query sequence (1-based). Query insertions aligned
    to template gaps are included in SEQRES only.

    Args:
        alignment: Pairwise alignment with gapped sequences.
        structure_string: Template structure file contents.
        filetype: ``"pdb"`` or ``"mmcif"``.
        chain: Chain to read from the template structure.
        output_chain: Chain identifier for the carved PDB.

    Returns:
        PDB file contents as a string.
    """
    if structure is None:
        structure = load_structure(structure_string, filetype=filetype)
    residue_groups = get_residue_atom_groups(structure, chain=chain)
    target_to_query = build_target_to_query_map(alignment.gapped_sequence,
                                                alignment.gapped_target)

    if len(residue_groups) != len(target_to_query):
        raise ValueError(
            f"Template residue count ({len(residue_groups)}) does not match "
            f"alignment target length ({len(target_to_query)}).")

    carved_groups: list[np.ndarray] = []
    for target_idx, query_idx in enumerate(target_to_query):
        if query_idx < 0:
            continue

        residue_atoms = residue_groups[target_idx].copy()
        three_letter = _query_residue_to_three_letter(
            alignment.query_sequence[query_idx])
        residue_atoms.res_name[:] = three_letter
        residue_atoms.res_id[:] = query_idx + 1
        residue_atoms.chain_id[:] = np.full(len(residue_atoms),
                                            output_chain,
                                            dtype=residue_atoms.chain_id.dtype)
        carved_groups.append(residue_atoms)

    header_lines = _format_carved_pdb_header(alignment, chain=output_chain)
    if not carved_groups:
        atom_section = ""
    else:
        carved_structure = concatenate(carved_groups)
        pdb_file = PDBFile()
        pdb_file.set_structure(carved_structure)
        atom_buffer = StringIO()
        pdb_file.write(atom_buffer)
        atom_section = atom_buffer.getvalue()

    return "\n".join(header_lines) + "\n" + atom_section


def resolve_template_structure(
        alignment: AlignmentResult,
        databases: Tuple[object, ...] = ()) -> Tuple[str, str, str]:
    """
    Load the template structure used for an alignment.

    Args:
        alignment: Alignment result with database metadata.
        databases: FoldComp/PDB databases from the pipeline search.

    Returns:
        Tuple of structure string, filetype (``"pdb"`` or ``"mmcif"``), and
        chain identifier.
    """
    if alignment.db_name == "custom_mapping":
        if not alignment.structure_path:
            raise ValueError(
                f"No structure path recorded for {alignment.query_name}.")
        struct_path = pathlib.Path(alignment.structure_path)
        if not struct_path.exists():
            raise FileNotFoundError(
                f"Structure file not found: {alignment.structure_path}")
        structure_string = struct_path.read_text(encoding="utf-8")
        suffix = struct_path.suffix.lower()
        if suffix in {".cif", ".mmcif"}:
            filetype = "mmcif"
        elif suffix == ".pdb":
            filetype = "pdb"
        else:
            filetype = "mmcif"
        chain = chain_id_from_filename(str(struct_path)) or "A"
        return structure_string, filetype, chain

    if alignment.db_name and "pdb100" in alignment.db_name:
        from mDeepFRI.pdb import get_pdb_structure

        target_id = alignment.target_name.rsplit(".", 1)[0]
        pdb_id, _ = target_id.upper().split("_", 1)
        chain = chain_id_from_filename(target_id) or "A"
        structure_string = get_pdb_structure(pdb_id.lower())
        return structure_string, "mmcif", chain

    db = next((db for db in databases if db.name == alignment.db_name), None)
    if db is None:
        raise ValueError(
            f"Database '{alignment.db_name}' not found for "
            f"{alignment.query_name}.")

    target_id = alignment.target_name.rsplit(".", 1)[0]
    suffix = foldcomp_sniff_suffix(target_id, str(db.foldcomp_db))
    if suffix:
        target_id = f"{target_id}{suffix}"

    with foldcomp.open(str(db.foldcomp_db), ids=[target_id]) as struct_db:
        for _, structure_string in struct_db:
            return structure_string, "pdb", "A"

    raise ValueError(
        f"Structure '{alignment.target_name}' not found in {alignment.db_name}."
    )


def _alignment_chain(alignment: AlignmentResult) -> str:
    if alignment.db_name == "custom_mapping":
        if alignment.structure_path:
            return chain_id_from_filename(alignment.structure_path) or "A"
        return "A"
    target_id = alignment.target_name.rsplit(".", 1)[0]
    if alignment.db_name and "pdb100" in alignment.db_name:
        return chain_id_from_filename(target_id) or "A"
    return "A"


def _template_cache_key(alignment: AlignmentResult) -> str:
    if alignment.db_name == "custom_mapping":
        return f"custom:{alignment.structure_path}"
    target_id = alignment.target_name.rsplit(".", 1)[0]
    if alignment.db_name and "pdb100" in alignment.db_name:
        pdb_id, _ = target_id.upper().split("_", 1)
        return f"pdb100:{pdb_id.lower()}"
    return f"foldcomp:{alignment.db_name}:{target_id}"


def _filetype_from_path(structure_path: str) -> Literal["mmcif", "pdb"]:
    suffix = pathlib.Path(structure_path).suffix.lower()
    if suffix in {".cif", ".mmcif"}:
        return "mmcif"
    if suffix == ".pdb":
        return "pdb"
    return "mmcif"


def prefetch_template_structures(
        alignments: List[AlignmentResult],
        databases: Tuple[object, ...] = ()) -> Dict[str, Tuple[str, str]]:
    """
    Prefetch unique template structures into an in-memory cache.

    Returns:
        Mapping from template cache key to ``(structure_string, filetype)``.
        Chain identifiers are resolved per alignment via :func:`_alignment_chain`.
    """
    cache: Dict[str, Tuple[str, str]] = {}

    for aln in alignments:
        if aln.structure_string and aln.structure_path:
            cache[_template_cache_key(aln)] = (
                aln.structure_string,
                _filetype_from_path(aln.structure_path),
            )

    custom_paths = {
        aln.structure_path
        for aln in alignments
        if aln.db_name == "custom_mapping" and aln.structure_path
        and _template_cache_key(aln) not in cache
    }
    for structure_path in custom_paths:
        struct_path = pathlib.Path(structure_path)
        cache[f"custom:{structure_path}"] = (
            struct_path.read_text(encoding="utf-8"),
            _filetype_from_path(structure_path),
        )

    pdb_ids = {
        aln.target_name.rsplit(".", 1)[0].upper().split("_", 1)[0].lower()
        for aln in alignments
        if aln.db_name and "pdb100" in aln.db_name
    }
    if pdb_ids:
        from mDeepFRI.pdb import get_pdb_structure

        for pdb_id in pdb_ids:
            cache[f"pdb100:{pdb_id}"] = (get_pdb_structure(pdb_id), "mmcif")

    foldcomp_by_db: Dict[str, set[str]] = {}
    for aln in alignments:
        if (aln.db_name == "custom_mapping"
                or (aln.db_name and "pdb100" in aln.db_name)):
            continue
        target_id = aln.target_name.rsplit(".", 1)[0]
        foldcomp_by_db.setdefault(aln.db_name, set()).add(target_id)

    db_by_name = {db.name: db for db in databases}
    for db_name, target_ids in foldcomp_by_db.items():
        db = db_by_name.get(db_name)
        if db is None:
            continue
        ids = list(target_ids)
        suffix = foldcomp_sniff_suffix(ids[0], str(db.foldcomp_db))
        if suffix:
            ids = [f"{target_id}{suffix}" for target_id in ids]
        with foldcomp.open(str(db.foldcomp_db), ids=ids) as struct_db:
            for idx, structure_string in struct_db:
                base_id = idx.removesuffix(suffix) if suffix else idx
                cache[f"foldcomp:{db_name}:{base_id}"] = (structure_string,
                                                           "pdb")

    return cache


def _resolve_carving_chain(alignment: AlignmentResult, structure_string: str,
                           filetype: str, chain: str) -> str:
    structure_path = alignment.structure_path
    if not structure_path and "_" in alignment.target_name:
        structure_path = alignment.target_name
    structure = load_structure(structure_string, filetype=filetype)
    return resolve_structure_chain(structure, structure_path, chain)


def _get_template_for_carving(
        alignment: AlignmentResult,
        databases: Tuple[object, ...] = ()) -> Tuple[str, str, str]:
    if alignment.structure_path:
        struct_path = pathlib.Path(alignment.structure_path)
        if struct_path.exists():
            structure_string = struct_path.read_text(encoding="utf-8")
            filetype = _filetype_from_path(str(struct_path))
            chain = _alignment_chain(alignment)
            chain = _resolve_carving_chain(alignment, structure_string,
                                           filetype, chain)
            return structure_string, filetype, chain

    if alignment.structure_string:
        filetype = "mmcif"
        if alignment.structure_path:
            filetype = _filetype_from_path(alignment.structure_path)
        chain = _alignment_chain(alignment)
        chain = _resolve_carving_chain(alignment, alignment.structure_string,
                                       filetype, chain)
        return alignment.structure_string, filetype, chain

    structure_string, filetype, chain = resolve_template_structure(
        alignment, databases)
    chain = _resolve_carving_chain(alignment, structure_string, filetype,
                                   chain)
    return structure_string, filetype, chain


class CarveJob(NamedTuple):
    query_name: str
    structure_path: str
    query_sequence: str
    gapped_query: str
    gapped_target: str
    target_name: str
    carve_dir: str


def _carve_single_job(job: CarveJob) -> Tuple[str, Optional[str]]:
    try:
        struct_path = pathlib.Path(job.structure_path)
        structure_string = struct_path.read_text(encoding="utf-8")
        filetype = _filetype_from_path(str(struct_path))
        structure = load_structure(structure_string, filetype=filetype)
        chain = resolve_structure_chain(
            structure,
            str(struct_path),
            chain_id_from_filename(str(struct_path)) or "A",
        )
        alignment = AlignmentResult(
            query_name=job.query_name,
            query_sequence=job.query_sequence,
            target_name=job.target_name,
            target_sequence="",
            alignment="",
        )
        alignment.gapped_sequence = job.gapped_query
        alignment.gapped_target = job.gapped_target
        pdb_content = carve_aligned_pdb(alignment,
                                        structure_string,
                                        filetype=filetype,
                                        chain=chain,
                                        structure=structure)
        pdb_path = pathlib.Path(job.carve_dir) / f"{job.query_name}.pdb"
        with open(pdb_path, "w", encoding="utf-8") as carved_output:
            carved_output.write(pdb_content)
        return job.query_name, None
    except Exception as exc:
        return job.query_name, str(exc)


def _carve_from_database(
        alignment: AlignmentResult,
        databases: Tuple[object, ...],
        carve_dir: str) -> Tuple[str, Optional[str]]:
    try:
        structure_string, filetype, chain = _get_template_for_carving(
            alignment, databases)
        structure = load_structure(structure_string, filetype=filetype)
        pdb_content = carve_aligned_pdb(alignment,
                                        structure_string,
                                        filetype=filetype,
                                        chain=chain,
                                        structure=structure)
        pdb_path = pathlib.Path(carve_dir) / f"{alignment.query_name}.pdb"
        with open(pdb_path, "w", encoding="utf-8") as carved_output:
            carved_output.write(pdb_content)
        return alignment.query_name, None
    except Exception as exc:
        return alignment.query_name, str(exc)


def _carve_job_shard(jobs: List[CarveJob]) -> List[Tuple[str, Optional[str]]]:
    return [_carve_single_job(job) for job in jobs]


def _carve_legacy_shard(
        jobs: List[Tuple[AlignmentResult, Tuple[object, ...], str]]
) -> List[Tuple[str, Optional[str]]]:
    return [_carve_from_database(alignment, databases, carve_dir)
            for alignment, databases, carve_dir in jobs]


def write_carved_pdbs(
        aligned_cmaps: List[Tuple[AlignmentResult, np.ndarray]],
        databases: Tuple[object, ...],
        carve_dir: pathlib.Path,
        threads: int = 1,
        release_contact_maps: bool = False) -> int:
    """
    Carve and write query-aligned PDB files in parallel.

    Template structures are loaded from ``structure_path`` in each worker
    process. Jobs carry only the alignment fields required for carving so
    worker processes can be forked without duplicating contact maps.

    Args:
        aligned_cmaps: Aligned structure results from the pipeline.
        databases: Unused; kept for API compatibility.
        carve_dir: Output directory for carved PDB files.
        threads: Worker count for parallel carving.
        release_contact_maps: Drop contact maps and template coordinates
            from ``aligned_cmaps`` before forking workers.

    Returns:
        Number of successfully carved PDB files.
    """
    carve_dir.mkdir(parents=True, exist_ok=True)
    alignments = [aln for aln, _ in aligned_cmaps]
    if not alignments:
        return 0

    if release_contact_maps:
        for index, (alignment, _) in enumerate(aligned_cmaps):
            aligned_cmaps[index] = (alignment, None)
            alignment.coords = None
        gc.collect()

    jobs = [
        CarveJob(alignment.query_name,
                 alignment.structure_path,
                 alignment.query_sequence,
                 alignment.gapped_sequence,
                 alignment.gapped_target,
                 alignment.target_name,
                 str(carve_dir))
        for alignment in alignments
        if alignment.structure_path
    ]
    legacy_jobs = [(alignment, databases, str(carve_dir))
                   for alignment in alignments
                   if not alignment.structure_path]
    if not jobs and not legacy_jobs:
        return 0

    worker_count = max(1, min(threads, len(jobs) or len(legacy_jobs)))
    logger.info("Carving %d structure(s) using %d parallel worker process(es).",
                len(jobs) + len(legacy_jobs), worker_count)

    carved_count = 0
    results: List[Tuple[str, Optional[str]]] = []

    if worker_count == 1:
        results.extend(_carve_job_shard(jobs))
        results.extend(_carve_legacy_shard(legacy_jobs))
    else:
        ctx = multiprocessing.get_context("fork")
        job_shards = [jobs[i::worker_count] for i in range(worker_count)]
        job_shards = [shard for shard in job_shards if shard]
        legacy_shards = [legacy_jobs[i::worker_count]
                         for i in range(worker_count)]
        legacy_shards = [shard for shard in legacy_shards if shard]

        with ProcessPoolExecutor(max_workers=worker_count,
                                 mp_context=ctx,
                                 initializer=_init_parallel_worker) as executor:
            if job_shards:
                for shard_index, shard_results in enumerate(
                        executor.map(_carve_job_shard, job_shards), start=1):
                    results.extend(shard_results)
                    logger.info(
                        "Carved shard %d/%d (%d structure(s) processed).",
                        shard_index, len(job_shards), len(results))
            if legacy_shards:
                for shard_results in executor.map(_carve_legacy_shard,
                                                  legacy_shards):
                    results.extend(shard_results)

    for query_name, error in results:
        if error is None:
            carved_count += 1
        else:
            logger.warning("Failed to carve PDB for %s: %s", query_name, error)

    logger.info("Wrote %d carved PDB file(s) to %s.", carved_count, carve_dir)
    return carved_count


def compress_carved_structures(carve_dir: pathlib.Path,
                               output_db: pathlib.Path,
                               threads: int = 1) -> None:
    """
    Compress carved PDB files into a FoldComp database via ``foldcomp_bin``.

    Args:
        carve_dir: Directory containing carved ``*.pdb`` files.
        output_db: Output FoldComp database path.
        threads: Thread count passed to ``foldcomp compress -t``.
    """
    if not any(carve_dir.glob("*.pdb")):
        logger.warning("No PDB files found in %s; skipping compression.",
                       carve_dir)
        return

    foldcomp_bin = FOLDCOMP_PATH
    if not foldcomp_bin.exists():
        raise FileNotFoundError(
            f"FoldComp binary not found at {foldcomp_bin}. "
            "Run `python setup.py build_binaries --inplace` before using "
            "--compress-structures.")

    subprocess.run(
        [
            str(foldcomp_bin),
            "compress",
            "-t",
            str(threads),
            "-d",
            "-y",
            str(carve_dir),
            str(output_db),
        ],
        check=True,
    )
    pdb_files = list(carve_dir.glob("*.pdb"))
    for pdb_file in pdb_files:
        pdb_file.unlink()
    if carve_dir.exists() and not any(carve_dir.iterdir()):
        carve_dir.rmdir()

    logger.info(
        "Compressed carved structures to %s and removed %d individual PDB file(s).",
        output_db, len(pdb_files))


def calculate_contact_map(coordinates: np.ndarray,
                          threshold=6.0,
                          distance="sqeuclidean",
                          mode="matrix") -> np.ndarray:
    """
    Calculate contact map from PDB string.

    Args:
        pdb_string (str): PDB file read into string.
        threshold (float): Distance threshold for contact map.
        mode (str): Output mode. Either "matrix" or "sparse".

    Returns:
        np.ndarray: Contact map.
    """
    # squared euclidean is used for efficiency
    distance_functions = {"sqeuclidean": pairwise_sqeuclidean}

    if distance == "sqeuclidean":
        threshold = threshold**2

    distance_func = distance_functions[distance]

    distances = distance_func(coordinates)
    cmap = (distances < threshold).astype(np.int32)

    if mode == "sparse":
        cmap = np.argwhere(cmap == 1).astype(np.int32)
    else:
        pass

    return cmap


def get_residues_coordinates(structure: np.ndarray,
                             chain: str = "A",
                             structure_path: Optional[str] = None):
    """
    Retrieves residues and coordinates from biotite structure.

    Args:
        structure (np.ndarray): Structure file read into string.
        chain: Preferred chain identifier.
        structure_path: Optional path used to infer chain from filename.

    Returns:
        Tuple[str, np.ndarray]: Tuple of residues and coordinates.
    """
    chain = resolve_structure_chain(structure, structure_path, chain)
    protein_chain = structure[structure.chain_id == chain]
    # extract CA atoms coordinates
    ca_atoms = protein_chain[(protein_chain.atom_name == "CA")
                             & (protein_chain.hetero == False)]  # noqa

    residues = str(
        ProteinSequence(
            [substitutions.get(res, res) for res in ca_atoms.res_name]))
    coords = ca_atoms.coord

    return (residues, coords)


def load_structure(structure_string: str,
                   filetype: Literal["mmcif", "pdb"] = "mmcif") -> np.ndarray:
    """
    Load structure from string.

    Args:
        structure_string (str): Structure file read into string.
        filetype (str): Filetype of the structure.

    Returns:
        np.ndarray: Structure.
    """
    if filetype == "mmcif":
        mmcif = CIFFile.read(StringIO(structure_string))
        structure = get_structure(mmcif, model=1)
    elif filetype == "pdb":
        pdb = PDBFile.read(StringIO(structure_string))
        structure = pdb.get_structure()[0]
    else:
        raise NotImplementedError(f"Filetype {filetype} not supported.")

    return structure


def extract_residue_sequence(
        structure_string: str,
        chain: str = "A",
        filetype: Literal["mmcif", "pdb"] = "mmcif",
        structure_path: Optional[str] = None) -> Optional[str]:
    """Extract one-letter sequence from a structure without loading coordinates."""
    structure = load_structure(structure_string, filetype=filetype)
    chain = resolve_structure_chain(structure, structure_path, chain)
    protein_chain = structure[structure.chain_id == chain]
    ca_atoms = protein_chain[(protein_chain.atom_name == "CA")
                             & (protein_chain.hetero == False)]  # noqa
    if len(ca_atoms) == 0:
        return None
    return str(
        ProteinSequence(
            [substitutions.get(res, res) for res in ca_atoms.res_name]))


def extract_residues_coordinates(
        structure_string: str,
        chain: str = "A",
        filetype: Literal["mmcif", "pdb"] = "mmcif",
        structure_path: Optional[str] = None,
        save_directory: Optional[str] = None) -> Tuple[str, np.ndarray]:
    """
    Extracts residues and coordinates from structural string.
    Automatically processes PDB and mmCIF files.

    Args:
        structure_string (str): Structure file read into string.
        max_seq_len (int): Maximum sequence length.

    Returns:
        Tuple[str, np.ndarray]: Tuple of residues and coordinates.
    """

    structure = load_structure(structure_string, filetype=filetype)
    residues, coords = get_residues_coordinates(
        structure, chain=chain, structure_path=structure_path)

    return (residues, coords)


def foldcomp_sniff_suffix(idx: str, database_path: str) -> Optional[str]:
    """
    Sniff suffix for FoldComp database.

    Args:
        idx (str): Protein ID.
        database_path (str): Path to FoldComp database.

    Returns:
        str: Suffix for the database ids.
    """
    with foldcomp.open(database_path, ids=[idx]) as db:
        for _, structure in db:
            suffix = None
    if "structure" not in locals():
        idx = idx + ".pdb"
        with foldcomp.open(database_path, ids=[idx]) as db:
            for _, structure in db:
                suffix = ".pdb"

    return suffix


def get_foldcomp_structures(ids: List[str], database_path: str) -> List[str]:
    """
    Retrieves structure either from PDB or supplied FoldComp database.
    Extracts sequence and coordinate infromaton.

    Args:
        ids (List[str]): List of protein ids.
        database_path (str): Path to FoldComp database. If empty, the structure will be retrieved from the PDB.

    Returns:
        List[str]: List of structures.
    """
    structures = []
    with foldcomp.open(database_path, ids=ids) as db:
        for _, pdb in db:
            structures.append(pdb)

    return structures


def build_align_contact_map(
        alignment: AlignmentResult,
        threshold: float = 6,
        generated_contacts: int = 2) -> Tuple[AlignmentResult, np.ndarray]:
    """
    Retrieve contact map for aligned sequences.

    Args:
        alignment (AlignmentResult): Alignment of query and target sequences.
        database (str): Path to FoldComp database. If empty, the structure will be retrieved from the PDB.
        threshold (float): Distance threshold for contact map.
        generated_contacts (int): Number of generated contacts to add for gapped regions in the query alignment.

    Returns:
        Tuple[AlignmentResult, np.ndarray]: Tuple of alignment and contact map.
    """
    idx = alignment.target_name.rsplit(".", 1)[0]
    coordinates = alignment.coords
    if coordinates is not None:
        cmap = calculate_contact_map(coordinates,
                                     threshold=threshold,
                                     mode="sparse")
        try:
            aligned_cmap = align_contact_map(alignment.gapped_sequence,
                                             alignment.gapped_target, cmap,
                                             generated_contacts)
        except IndexError:
            pdb_id, chain = idx.upper().split("_")
            logger.warning(
                f"Error aligning contact map for PDB ID {pdb_id}[Chain {chain}] "
                f"against {alignment.query_name}.")
            aligned_cmap = None

    else:
        logger.warning(f"No coordinates found for {alignment.target_name}.")
        aligned_cmap = None

    return (alignment, aligned_cmap)
