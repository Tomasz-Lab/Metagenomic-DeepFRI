import gzip
import warnings
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Optional, Tuple

import foldcomp
import numpy as np
import requests
from pysam import tabix_compress

import mDeepFRI
from mDeepFRI.bio_utils import (extract_residues_coordinates,
                                foldcomp_sniff_suffix)
from mDeepFRI.database import Database
from mDeepFRI.mmseqs import _createdb, _createindex
from mDeepFRI.utils import download_file, stdout_warn

warnings.showwarning = stdout_warn


def create_pdb_mmseqs(threads: int = 1):
    """
    Downloads PDB100 database and creates an MMseqs2 database from it.

    Args:
        threads (int): Number of threads to use.

    Returns:
        Database: PDB100 database.
    """

    PDB100 = "https://wwwuser.gwdg.de/~compbiol/colabfold/pdb100_230517.fasta.gz"
    # check if pdb exists in a build dir
    build_dir = Path(mDeepFRI.__path__[0]).parent
    pdb100_path = Path(build_dir / "pdb100_230517.fasta.gz")
    # remove additional suffix
    base, first_ext, second_ext = pdb100_path.name.partition(".")
    pdb100_path = pdb100_path.with_name(base)
    uncompressed_path = pdb100_path.with_suffix(".fasta")
    compressed_path = pdb100_path.with_suffix(".fasta.gz")

    if not (compressed_path).exists():
        download_file(PDB100, compressed_path)

        # re-compress with bgzip (tabix_compress)
        with gzip.open(compressed_path,
                       "rb") as f_in, open(uncompressed_path, "wb") as f_out:
            f_out.write(f_in.read())

        tabix_compress(uncompressed_path, compressed_path, force=True)

        # remove uncompressed
        uncompressed_path.unlink()

    # create an MMSeqs database from PDB100
    # in a build directory
    pdb100_mmseqs = build_dir / "pdb100_230517.mmseqsDB"
    # check if database exists
    if not pdb100_mmseqs.exists():
        _createdb(compressed_path, pdb100_mmseqs)
        _createindex(pdb100_mmseqs, threads=threads)

    pdb_db = Database(foldcomp_db=pdb100_path.stem,
                      sequence_db=compressed_path,
                      mmseqs_db=pdb100_mmseqs)

    return pdb_db


def get_pdb_structure(pdb_id: str,
                      save_directory: Optional[str] = None) -> str:
    """
    Get PDB structure from the RCSB PDB database.

    Args:
        pdb_id (str): PDB ID.

    Returns:
        str: PDB structure in mmCIF format as a string.

    Raises:
        requests.RequestException: If the structure cannot be downloaded.
    """

    pdb_http = "https://files.rcsb.org/view/{pdb_id}.cif"
    pdb_id = pdb_id.lower()
    url = pdb_http.format(pdb_id=pdb_id)
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    structure = response.text
    if save_directory:
        with open(save_directory / f"{pdb_id}.cif", "w") as f:
            f.write(structure)
    return structure


# TODO: pdbfixer should remove error catching in this function
# only needed to run a function with multiprocessing
def get_pdb_seq_coords(
    pdb_id_chain: str,
    query_name: str,
    save_directory: Optional[str] = None
) -> Tuple[Optional[str], Optional[np.ndarray]]:
    """
    Get a sequence and coordinates of a protein chain from the PDB database.

    Args:
        pdb_id_chain (str): PDB ID and chain identifier separated by an underscore.
        query_name (str): Name of the query sequence. Not essential, used for logging.

    Returns:
        Tuple[str, np.ndarray]: A tuple containing a sequence and coordinates of a protein chain.
    """
    pdb_id, chain = pdb_id_chain.split("_")

    try:
        structure = get_pdb_structure(pdb_id, save_directory=save_directory)
        sequence, coords = extract_residues_coordinates(structure,
                                                        chain=chain,
                                                        filetype="mmcif")
    except (requests.RequestException, KeyError, ValueError) as e:
        sequence, coords = None, None
        pdb_id = pdb_id.upper()
        warnings.warn(
            f"Error extracting residues and coordinates for PDB ID {pdb_id}[Chain {chain}] - "
            f"{e!r}; {query_name} alignment skipped.")

    return sequence, coords


def extract_calpha_coords(db: Database,
                          target_ids: list[str],
                          query_ids: list[str],
                          save_directory: Optional[str] = None,
                          threads: int = 1) -> list[tuple]:

    if "pdb100" in db.name:
        # add save option
        if save_directory:
            get_pdb_seq_coords_parallel = partial(
                get_pdb_seq_coords, save_directory=save_directory)
        else:
            get_pdb_seq_coords_parallel = get_pdb_seq_coords

        with Pool(threads) as p:
            results = p.starmap(get_pdb_seq_coords_parallel,
                                zip(target_ids, query_ids))
        coords = [coord for _, coord in results]

    else:
        suffix = foldcomp_sniff_suffix(target_ids[0], db.foldcomp_db)
        if suffix:
            target_ids = [f"{t}{suffix}" for t in target_ids]
        coords = []
        with foldcomp.open(db.foldcomp_db, ids=target_ids) as struct_db:
            for idx, struct in struct_db:
                _, coord = extract_residues_coordinates(struct, filetype="pdb")
                coords.append(coord)
                if save_directory:
                    with open(save_directory / f"{idx}.pdb", "w") as f:
                        f.write(struct)

    return coords
