"""
Database module for managing FoldComp and MMseqs2 databases.

This module handles the creation, indexing, and management of FoldComp structure
databases and corresponding MMseqs2 sequence databases. It provides utilities for
extracting FASTA sequences from compressed FoldComp databases and preparing them
for efficient MMseqs2-based similarity searches.

Classes:
    Database: Dataclass for storing database file paths and metadata.

Functions:
    build_database: Create MMseqs2 database from FoldComp structures.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from mDeepFRI.mmseqs import _createdb, _createindex, extract_fasta_foldcomp

logger = logging.getLogger(__name__)


@dataclass
class Database:
    """
    Container for storing database file paths and metadata.

    This dataclass maintains references to FoldComp structure databases and their
    corresponding MMseqs2 sequence databases. All paths are automatically converted
    to Path objects for consistency.

    Attributes:
        foldcomp_db (Path): Path to FoldComp database containing compressed structures.
        sequence_db (Path): Path to extracted FASTA sequence database.
        mmseqs_db (Path): Path to MMseqs2 index database for fast searches.
        mmseqs_result (Path, optional): Path to MMseqs2 search results (if available).
            Defaults to None.
        name (str): Database identifier derived from sequence_db stem.

    Example:
        >>> db = Database(
        ...     foldcomp_db=Path("afdb_swissprot"),
        ...     sequence_db=Path("afdb_swissprot.fasta.gz"),
        ...     mmseqs_db=Path("afdb_swissprot.mmseqsDB")
        ... )
        >>> db.name
        'afdb_swissprot'
    """
    foldcomp_db: Path
    sequence_db: Path
    mmseqs_db: Path
    mmseqs_result: Optional[Path] = None

    def __post_init__(self):
        self.foldcomp_db = Path(self.foldcomp_db)
        self.sequence_db = Path(self.sequence_db)
        self.mmseqs_db = Path(self.mmseqs_db)
        if self.mmseqs_result:
            self.mmseqs_result = Path(self.mmseqs_result)
        self.name = self.sequence_db.stem.rsplit(".", 1)[0]


def build_database(
    input_path: str,
    output_path: str,
    overwrite: bool = False,
    threads: int = 1,
) -> Database:
    """
    Extract FASTA sequences from FoldComp database and create MMseqs2 index.

    This function performs the following steps:
    1. Extracts FASTA sequences from a compressed FoldComp database
    2. Creates an MMseqs2 database from the sequences
    3. Indexes the MMseqs2 database for fast similarity searches

    The function intelligently avoids redundant work by checking for existing
    files and only processing them if they don't exist or overwrite is requested.

    Args:
        input_path (str): Path to FoldComp database file (typically .fcz format).
        output_path (str): Path to output directory where database files will be created.
        overwrite (bool, optional): If True, regenerate all files even if they exist.
            Defaults to False.
        threads (int, optional): Number of threads to use for parallel processing.
            Defaults to 1.

    Returns:
        Database: Database object containing paths to all created database files.

    Raises:
        FileNotFoundError: If input_path does not exist.
        PermissionError: If unable to write to output_path.
        RuntimeError: If MMseqs2 database creation fails.

    Note:
        - Temporary disk space approximately 2-3× the input database size may be needed.
        - The function logs its progress via the logger module.
        - For large databases, this operation may take several minutes.
        - FASTA sequences are stored in gzip-compressed format to save disk space.

    Example:
        >>> db = build_database(
        ...     input_path="afdb_swissprot",
        ...     output_path="./databases",
        ...     threads=4
        ... )
        >>> print(db.name)
        'afdb_swissprot'
        >>> print(db.mmseqs_db)
        databases/afdb_swissprot.mmseqsDB

    See Also:
        Database: Dataclass for storing database paths
        extract_fasta_foldcomp: Function that extracts FASTA from FoldComp
    """

    logger.info("Building MMseqs2 database from %s", input_path)
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    output_sequences = output_path / Path(input_path.stem + ".fasta.gz")
    unzipped_sequences = output_sequences.with_suffix("")
    needs_new_mmseqs = False

    # check if files exist in output directory
    if output_sequences.exists() and not overwrite:
        logger.info("Found %s in %s", output_sequences, output_path)
        logger.info(
            "Skipping extraction of FASTA file from FoldComp database.")
    else:
        logger.info("Extracting FASTA file from FoldComp database.")
        output_sequences = extract_fasta_foldcomp(input_path,
                                                  unzipped_sequences, threads)
        logger.info("FASTA file extracted to %s", output_sequences)
        needs_new_mmseqs = True

    # create mmseqs db
    mmseqs_path = output_path / Path(input_path.stem + ".mmseqsDB")
    if overwrite or needs_new_mmseqs:
        logger.info("Creating and indexing MMseqs2 database.")
        _createdb(output_sequences, mmseqs_path)
        _createindex(mmseqs_path, threads)
        logger.info("Database created at %s", output_path)
    else:
        logger.info("Found %s in %s", mmseqs_path, output_path)
        logger.info("Skipping creation of MMseqs2 database.")

    database = Database(foldcomp_db=input_path,
                        sequence_db=output_sequences,
                        mmseqs_db=mmseqs_path)

    return database
