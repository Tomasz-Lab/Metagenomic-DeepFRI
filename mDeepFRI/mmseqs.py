"""
MMseqs2 integration module for fast protein sequence searching.

This module provides Python bindings and utilities for MMseqs2 (Many-vs-Many
sequence searching), enabling fast similarity searches against large protein
databases. It also includes utilities for FoldComp database manipulation and
result parsing.

Key Features:
    - Fast similarity search using MMseqs2
    - Support for FoldComp compressed structure databases
    - Query file handling with sequence filtering
    - Comprehensive result parsing and filtering
    - Multi-threaded searching capability

Classes:
    ValueRange: Annotation class for parameter range validation.
    QueryFile: Class for managing query protein sequences.
    MMseqsResult: Dataclass for storing search results.

Functions:
    _createdb: Convert FASTA to MMseqs2 database format.
    _createindex: Create MMseqs2 database index for fast searches.
    _search: Perform MMseqs2 similarity search.
    _convertalis: Convert MMseqs2 results to alignment format.
    extract_fasta_foldcomp: Extract FASTA sequences from FoldComp database.
"""

import csv
import os
import tempfile
from dataclasses import dataclass
from functools import partial
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Annotated, Callable, Dict, Iterable, List, Literal, Optional

import numpy as np
import numpy.lib.recfunctions as rfn
from pysam import FastxFile, tabix_compress

import mDeepFRI
from mDeepFRI.utils import retrieve_fasta_entries_as_dict, run_command

MMSEQS_PATH = Path(mDeepFRI.__path__[0]).parent / "mmseqs" / "bin" / "mmseqs"
FOLDCOMP_PATH = Path(mDeepFRI.__path__[0]).parent / "foldcomp_bin"
STRING_COLUMNS = ("query", "target", "query_file", "database_file")


@dataclass
class ValueRange:
    """
    Range constraint for parameter validation.

    This dataclass can be used with type annotations to document and validate
    acceptable parameter ranges in function signatures.

    Attributes:
        min (float): Minimum acceptable value (inclusive).
        max (float): Maximum acceptable value (inclusive).

    Example:
        >>> range_constraint = ValueRange(min=1.0, max=7.5)
        >>> range_constraint.min
        1.0
        >>> range_constraint.max
        7.5
    """
    min: float
    max: float


def _createdb(sequences_file, db_path):
    """
    Convert FASTA file to MMseqs2 database format.

    This function converts a FASTA sequence file into the binary database format
    required by MMseqs2. It generates multiple files including the main database,
    header database, and associated index files.

    Args:
        sequences_file (str): Path to input FASTA file containing protein sequences.
        db_path (str): Path prefix for output database files. Multiple files will be
            created with this prefix (e.g., db_path, db_path_h, db_path.index).

    Returns:
        None

    Raises:
        FileNotFoundError: If sequences_file does not exist.
        RuntimeError: If MMseqs2 command fails.

    Note:
        This function creates the following files:
        - {db_path}: Main database file
        - {db_path}_h: Header database file
        - {db_path}.index: Database index
        - {db_path}_h.index: Header index
        - {db_path}.lookup: Lookup table

    See Also:
        _createindex: Index the created database for fast searches.
    """
    run_command(
        f"{MMSEQS_PATH} createdb {sequences_file} {db_path} --dbtype 1")


def _createindex(db_path: str, threads: int = 1):
    """
    Create index for an MMseqs2 database.

    Indexing enables fast lookup of sequences and significantly speeds up
    subsequent similarity searches. This is a required step before performing
    searches with the database.

    Args:
        db_path (str): Path to MMseqs2 database to index.
        threads (int, optional): Number of threads for indexing. Defaults to 1.

    Returns:
        None

    Raises:
        RuntimeError: If MMseqs2 command fails.

    Note:
        This function uses a temporary directory for intermediate files which
        are automatically cleaned up upon completion.

    See Also:
        _createdb: Create database from FASTA file.
    """
    with tempfile.TemporaryDirectory() as tmp_path:
        run_command(
            f"{MMSEQS_PATH} createindex {db_path} {tmp_path} --threads {threads}"
        )


def _search(query_db: str,
            target_db: str,
            result_db: str,
            mmseqs_max_eval: float = 10e-5,
            mmseqs_sensitivity: Annotated[float,
                                          ValueRange(min=1.0, max=7.5)] = 5.7,
            threads: int = 1):
    """
    Perform MMseqs2 similarity search.

    Searches query sequences against a target database using MMseqs2's
    efficient sequence searching algorithm. Results are stored in MMseqs2
    database format and can be converted to human-readable alignment format
    using _convertalis.

    Args:
        query_db (str): Path to query MMseqs2 database.
        target_db (str): Path to target MMseqs2 database.
        result_db (str): Path prefix for output result database files.
        mmseqs_max_eval (float, optional): Maximum E-value threshold for hits.
            Defaults to 10e-5 (1e-4).
        mmseqs_sensitivity (float, optional): Sensitivity of the search.
            Range: 1.0 (very fast, low sensitivity) to 7.5 (slow, high sensitivity).
            Defaults to 5.7 (recommended balance).
        threads (int, optional): Number of threads for parallel searching.
            Defaults to 1.

    Returns:
        None

    Raises:
        RuntimeError: If MMseqs2 command fails.
        FileNotFoundError: If query_db or target_db do not exist.

    Note:
        Intermediate files are stored in a temporary directory which is
        automatically cleaned up upon completion.

    Example:
        >>> _search("query.mmseqsDB", "target.mmseqsDB", "results.mmseqsDB")

    See Also:
        _convertalis: Convert results to alignment format.
        _createindex: Prepare database for searching.
    """
    with tempfile.TemporaryDirectory() as tmp_path:
        run_command(
            f"{MMSEQS_PATH} search -e {mmseqs_max_eval} --threads {threads} "
            f"-s {mmseqs_sensitivity} {query_db} {target_db} {result_db} {tmp_path}"
        )


def _convertalis(query_db: str,
                 target_db: str,
                 result_db: str,
                 output_file: str,
                 threads: int,
                 columns: list[str] | None = None):
    if columns is None:
        columns = [
            "query", "target", "fident", "alnlen", "mismatch", "gapopen",
            "qstart", "qend", "tstart", "tend", "qcov", "tcov", "evalue",
            "bits"
        ]
    args = ",".join(columns)
    run_command(
        f"{MMSEQS_PATH} convertalis {query_db} {target_db} {result_db} {output_file} --format-mode 4 "
        f"--format-output {args} --threads {threads}")


def _read_results_tsv(filepath: str) -> np.recarray:
    """
    Read an MMseqs2 results table with a header row.

    Identifiers and paths are kept as strings, all other columns are
    parsed as floats.

    Args:
        filepath (str): Path to TSV file.

    Returns:
        np.recarray: One record per row; empty if the table has no rows.
    """
    with open(filepath, newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        names = next(reader)
        rows = [row for row in reader if row]

    columns = list(zip(*rows)) or [()] * len(names)
    arrays = [
        np.array(column, dtype=str if name in STRING_COLUMNS else float)
        for name, column in zip(names, columns)
    ]
    return np.rec.fromarrays(arrays, names=names)


class MMseqsResult(np.recarray):
    """
    Class for handling MMseqs2 search results. The results are stored in a TSV file.
    Inherits from numpy.recarray.

    Args:
        data (np.recarray): MMseqs2 search results.
        query_fasta (str): Path to query FASTA file.
        database (str): Path to MMseqs2 database.

    Attributes:
        data (np.recarray): MMseqs2 search results.
        query_fasta (str): Path to query FASTA file.
        database (str): Path to MMseqs2 database.
        columns (np.array): Array with column names.

    Example:

            >>> from mDeepFRI.mmseqs import MMseqsResult
            >>> result = MMseqsResult.from_mmseqs_result("path/to/file.tsv")
            >>> # sort file by identity and select seq1 hits only
            >>> result[::-1].sort(order=["fident"])
            >>> seq1_hits = result[result["query"] == "seq1"]
            >>> # save results
            >>> result.save("path/to/file.tsv")
    """
    def __init__(self, result_arr, query_fasta=None, database=None):
        self.result_arr = result_arr
        self.query_fasta = Path(query_fasta).resolve() if query_fasta else None
        self.database = Path(database).resolve() if database else None

    def __new__(cls, result_arr, query_fasta=None, database=None):
        obj = np.asarray(result_arr).view(cls)
        obj.query_fasta = query_fasta
        obj.database = database
        return obj

    def apply_mask(self, mask: np.ndarray):
        """
        Query search results.

        Args:
            mask (np.ndarray): Boolean mask.

        Returns:
            MMseqsResult: MMseqs2 search results.

        Example:

                >>> from mDeepFRI.mmseqs import MMseqsResult
                >>> result = MMseqsResult.from_mmseqs_result("path/to/file.tsv")
                >>> mask = result["fident"] > 0.5
                >>> filtered = result.apply_mask(mask)
        """

        return MMseqsResult(self.result_arr[mask], self.query_fasta,
                            self.database)

    @property
    def columns(self) -> np.ndarray:
        return np.array(self.result_arr.dtype.names)

    def save(self, filepath, filetype: Literal["tsv", "npz"] = "tsv"):
        """
        Save search results to TSV or NumPy compressed file.

        Args:
            filepath (str): Path to output file.
            filetype (str): File type to save. Options: "tsv" or "npz".

        Returns:
            None

        Example:

            >>> from mDeepFRI.mmseqs import MMseqsResult
            >>> result = MMseqsResult.from_mmseqs_result("path/to/file.tsv")
            >>> result.save("path/to/file.tsv")
        """
        # append query file column to the result array
        if self.query_fasta:
            query_col = np.array([self.query_fasta] * len(self.result_arr),
                                 dtype="U")
            self.result_arr = rfn.append_fields(self.result_arr,
                                                "query_file",
                                                query_col,
                                                usemask=False)

        # append database column to the result array
        if self.database:
            db_col = np.array([self.database] * len(self.result_arr),
                              dtype="U")
            self.result_arr = rfn.append_fields(self.result_arr,
                                                "database_file",
                                                db_col,
                                                usemask=False)

        if filetype == "tsv":
            with open(filepath, "w", newline="") as f:
                # write tsv
                writer = csv.writer(f, delimiter="\t")
                writer.writerow(self.result_arr.dtype.names)
                for row in self.result_arr:
                    writer.writerow(row)

        elif filetype == "npz":
            np.savez_compressed(filepath, self.result_arr)

        else:
            raise ValueError("File type should be 'tsv' or 'npz'.")

    def apply_filters(self,
                      min_cov: float = 0.0,
                      min_ident: float = 0.0,
                      min_bits: float = 0) -> "MMseqsResult":
        """
        Filter search results optionally by coverage, identity and bit score.

        Args:
            min_cov (float): Minimum coverage of query and target sequences.
            min_ident (float): Minimum identity of query and target sequences.
            min_bits (float): Minimum bit score.

        Returns:
            MMseqsResult: Filtered MMseqs2 search results.

        Example:
            >>> from mDeepFRI.mmseqs import MMseqsResult
            >>> result = MMseqsResult.from_mmseqs_result("path/to/file.tsv")
            >>> filtered = result.apply_filters(min_cov=30, min_ident=0.4, min_bits=50)
        """

        mask = (self.result_arr["qcov"] >= min_cov) & (self.result_arr["tcov"] >= min_cov) & \
                (self.result_arr["fident"] >= min_ident) & (self.result_arr["bits"] >= min_bits)

        return self.apply_mask(mask)

    def find_best_matches(self,
                          k: int = 5,
                          threads: int = 1) -> "MMseqsResult":
        """
        Selects k best matches for each query sequence based on bit score and identity.

        Args:
            k (int): Number of best matches to select.
            threads (int): Number of threads to use.

        Returns:
            MMseqsResult: MMseqs2 search results with best matches.
        """
        def select_top_k(query, db, k=30):
            return db[db["query"] == query][:k]

        # sort by bit score
        self.result_arr.sort(order=["query", "bits", "fident"],
                             kind="quicksort")

        select_top = partial(select_top_k, db=self.result_arr[::-1], k=k)
        # select k best hits
        top_k = []
        with ThreadPool(threads) as p:
            top_k = p.map(select_top, np.unique(self.result_arr["query"]))

        try:
            final_table = np.concatenate(top_k)
        except ValueError:
            final_table = self.result_arr[:0]

        return MMseqsResult(final_table, self.query_fasta, self.database)

    def get_queries(self):
        """
        Get unique query sequences.

        Returns:
            List[str]: List of unique query sequences.

        Example:

            >>> from mDeepFRI.mmseqs import MMseqsResult
            >>> result = MMseqsResult.from_mmseqs_result("path/to/file.tsv")
            >>> queries = result.get_queries()
        """
        return np.unique(self.result_arr["query"])

    def get_targets(self):
        """
        Get unique target sequences.

        Returns:
            List[str]: List of unique target sequences.

        Example:

            >>> from mDeepFRI.mmseqs import MMseqsResult
            >>> result = MMseqsResult.from_mmseqs_result("path/to/file.tsv")
            >>> targets = result.get_targets()
        """
        return np.unique(self.result_arr["target"])

    def get_query_targets(self, query: str):
        """
        Get target sequences for a specified query.

        Args:
            query (str): Query sequence ID.

        Returns:
            List[str]: List of target sequences for the query.

        Example:

            >>> from mDeepFRI.mmseqs import MMseqsResult
            >>> result = MMseqsResult.from_mmseqs_result("path/to/file.tsv")
            >>> targets = result.get_query_to_target("seq1")
        """
        return np.unique(
            self.result_arr[self.result_arr["query"] == query]["target"])

    @classmethod
    def from_mmseqs_result(cls,
                           filepath,
                           query_fasta=None,
                           database=None) -> "MMseqsResult":
        """
        Load search results from TSV file.

        Args:
            filepath (str): Path to TSV file from convertalis.
            query_fasta (str): Path to query FASTA file (optional).
            database (str): Path to MMseqs2 database (optional).

        Returns:
            MMseqsResult: MMseqs2 search results.

        Example:

                >>> from mDeepFRI.mmseqs import MMseqsResult
                >>> result = MMseqsResult.from_mmseqs_result("path/to/file.tsv")
        """

        result_arr = _read_results_tsv(filepath)
        return cls(result_arr, query_fasta, database)

    @classmethod
    def from_best_matches(cls, filepath: str) -> 'MMseqsResult':
        """
        Load best matches from TSV file.

        Args:
            filepath (str): Path to TSV file with best matches.

        Returns:
            MMseqsResult: MMseqs2 search results.

        Example:

                >>> from mDeepFRI.mmseqs import MMseqsResult
                >>> result = MMseqsResult.from_best_matches("path/to/file.tsv")
        """

        result_arr = _read_results_tsv(filepath)
        try:
            query_file = np.unique(result_arr["query_file"])[0]
        except IndexError:
            query_file = None
        try:
            database = np.unique(result_arr["database_file"])[0]
        except IndexError:
            database = None

        return cls(result_arr, query_file, database)


class QueryFile:
    """
    Class for handling FASTA files with sequences to query against MMseqs2 database.

    Args:
        filepath (str): Path to FASTA file.

    Attributes:
        filepath (str): Path to FASTA file.
        sequences (Dict[str, str]): Dictionary with sequence IDs as keys and sequences as values.
    """
    def __init__(self, filepath: str) -> None:
        self.filepath: str = filepath
        self.sequences: Dict[str, str] = {}
        self.filtered_out: Dict[str, str] = {}

    def __repr__(self) -> str:
        return f"QueryFile(filepath={self.filepath})"

    def __str__(self) -> str:
        return f"QueryFile(filepath={self.filepath})"

    def __setitem__(self, key, value) -> None:
        self.sequences[key] = value

    def __getitem__(self, key) -> None:
        return self.sequences[key]

    def load_ids(self, ids: Iterable[str]) -> None:
        """
        Load sequences by ID from FASTA file. The file is indexed with `samtools faidx`
        to speed up the process. Sequences are stored in a dictionary with
        sequence IDs as keys and sequences as values.

        Note:
            This method allows to load only sequences with specified IDs,
            which can be useful when working with large FASTA files.
            `samtools faidx` works with uncompressed FASTA files, and files compressed with bgzip.
            If compression is wrong, the file will be automatically re-compressed.

        Args:
            ids (List[str]): List of sequence IDs to load.

        Returns:
            None

        Raises:
            ValueError: If sequence with specified ID is not found in FASTA file.

        Example:
                >>> from mDeepFRI.mmseqs import QueryFile
                >>> query_file = QueryFile("path/to/file.fasta")
                >>> query_file.load_ids(["seq1", "seq2"])
        """
        # check if exists
        filepath = Path(self.filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"File {self.filepath} not found.")
        self.sequences = retrieve_fasta_entries_as_dict(filepath, ids)

    def load_sequences(self,
                       ids: Optional[list[str]] = None,
                       sort: bool = True) -> None:
        """
        Load sequences from FASTA file. Sequences are stored in a dictionary with sequence
        IDs as keys and sequences as values.



        Note:
            This method should be called only if maniuplating sequences directly is needed.

        Returns:
            None

        Example:

            >>> from mDeepFRI.mmseqs import QueryFile
            >>> query_file = QueryFile("path/to/file.fasta")
            >>> query_file.load_sequences()
        """

        if ids:
            self.load_ids(ids)
        else:
            with FastxFile(self.filepath) as f:
                for entry in f:
                    self.sequences[entry.name] = entry.sequence

        # sort sequences by length
        if sort:
            self.sequences = dict(
                sorted(self.sequences.items(), key=lambda x: len(x[1])))

    def remove_sequences(self, ids: List[str]):
        """
        Remove sequences by ID.

        Args:
            ids (List[str]): List of sequence IDs to remove.

        Returns:
            None

        Example:

            >>> from mDeepFRI.mmseqs import QueryFile
            >>> query_file = QueryFile("path/to/file.fasta")
            >>> query_file.load_sequences()
            >>> query_file.remove_sequences(["seq1", "seq2"])
        """
        for seq_id in ids:
            try:
                self.sequences.pop(seq_id, None)
            except KeyError:
                raise ValueError(
                    f"Sequence with ID {seq_id} not found in {self.filepath}")

    def filter_sequences(self,
                         condition: Optional[Callable[[str], bool]] = None):
        """
        Filter sequences by a custom condition.

        Args:
            condition (callable): A lambda function that takes a sequence as input and returns a boolean value.

        Returns:
            None

        Example:

            >>> from mDeepFRI.mmseqs import QueryFile
            >>> query_file = QueryFile("path/to/file.fasta")
            >>> query_file.load_sequences()
            >>> query_file.filter_sequences(lambda seq: 50 <= len(seq) <= 200)
            >>> query_file.filter_sequences(lambda seq: 'ATG' in seq)  # filter by presence of 'ATG' motif
            >>> query_file.filter_sequences(lambda seq: seq.count('N') < 5) # filter by number of 'N' characters
        """
        # check if sequences were loaded
        if not self.sequences:
            raise ValueError(
                "No sequences loaded. Use load_sequences() or load_ids() method to load sequences from FASTA file."
            )

        filtered_sequences = self.sequences.copy()
        if condition:
            filtered_sequences = {
                k: v
                for k, v in filtered_sequences.items() if condition(v)
            }
            for seq_id, seq in self.sequences.items():
                if seq_id not in filtered_sequences:
                    self.filtered_out[seq_id] = seq

        self.sequences = filtered_sequences

        if not self.sequences:
            raise ValueError("No sequences left after filtering.")

    def remove_selenocysteine(self) -> list[str]:
        """
        Remove sequences containing selenocysteine (U) and return their IDs.

        Returns:
            list[str]: IDs of sequences that were removed.

        Raises:
            ValueError: If no sequences are loaded.
        """
        if not self.sequences:
            raise ValueError(
                "No sequences loaded. Use load_sequences() or load_ids() before removing selenocysteine sequences."
            )

        removed = [
            seq_id for seq_id, seq in self.sequences.items() if "U" in seq
        ]
        for seq_id in removed:
            self.filtered_out[seq_id] = self.sequences.pop(seq_id)

        return removed

    def search(
            self,
            database_path: str,
            eval: float = 10e-5,
            mmseqs_sensitivity: Annotated[float,
                                          ValueRange(min=1.0, max=7.5)] = 5.7,
            index_target: bool = False,
            tmpdir=None,
            threads: int = 1):
        """
        Queries sequences against MMseqs2 database. The search results are stored in a tabular format.

        Args:
            database_path (str): Path to MMseqs2 database or database FASTA.
            eval (float): Maximum e-value for MMseqs2 search.
            mmseqs_sensitivity (float): Sensitivity value for MMseqs2 search.
            index_target (bool): Create index for target database. Advised for repeated searches.
            tmpdir (str): Path to temporary directory. MMseqs2 creates a lot of temporary files.
                          For large queries, needs to be set to a directory with enough space.
            threads (int): Number of threads to use.

        Returns:
            MMseqsResult: MMseqs2 search results.

        Example:

                >>> from mDeepFRI.mmseqs import QueryFile
                >>> query_file = QueryFile("path/to/file.fasta")
                >>> query_file.load_sequences()
                >>> query_file.filter_sequences(min_length=50, max_length=200)
                >>> result = query_file.search("path/to/database")
        """
        # check MMseqs2sensitivity values
        if not 1.0 <= mmseqs_sensitivity <= 7.5:
            raise ValueError(
                "MMseqs2 sensitivity value should be between 1.0 and 7.5.")

        with tempfile.TemporaryDirectory(dir=tmpdir) as tmp_path:
            if self.sequences:
                fasta_path = Path(tmp_path) / "filtered_query.fa"
                with open(fasta_path, "w") as f:
                    for seq_id, seq in self.sequences.items():
                        f.write(f">{seq_id}\n{seq}\n")
            else:
                fasta_path = self.filepath

            # create query db
            input_db_path = Path(tmp_path) / "query.mmseqsDB"
            _createdb(fasta_path, input_db_path)

            # create target db
            with open(database_path, "rb") as f:
                first_line = f.readline()

            if first_line.startswith(b">"):
                target_db_path = Path(database_path).with_suffix(".mmseqsDB")
                _createdb(database_path, target_db_path)
                if index_target:
                    _createindex(target_db_path, threads)

            else:
                target_db_path = database_path

            result_db = Path(tmp_path) / "search_resultDB"
            _search(input_db_path, target_db_path, result_db, eval,
                    mmseqs_sensitivity, threads)

            output_file = Path(tmp_path) / "search_results.tsv"
            _convertalis(input_db_path, target_db_path, result_db, output_file,
                         threads)

            result = MMseqsResult.from_mmseqs_result(
                output_file,
                query_fasta=self.filepath,
                database=target_db_path,
            )

        return result


def extract_fasta_foldcomp(foldcomp_db: str,
                           output_file: str,
                           threads: int = 1):

    ESM_DATABASES = ["highquality_clust30", "esmatlas", "esmatlas_v2023_02"]
    """
    Extracts FASTA from Foldcomp database and compresses it using block gzip.

    Args:
        foldcomp_db (str): Path to Foldcomp database.
        output_file (str): Path to output FASTA file.
        threads (int): Number of threads to use.

    Returns:
        str: Path to gzipped FASTA file.
    """

    database_name = Path(foldcomp_db).stem

    # run command
    run_command(
        f"{FOLDCOMP_PATH} extract --fasta -t {threads} {foldcomp_db} {output_file}"
    )

    if database_name in ESM_DATABASES:
        # use sed to correct the headers
        os.system(
            fr"sed -i 's/^>\(ESMFOLD V0 PREDICTION FOR \)\(.*\)/>\2/' {output_file}"
        )

    # gzip fasta file
    tabix_compress(output_file, str(output_file) + ".gz", force=True)
    # remove unzipped file
    os.remove(output_file)
    # remove possible previous index, might lead to errror
    try:
        os.remove(str(output_file) + ".gz.fai")
        os.remove(str(output_file) + ".gz.gzi")
    except FileNotFoundError:
        pass

    return Path(str(output_file) + ".gz")
