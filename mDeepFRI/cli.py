"""
Command-line interface for Metagenomic-DeepFRI.

This module provides the Click-based CLI for the Metagenomic-DeepFRI pipeline,
including commands for:
- Protein function prediction (predict-function)
- Model weight downloading (get-models)
- Utility operations (calculate-contact-map)

The CLI offers a user-friendly interface to all pipeline features with
comprehensive help messages, input validation, and progress reporting.

Commands:
    cli: Main command group
    get_models: Download DeepFRI model weights
    predict_function: Run protein function prediction pipeline
    calculate_contact_map: Generate contact map from structure file

Functions:
    setup_logging: Configure logging for the application
    common_params: Decorator for shared CLI parameters
"""

import importlib.metadata
import logging
import os
import sys
from functools import wraps
from pathlib import Path

import click
import numpy as np
from click._compat import get_text_stderr
from click.exceptions import UsageError
from click.utils import echo

from mDeepFRI.bio_utils import (calculate_contact_map,
                                get_residues_coordinates, load_structure)
from mDeepFRI.pipeline import (hierarchical_database_search, load_query_file,
                               predict_protein_function)
from mDeepFRI.utils import download_model_weights, generate_config_json

logger = logging.getLogger(__name__)


def setup_logging(debug: bool = False):
    """Configures the root logger for the application."""
    log_level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format=
        "[%(asctime)s] %(module)s.%(funcName)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,  # Override any existing configuration
    )


def log_command_params(ctx: click.Context):
    """Logs all parameters passed to a click command."""
    logger.info("Command parameters:")
    max_key_len = max(len(k) for k in ctx.params)
    for key, value in ctx.params.items():
        logger.info(f"{key:<{max_key_len + 2}} : {value}")


def patch_usage_error():
    """
    Patches click.UsageError.show to print the full help message
    before the error, which is more user-friendly.
    """

    original_show = UsageError.show

    def _show_usage_error_with_help(self, file=None):
        """Custom show method that includes the command's help."""
        if file is None:
            file = get_text_stderr()
        color = None
        if self.ctx is not None:
            color = self.ctx.color
            # Show the full help text for the command
            echo(self.ctx.get_help() + "\n", file=file, color=color)

        # Call the original show method to display the error message
        # We need to temporarily restore the original to avoid recursion
        UsageError.show = original_show
        self.show(file=file)
        # Re-patch it for subsequent errors
        UsageError.show = _show_usage_error_with_help

    UsageError.show = _show_usage_error_with_help


# Reusable decorator for search options
def search_options(function):
    """
    A decorator to apply a standard set of search-related
    click options to a command.
    """

    # Note: Applying decorators from bottom to top
    # is easier to read than re-assigning the function variable.
    @click.option(
        "--tmpdir",
        default=None,
        type=click.Path(exists=False,
                        file_okay=False,
                        dir_okay=True,
                        path_type=Path),
        help="Path to a temporary directory. Required for very large searches.",
    )
    @click.option(
        "--skip-pdb",
        default=False,
        is_flag=True,
        help="Skip PDB100 database search.",
    )
    @click.option(
        "-t",
        "--threads",
        default=1,
        type=int,
        show_default=True,
        help="Number of threads to use.",
    )
    @click.option(
        "--overwrite",
        default=False,
        is_flag=True,
        help="Overwrite existing files.",
    )
    @click.option(
        "--top-k",
        default=5,
        type=int,
        show_default=True,
        help="Number of top MMseqs2 hits to save.",
    )
    @click.option(
        "--mmseqs-min-coverage",
        default=0.9,
        type=float,
        show_default=True,
        help=
        "Minimum coverage for MMseqs2 alignment for both query and target sequences.",
    )
    @click.option(
        "--mmseqs-min-identity",
        default=0.5,
        type=float,
        show_default=True,
        help="Minimum identity for MMseqs2 alignment.",
    )
    @click.option(
        "--mmseqs-max-evalue",
        default=0.001,
        type=float,
        show_default=True,
        help="Maximum e-value for MMseqs2 alignment.",
    )
    @click.option(
        "--mmseqs-min-bitscore",
        default=0,
        type=float,
        show_default=True,
        help="Minimum bitscore for MMseqs2 alignment.",
    )
    @click.option(
        "--max-length",
        default=None,
        type=int,
        help="Maximum length of the protein sequence.",
    )
    @click.option(
        "--min-length",
        default=None,
        type=int,
        help="Minimum length of the protein sequence.",
    )
    @click.option(
        "-s",
        "--mmseqs-sensitivity",
        default=5.7,
        type=click.FloatRange(1, 7.5),
        show_default=True,
        help="Sensitivity of the MMseqs2 search.",
    )
    @click.option(
        "-d",
        "--db-path",
        required=False,
        type=click.Path(exists=True,
                        dir_okay=False,
                        file_okay=True,
                        path_type=Path),
        multiple=True,
        help="Path to a structures database compessed with FoldComp.",
    )
    @click.option(
        "-o",
        "--output",
        required=True,
        type=click.Path(exists=False, path_type=Path),
        help="Path to output file or directory.",
    )
    @click.option(
        "-i",
        "--input",
        required=True,
        type=click.Path(exists=True,
                        dir_okay=False,
                        readable=True,
                        path_type=Path),
        help="Path to an input protein sequences (FASTA file, may be gzipped).",
    )
    @wraps(function)
    def wrapper(*args, **kwargs):
        return function(*args, **kwargs)

    return wrapper


@click.group()
@click.option("--debug/--no-debug", default=False)
@click.version_option(version=importlib.metadata.version("mDeepFRI"))
def main(debug):
    """mDeepFRI"""

    loggers = [
        logging.getLogger(name) for name in logging.root.manager.loggerDict
    ]
    for log in loggers:
        if debug:
            log.setLevel(logging.DEBUG)
        else:
            log.setLevel(logging.INFO)

    patch_usage_error()
    setup_logging(debug)


# CLI commands
@main.command
@click.option(
    "-o",
    "--output",
    required=True,
    type=click.Path(file_okay=False,
                    dir_okay=True,
                    writable=True,
                    path_type=Path),
    help="Path to folder where the model weights will be downloaded.",
)
@click.option("-v",
              "--version",
              required=True,
              type=click.Choice(["1.0", "1.1"]),
              help="Version of the model.")
def get_models(output, version):
    """Download model weights for mDeepFRI."""

    logger.info("Downloading DeepFRI models.")
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    download_model_weights(output_path, version)
    generate_config_json(output_path, version)
    logger.info(f"DeepFRI models v{version} downloaded to {output_path}.")


@main.command
@click.option(
    "-w",
    "--weights_path",
    required=True,
    type=click.Path(exists=True,
                    dir_okay=True,
                    file_okay=False,
                    path_type=Path),
    help="Path to a folder containing model weights.",
)
@click.option(
    "-v",
    "--version",
    required=True,
    type=click.Choice(["1.0", "1.1"]),
    help="Version of the model.",
)
def generate_config(weights_path, version):
    """
    Generate a config file for mDeepFRI.
    This is used only when the model weights are downloaded manually.
    """

    logger.info("Generating config file for mDeepFRI.")
    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise UsageError(f"Path {weights_path} does not exist.")
    generate_config_json(weights_path, version)
    logger.info(f"Config file generated in {weights_path}.")


@main.command
@search_options
@click.pass_context
def search_databases(ctx, input, output, db_path, mmseqs_sensitivity,
                     min_length, max_length, min_bits, max_eval, min_ident,
                     min_coverage, top_k, overwrite, threads, skip_pdb,
                     tmpdir):
    """
    Hierarchically search FoldComp databases for similar proteins with
    MMseqs2. Based on the thresholds from https://doi.org/10.1038/s41586-023-06510-w.
    """

    # write command parameters to log
    log_command_params(ctx)

    query_file = load_query_file(query_file=input,
                                 min_length=min_length,
                                 max_length=max_length)
    hierarchical_database_search(query_file=query_file,
                                 databases=db_path,
                                 output_path=output,
                                 mmseqs_sensitivity=mmseqs_sensitivity,
                                 min_seq_len=min_length,
                                 max_seq_len=max_length,
                                 min_bits=min_bits,
                                 max_eval=max_eval,
                                 min_ident=min_ident,
                                 min_coverage=min_coverage,
                                 top_k=top_k,
                                 skip_pdb=skip_pdb,
                                 overwrite=overwrite,
                                 tmpdir=tmpdir,
                                 threads=threads)


@main.command()
@search_options
@click.option(
    "-w",
    "--weights",
    required=True,
    type=click.Path(exists=True,
                    dir_okay=True,
                    file_okay=False,
                    path_type=Path),
    help="Path to a folder containing model weights.",
)
@click.option(
    "-p",
    "--processing-modes",
    default=["bp", "cc", "ec", "mf"],
    type=click.Choice(["bp", "cc", "ec", "mf"]),
    multiple=True,
    help="Processing modes. Default is all"
    "(biological process, cellular component, enzyme comission, molecular function).",
)
@click.option(
    "-a",
    "--angstrom-contact-thresh",
    default=6,
    type=float,
    help="Angstrom contact threshold. Default is 6.",
)
@click.option(
    "--generate-contacts",
    default=2,
    type=int,
    help="Gap fill threshold during contact map alignment.",
)
@click.option(
    "--alignment-gap-open",
    default=10,
    type=int,
    help="Gap open penalty for contact map alignment.",
)
@click.option(
    "--alignment-gap-extend",
    default=1,
    type=int,
    help="Gap extend penalty for contact map alignment.",
)
@click.option(
    "--remove-intermediate",
    default=False,
    type=bool,
    is_flag=True,
    help="Remove intermediate files.",
)
@click.option(
    "--save-structures",
    default=False,
    type=bool,
    is_flag=True,
    help="Save structures of the top hits.",
)
@click.option(
    "--save-cmaps",
    default=False,
    type=bool,
    is_flag=True,
    help="Save contact maps of the top hits.",
)
@click.option(
    "--skip-matrix",
    default=False,
    type=bool,
    is_flag=True,
    help="Skip writing prediction matrix files (saves disk space).",
)
@click.option(
    "--scoring-matrix",
    default="VTML80",
    type=str,
    show_default=True,
    help="Scoring matrix for sequence alignment (e.g., VTML80, BLOSUM62).",
)
@click.pass_context
def predict_function(ctx, input, db_path, weights, output, processing_modes,
                     angstrom_contact_thresh, generate_contacts,
                     mmseqs_sensitivity, mmseqs_min_bitscore,
                     mmseqs_max_evalue, mmseqs_min_identity,
                     mmseqs_min_coverage, top_k, alignment_gap_open,
                     alignment_gap_extend, remove_intermediate, overwrite,
                     threads, skip_pdb, min_length, max_length, tmpdir,
                     save_structures, save_cmaps, skip_matrix, scoring_matrix):
    """Predict protein function from sequence."""

    logger.info("Starting Metagenomic-DeepFRI.")

    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    # write command parameters to log
    log_command_params(ctx)

    query_file = load_query_file(query_file=input,
                                 min_length=min_length,
                                 max_length=max_length)

    deepfri_dbs = hierarchical_database_search(
        query_file=query_file,
        output_path=output_path / "database_search",
        databases=db_path,
        mmseqs_sensitivity=mmseqs_sensitivity,
        min_bits=mmseqs_min_bitscore,
        max_eval=mmseqs_max_evalue,
        min_ident=mmseqs_min_identity,
        min_coverage=mmseqs_min_coverage,
        top_k=top_k,
        skip_pdb=skip_pdb,
        overwrite=overwrite,
        tmpdir=tmpdir,
        threads=threads)

    # refresh query file
    # hierarchical_database_search filters out aligned sequences
    # to avoid redundant alignments

    query_file = load_query_file(query_file=input,
                                 min_length=min_length,
                                 max_length=max_length)

    # Reconstruct command string for metadata
    command_str = " ".join(sys.argv)
    version_str = importlib.metadata.version("mDeepFRI")

    predict_protein_function(
        query_file=query_file,
        databases=deepfri_dbs,
        weights=weights,
        output_path=output_path,
        deepfri_processing_modes=processing_modes,
        angstrom_contact_threshold=angstrom_contact_thresh,
        generate_contacts=generate_contacts,
        alignment_gap_open=alignment_gap_open,
        alignment_gap_continuation=alignment_gap_extend,
        remove_intermediate=remove_intermediate,
        save_structures=save_structures,
        save_cmaps=save_cmaps,
        skip_matrix=skip_matrix,
        scoring_matrix=scoring_matrix,
        command_str=command_str,
        version=version_str)


@main.command()
@click.option("--input_dir",
              "-i",
              type=click.Path(exists=True),
              required=True,
              help="Directory containing PDB or mmCIF files.")
@click.option("--output_dir",
              "-o",
              type=click.Path(),
              required=True,
              help="Directory to save computed contact maps.")
@click.option("--threshold",
              "-t",
              default=6.0,
              show_default=True,
              help="Distance threshold in Å for contact map.")
def make_cmaps(input_dir, output_dir, threshold):
    "Compute CA contact maps for all PDB/mmCIF files in a directory."
    os.makedirs(output_dir, exist_ok=True)
    for fname in os.listdir(input_dir):
        if not fname.endswith((".pdb", ".cif")):
            continue
        filetype = "pdb" if fname.endswith(".pdb") else "mmcif"
        with open(os.path.join(input_dir, fname)) as f:
            structure_str = f.read()
        residues, coords = get_residues_coordinates(load_structure(
            structure_str, filetype),
                                                    chain="A")
        cmap = calculate_contact_map(coords, threshold)
        np.save(os.path.join(output_dir, fname.replace(".pdb", "_cmap.npy")),
                cmap)


if __name__ == "__main__":
    main()
