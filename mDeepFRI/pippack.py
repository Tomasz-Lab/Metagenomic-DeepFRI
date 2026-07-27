"""External PIPPack side-chain packing for carved backbone PDBs.

PIPPack is invoked via subprocesses using a user-provided install directory
and Python interpreter. Each worker is single-threaded; parallelism comes from
running many workers (CPU) or a small number of GPU workers.
"""

from __future__ import annotations

import json
import logging
import multiprocessing
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

from mDeepFRI.bio_utils import (BACKBONE_ATOM_NAMES, BackboneValidationResult,
                                _init_parallel_worker, extract_pdb_header,
                                reattach_pdb_header, validate_backbone_pdb)

logger = logging.getLogger(__name__)

WORKER_SCRIPT = Path(__file__).resolve().parent / "scripts" / "pippack_worker.py"
DEFAULT_MODEL = "pippack_model_1"
# PIPPack drops these residue types; we stash their backbone and reinsert after packing.
NONSTANDARD_RES_NAMES = frozenset({"UNK", "SEC", "PYL"})
ResidueKey = Tuple[str, int]


class PippackConfigError(ValueError):
    """Raised when PIPPack is missing or misconfigured."""


def resolve_pippack_dir(pippack_dir: Optional[str] = None) -> Path:
    """Resolve PIPPack root from argument or ``PIPPACK_DIR``."""
    candidate = pippack_dir or os.environ.get("PIPPACK_DIR")
    if not candidate:
        raise PippackConfigError(
            "PIPPack is required for --carve-pdbs. Provide --pippack-dir or "
            "set PIPPACK_DIR to a PIPPack checkout "
            "(https://github.com/Kuhlman-Lab/PIPPack). "
            "Create the conda env from PIPPack/env/pippack_env.yaml and ensure "
            "model_weights/ is present.")
    path = Path(candidate).expanduser().resolve()
    if not (path / "inference.py").exists():
        raise PippackConfigError(
            f"PIPPack directory {path} does not contain inference.py.")
    weights = path / "model_weights"
    if not weights.is_dir():
        raise PippackConfigError(
            f"PIPPack model_weights/ not found under {path}.")
    return path


def resolve_pippack_python(pippack_python: Optional[str] = None) -> str:
    """Return interpreter used to run the PIPPack worker."""
    if pippack_python:
        return str(Path(pippack_python).expanduser())
    env_python = os.environ.get("PIPPACK_PYTHON")
    if env_python:
        return str(Path(env_python).expanduser())
    return shutil.which("python") or sys.executable


def probe_pippack_torch(pippack_python: Optional[str] = None) -> str:
    """
    Verify the PIPPack interpreter can import torch.

    Returns:
        Absolute path of the probed Python interpreter.

    Raises:
        PippackConfigError: If torch cannot be imported.
    """
    python_bin = resolve_pippack_python(pippack_python)
    probe = subprocess.run(
        [python_bin, "-c", "import torch; print(torch.__version__)"],
        capture_output=True,
        text=True,
        check=False,
    )
    if probe.returncode != 0:
        raise PippackConfigError(
            f"PIPPack Python interpreter cannot import torch: {python_bin}\n"
            f"{(probe.stderr or probe.stdout).strip()}\n"
            "Pass --pippack-python pointing at the PIPPack conda/env "
            "interpreter (or set PIPPACK_PYTHON).")
    return python_bin


def _default_worker_count(device: str,
                          threads: int,
                          pippack_workers: Optional[int]) -> int:
    if pippack_workers is not None:
        return max(1, pippack_workers)
    if device == "gpu":
        return 1
    return max(1, threads)


def _shard_paths(paths: Sequence[Path], worker_count: int) -> List[List[Path]]:
    shards = [list(paths[i::worker_count]) for i in range(worker_count)]
    return [shard for shard in shards if shard]


class _PrepResult(NamedTuple):
    path: Path
    ok: bool
    validation: BackboneValidationResult
    header: str
    stashed: Dict[ResidueKey, List[str]]
    stash_counts: Dict[str, int]


def _validate_and_prepare_pdb(pdb_path: Path) -> _PrepResult:
    """Validate one carved PDB and prepare header/stash data if packable."""
    pdb_text = pdb_path.read_text(encoding="utf-8")
    validation = validate_backbone_pdb(pdb_text)
    if not validation.ok:
        return _PrepResult(pdb_path, False, validation, "", {}, {})
    header = extract_pdb_header(pdb_text)
    stashed, stash_counts = stash_nonstandard_backbone(pdb_text)
    return _PrepResult(pdb_path, True, validation, header, stashed, stash_counts)


def _run_backbone_validation(
        pdb_files: Sequence[Path],
        threads: int,
) -> Tuple[List[Path], Dict[str, str], Dict[str, Dict[ResidueKey, List[str]]],
           Dict[str, Dict[str, int]], int]:
    """
    Validate carved PDBs in parallel and collect packing metadata.

    Uses ``threads`` worker processes (same pool size as carving). Returns
    packable paths in the original ``pdb_files`` order.
    """
    total = len(pdb_files)
    worker_count = max(1, min(threads, total))
    logger.info(
        "Validating backbone completeness for %d carved PDB(s) "
        "using %d worker process(es)...",
        total,
        worker_count,
    )

    if worker_count == 1:
        prep_results = [_validate_and_prepare_pdb(path) for path in pdb_files]
    else:
        chunksize = max(1, total // (worker_count * 4))
        ctx = multiprocessing.get_context("fork")
        with ProcessPoolExecutor(max_workers=worker_count,
                                 mp_context=ctx,
                                 initializer=_init_parallel_worker) as executor:
            prep_results = list(
                executor.map(_validate_and_prepare_pdb,
                             pdb_files,
                             chunksize=chunksize))

    packable: List[Path] = []
    headers: Dict[str, str] = {}
    stashes: Dict[str, Dict[ResidueKey, List[str]]] = {}
    stash_counts_by_name: Dict[str, Dict[str, int]] = {}
    skipped = 0
    for result in prep_results:
        if not result.ok:
            skipped += 1
            _log_incomplete_backbone(result.path.stem, result.validation)
            continue
        headers[result.path.stem] = result.header
        stashes[result.path.stem] = result.stashed
        stash_counts_by_name[result.path.stem] = result.stash_counts
        packable.append(result.path)

    logger.info("  Backbone validation done: %d packable, %d skipped.",
                len(packable), skipped)
    return packable, headers, stashes, stash_counts_by_name, skipped


def _count_ca_residues(pdb_text: str) -> int:
    count = 0
    for line in pdb_text.splitlines():
        if line.startswith(("ATOM", "HETATM")) and line[12:16].strip() == "CA":
            count += 1
    return count


def _pdb_res_name(line: str) -> str:
    return line[17:20].strip() if len(line) >= 20 else ""


def _pdb_chain_id(line: str) -> str:
    return line[21:22] if len(line) >= 22 else " "


def _pdb_res_id(line: str) -> int:
    return int(line[22:26]) if len(line) >= 26 else 0


def _pdb_atom_name(line: str) -> str:
    return line[12:16].strip() if len(line) >= 16 else ""


def _renumber_atom_serial(line: str, serial: int) -> str:
    """Rewrite columns 7-11 with a new atom serial number."""
    padded = line.ljust(80)
    return f"{padded[:6]}{serial:5d}{padded[11:]}".rstrip()


def stash_nonstandard_backbone(
        pdb_text: str
) -> Tuple[Dict[ResidueKey, List[str]], Dict[str, int]]:
    """
    Extract N/CA/C/O atoms for UNK/SEC/PYL residues from a carved PDB.

    Returns:
        Mapping of ``(chain_id, res_id)`` to backbone ATOM/HETATM lines, and
        a per-residue-name count of stashed residues.
    """
    stashed: Dict[ResidueKey, List[str]] = {}
    counts = {"UNK": 0, "SEC": 0, "PYL": 0}
    for line in pdb_text.splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        res_name = _pdb_res_name(line)
        if res_name not in NONSTANDARD_RES_NAMES:
            continue
        if _pdb_atom_name(line) not in BACKBONE_ATOM_NAMES:
            continue
        key = (_pdb_chain_id(line), _pdb_res_id(line))
        if key not in stashed:
            stashed[key] = []
            counts[res_name] = counts.get(res_name, 0) + 1
        stashed[key].append(line)
    return stashed, counts


def reinsert_nonstandard_backbone(
        packed_text: str,
        stashed: Dict[ResidueKey, List[str]],
) -> str:
    """
    Merge PIPPack output with stashed non-standard backbone residues.

    Residues are emitted in ``(chain_id, res_id)`` order. Atom serial numbers
    are renumbered sequentially.

    Preserves a valid PDB record order from the PIPPack template::

        MODEL (optional)
        ATOM / HETATM ...
        TER
        ENDMDL (optional)
        END
    """
    if not stashed:
        return packed_text

    header: List[str] = []
    packed_groups: Dict[ResidueKey, List[str]] = {}
    trailer: List[str] = []
    seen_atom = False
    for line in packed_text.splitlines():
        if line.startswith(("ATOM", "HETATM")):
            seen_atom = True
            key = (_pdb_chain_id(line), _pdb_res_id(line))
            packed_groups.setdefault(key, []).append(line)
        elif not seen_atom:
            # MODEL and any other pre-ATOM records stay at the top.
            header.append(line)
        else:
            trailer.append(line)

    overlap = set(packed_groups) & set(stashed)
    if overlap:
        overlap_text = ", ".join(f"{c}:{r}" for c, r in sorted(overlap)[:5])
        raise ValueError(
            "Stashed non-standard residues already present in PIPPack output: "
            f"{overlap_text}")

    all_keys = sorted(set(packed_groups) | set(stashed),
                      key=lambda item: (item[0], item[1]))
    out_lines: List[str] = [line for line in header if line.strip()]
    serial = 1
    last_res_name = ""
    last_chain = "A"
    last_res_id = 0
    for key in all_keys:
        residue_lines = packed_groups.get(key) or stashed[key]
        for line in residue_lines:
            out_lines.append(_renumber_atom_serial(line, serial))
            serial += 1
            last_res_name = _pdb_res_name(line)
            last_chain = _pdb_chain_id(line)
            last_res_id = _pdb_res_id(line)

    for line in trailer:
        if not line.strip():
            continue
        if line.startswith("TER"):
            # Keep TER after atoms; refresh serial / terminal residue identity.
            out_lines.append(
                f"TER   {serial:5d}      {last_res_name:>3} "
                f"{last_chain}{last_res_id:4d}")
            serial += 1
        else:
            out_lines.append(line)

    return "\n".join(out_lines) + "\n"


def _log_incomplete_backbone(query_id: str, validation) -> None:
    examples = validation.incomplete_residues[:5]
    example_text = ", ".join(
        f"res_id={res_id} missing {','.join(missing) or 'backbone'}"
        for res_id, missing in examples)
    logger.warning(
        "%s: CA-only / incomplete backbone template residue(s) "
        "(%s). Side-chain recreation not possible. Provide a valid "
        "full-backbone template. Keeping backbone-only PDB.",
        query_id,
        example_text or "no complete N/CA/C/O residues",
    )


def _run_worker(
        *,
        python_bin: str,
        pippack_dir: Path,
        weights_path: Path,
        model_name: str,
        device: str,
        n_recycle: int,
        temperature: float,
        use_resample: bool,
        seed: int,
        pdb_paths: Sequence[Path],
        packed_dir: Path,
        worker_index: int,
) -> Tuple[int, List[str]]:
    """Launch one PIPPack worker subprocess for a PDB shard."""
    with tempfile.TemporaryDirectory(prefix=f"pippack_worker_{worker_index}_") as tmp:
        tmp_path = Path(tmp)
        list_path = tmp_path / "pdb_list.json"
        list_path.write_text(
            json.dumps([str(path) for path in pdb_paths]),
            encoding="utf-8",
        )
        worker_out = packed_dir / f"_worker_{worker_index}"
        worker_out.mkdir(parents=True, exist_ok=True)

        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = "1"
        env["OPENBLAS_NUM_THREADS"] = "1"
        env["MKL_NUM_THREADS"] = "1"
        env["NUMEXPR_NUM_THREADS"] = "1"
        env["VECLIB_MAXIMUM_THREADS"] = "1"
        env["PIPPACK_THREADS"] = "1"
        if device == "cpu":
            env["CUDA_VISIBLE_DEVICES"] = ""

        cmd = [
            python_bin,
            str(WORKER_SCRIPT),
            "--pippack-dir",
            str(pippack_dir),
            "--weights-path",
            str(weights_path),
            "--model-name",
            model_name,
            "--output-dir",
            str(worker_out),
            "--device",
            device,
            "--n-recycle",
            str(n_recycle),
            "--temperature",
            str(temperature),
            "--seed",
            str(seed),
            "--pdb-list",
            str(list_path),
        ]
        if use_resample:
            cmd.append("--use-resample")
        completed = subprocess.run(cmd,
                                   env=env,
                                   capture_output=True,
                                   text=True,
                                   check=False)
        if completed.stdout.strip():
            logger.debug("PIPPack worker %d stdout:\n%s", worker_index,
                         completed.stdout)
        if completed.stderr.strip():
            # Always keep a short ERROR/WARNING trail for failed workers; full
            # stderr stays at DEBUG to avoid flooding multi-structure runs.
            level = logging.WARNING if completed.returncode != 0 else logging.DEBUG
            logger.log(level,
                       "PIPPack worker %d stderr (exit %d):\n%s",
                       worker_index,
                       completed.returncode,
                       completed.stderr[-2000:])

        failures: List[str] = []
        status_path = worker_out / "_worker_status.json"
        if status_path.exists():
            status = json.loads(status_path.read_text(encoding="utf-8"))
            failures = [item["name"] for item in status.get("failures", [])]
        elif completed.returncode != 0:
            failures = [path.stem for path in pdb_paths]
            logger.warning(
                "PIPPack worker %d failed (exit %d). Keeping backbone-only "
                "PDBs for this shard.",
                worker_index,
                completed.returncode,
            )

        for packed_file in worker_out.glob("*.pdb"):
            dest = packed_dir / packed_file.name
            shutil.move(str(packed_file), str(dest))

        shutil.rmtree(worker_out, ignore_errors=True)
        return completed.returncode, failures


def pack_carved_structures(
        carve_dir: Path,
        *,
        pippack_dir: Optional[str] = None,
        pippack_python: Optional[str] = None,
        device: str = "cpu",
        threads: int = 1,
        pippack_workers: Optional[int] = None,
        model_name: str = DEFAULT_MODEL,
        n_recycle: int = 3,
        temperature: float = 0.0,
        use_resample: bool = False,
        seed: int = 42,
) -> Tuple[int, int]:
    """
    Validate carved backbone PDBs and rebuild side chains with PIPPack.

    Incomplete / CA-only structures are skipped (backbone PDB kept). UNK/SEC/PYL
    backbone atoms are stashed before packing (PIPPack drops them) and
    reinserted afterward. Packed outputs replace the carved files in place after
    REMARK/SEQRES headers are reattached.

    Returns:
        Tuple of ``(packed_count, skipped_count)``.
    """
    carve_dir = Path(carve_dir)
    pdb_files = sorted(carve_dir.glob("*.pdb"))
    if not pdb_files:
        logger.warning("No carved PDB files found in %s; skipping PIPPack.",
                       carve_dir)
        return 0, 0

    pippack_root = resolve_pippack_dir(pippack_dir)
    python_bin = probe_pippack_torch(pippack_python)
    weights_path = pippack_root / "model_weights"
    if not (weights_path / f"{model_name}_ckpt.pt").exists():
        raise PippackConfigError(
            f"PIPPack checkpoint {model_name}_ckpt.pt not found in {weights_path}."
        )

    if device not in {"cpu", "gpu"}:
        raise ValueError(f"Unsupported PIPPack device: {device!r}")

    logger.info("Using PIPPack python=%s device=%s model=%s", python_bin,
                device, model_name)

    packable, headers, stashes, stash_counts_by_name, skipped = (
        _run_backbone_validation(pdb_files, threads))

    if not packable:
        logger.warning(
            "No packable carved PDBs (all incomplete backbone). "
            "Skipping PIPPack.")
        return 0, skipped

    worker_count = _default_worker_count(device, threads, pippack_workers)
    worker_count = min(worker_count, len(packable))
    shards = _shard_paths(packable, worker_count)
    stashed_residue_count = sum(len(items) for items in stashes.values())
    logger.info(
        "Starting PIPPack packing of %d structure(s) with %d single-threaded "
        "worker(s) on %s (skipped %d incomplete; %d non-standard backbone "
        "residue(s) will be reinserted after packing; n_recycle=%d, "
        "temperature=%s, use_resample=%s). Workers run until all shards "
        "finish — this is usually the longest step.",
        len(packable),
        len(shards),
        device,
        skipped,
        stashed_residue_count,
        n_recycle,
        temperature,
        use_resample,
    )

    with tempfile.TemporaryDirectory(prefix="pippack_packed_") as tmp:
        packed_dir = Path(tmp)
        failed_names = set()

        # Thread pool only orchestrates subprocesses; each worker is its own process.
        with ThreadPoolExecutor(max_workers=len(shards)) as executor:
            futures = {
                executor.submit(
                    _run_worker,
                    python_bin=python_bin,
                    pippack_dir=pippack_root,
                    weights_path=weights_path,
                    model_name=model_name,
                    device=device,
                    n_recycle=n_recycle,
                    temperature=temperature,
                    use_resample=use_resample,
                    seed=seed,
                    pdb_paths=shard,
                    packed_dir=packed_dir,
                    worker_index=index,
                ): (index, shard)
                for index, shard in enumerate(shards)
            }
            finished_workers = 0
            for future in as_completed(futures):
                worker_index, shard = futures[future]
                _, failures = future.result()
                failed_names.update(failures)
                finished_workers += 1
                logger.info(
                    "  PIPPack worker %d/%d finished (%d structure(s) in shard, "
                    "%d failure(s) in shard).",
                    finished_workers,
                    len(shards),
                    len(shard),
                    len(failures),
                )

        logger.info("Merging packed PDBs and reattaching SEQRES/REMARK headers...")
        packed_count = 0
        total_input_ca = 0
        reinserted_counts = {"UNK": 0, "SEC": 0, "PYL": 0}
        for merge_index, pdb_path in enumerate(packable, start=1):
            name = pdb_path.stem
            packed_path = packed_dir / f"{name}.pdb"
            if name in failed_names or not packed_path.exists():
                logger.warning(
                    "%s: PIPPack packing failed; keeping backbone-only PDB.",
                    name,
                )
                continue

            input_text = pdb_path.read_text(encoding="utf-8")
            input_ca = _count_ca_residues(input_text)
            stashed = stashes.get(name, {})
            packed_text = packed_path.read_text(encoding="utf-8")
            output_ca = _count_ca_residues(packed_text)
            expected_packed_ca = input_ca - len(stashed)
            if output_ca < expected_packed_ca:
                logger.warning(
                    "%s: PIPPack output has fewer CA residues than expected "
                    "(%d < %d after accounting for %d stashed non-standard); "
                    "keeping backbone-only PDB.",
                    name,
                    output_ca,
                    expected_packed_ca,
                    len(stashed),
                )
                continue

            try:
                merged_atoms = reinsert_nonstandard_backbone(
                    packed_text, stashed)
            except ValueError as exc:
                logger.warning(
                    "%s: non-standard backbone reinsertion failed (%s); "
                    "keeping backbone-only PDB.",
                    name,
                    exc,
                )
                continue

            merged_ca = _count_ca_residues(merged_atoms)
            if merged_ca != input_ca:
                logger.warning(
                    "%s: after UNK/SEC/PYL reinsertion CA count mismatch "
                    "(%d != %d); keeping backbone-only PDB.",
                    name,
                    merged_ca,
                    input_ca,
                )
                continue

            for res_name, count in stash_counts_by_name.get(name, {}).items():
                reinserted_counts[res_name] = (
                    reinserted_counts.get(res_name, 0) + count)
            total_input_ca += input_ca

            merged = reattach_pdb_header(headers[name], merged_atoms)
            pdb_path.write_text(merged, encoding="utf-8")
            packed_count += 1
            if merge_index % 2000 == 0 or merge_index == len(packable):
                logger.info("  Merge progress: %d/%d.", merge_index,
                            len(packable))

    unk_count = reinserted_counts.get("UNK", 0)
    unk_pct = (100.0 * unk_count / total_input_ca) if total_input_ca else 0.0
    logger.info(
        "PIPPack packed %d structure(s); skipped %d incomplete. "
        "Reinserted UNK backbone for %d residue(s) (%.2f%% of %d packed "
        "residues); SEC=%d, PYL=%d.",
        packed_count,
        skipped,
        unk_count,
        unk_pct,
        total_input_ca,
        reinserted_counts.get("SEC", 0),
        reinserted_counts.get("PYL", 0),
    )
    return packed_count, skipped
