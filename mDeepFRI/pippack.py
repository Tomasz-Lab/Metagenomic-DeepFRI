"""External PIPPack side-chain packing for carved backbone PDBs.

PIPPack is invoked via subprocesses using a user-provided install directory
and Python interpreter. Each worker is single-threaded; parallelism comes from
running many workers (CPU) or a small number of GPU workers.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from mDeepFRI.bio_utils import (extract_pdb_header, reattach_pdb_header,
                                validate_backbone_pdb)

logger = logging.getLogger(__name__)

WORKER_SCRIPT = Path(__file__).resolve().parent / "scripts" / "pippack_worker.py"
DEFAULT_MODEL = "pippack_model_1"


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


def _count_ca_residues(pdb_text: str) -> int:
    count = 0
    for line in pdb_text.splitlines():
        if line.startswith("ATOM") and line[12:16].strip() == "CA":
            count += 1
    return count


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

    Incomplete / CA-only structures are skipped (backbone PDB kept). Packed
    outputs replace the carved files in place after REMARK/SEQRES headers are
    reattached.

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
    python_bin = resolve_pippack_python(pippack_python)
    weights_path = pippack_root / "model_weights"
    if not (weights_path / f"{model_name}_ckpt.pt").exists():
        raise PippackConfigError(
            f"PIPPack checkpoint {model_name}_ckpt.pt not found in {weights_path}."
        )

    # Fail fast if the chosen interpreter cannot import torch (common when
    # --pippack-python is omitted and the mDeepFRI env is used by mistake).
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

    if device not in {"cpu", "gpu"}:
        raise ValueError(f"Unsupported PIPPack device: {device!r}")

    logger.info("Using PIPPack python=%s device=%s model=%s", python_bin,
                device, model_name)

    logger.info("Validating backbone completeness for %d carved PDB(s)...",
                len(pdb_files))
    packable: List[Path] = []
    headers = {}
    skipped = 0
    for index, pdb_path in enumerate(pdb_files, start=1):
        pdb_text = pdb_path.read_text(encoding="utf-8")
        validation = validate_backbone_pdb(pdb_text)
        if not validation.ok:
            skipped += 1
            _log_incomplete_backbone(pdb_path.stem, validation)
            continue
        headers[pdb_path.stem] = extract_pdb_header(pdb_text)
        packable.append(pdb_path)
        if index % 1000 == 0 or index == len(pdb_files):
            logger.info("  Backbone validation progress: %d/%d "
                        "(%d packable, %d skipped).",
                        index, len(pdb_files), len(packable), skipped)

    if not packable:
        logger.warning(
            "No packable carved PDBs (all incomplete backbone). "
            "Skipping PIPPack.")
        return 0, skipped

    worker_count = _default_worker_count(device, threads, pippack_workers)
    worker_count = min(worker_count, len(packable))
    shards = _shard_paths(packable, worker_count)
    logger.info(
        "Starting PIPPack packing of %d structure(s) with %d single-threaded "
        "worker(s) on %s (skipped %d incomplete; n_recycle=%d, "
        "temperature=%s, use_resample=%s). Workers run until all shards "
        "finish — this is usually the longest step.",
        len(packable),
        len(shards),
        device,
        skipped,
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
        for merge_index, pdb_path in enumerate(packable, start=1):
            name = pdb_path.stem
            packed_path = packed_dir / f"{name}.pdb"
            if name in failed_names or not packed_path.exists():
                logger.warning(
                    "%s: PIPPack packing failed; keeping backbone-only PDB.",
                    name,
                )
                continue

            packed_text = packed_path.read_text(encoding="utf-8")
            input_ca = _count_ca_residues(pdb_path.read_text(encoding="utf-8"))
            output_ca = _count_ca_residues(packed_text)
            if output_ca < input_ca:
                logger.warning(
                    "%s: PIPPack output has fewer CA residues (%d < %d); "
                    "keeping backbone-only PDB.",
                    name,
                    output_ca,
                    input_ca,
                )
                continue

            merged = reattach_pdb_header(headers[name], packed_text)
            pdb_path.write_text(merged, encoding="utf-8")
            packed_count += 1
            if merge_index % 2000 == 0 or merge_index == len(packable):
                logger.info("  Merge progress: %d/%d.", merge_index,
                            len(packable))

    logger.info("PIPPack packed %d structure(s); skipped %d incomplete.",
                packed_count, skipped)
    return packed_count, skipped
