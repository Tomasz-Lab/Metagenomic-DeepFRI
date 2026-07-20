#!/usr/bin/env python3
"""Pack a shard of backbone PDBs with PIPPack (single-threaded worker).

Designed to be launched as a subprocess by ``mDeepFRI.pippack``. Expects the
PIPPack repository on ``sys.path`` via ``--pippack-dir``.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

# PIPPack defaults from config/inference.yaml
DEFAULT_RESAMPLE_ARGS = {
    "sample_temp": 0.5,
    "clash_overlap_tolerance": 0.6,
    "pro_tolerance_factor": 12,
    "max_iters": 10,
    "metropolis_temp": 0.000005,
}


def _pin_threads() -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["PIPPACK_THREADS"] = "1"


def main() -> int:
    _pin_threads()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pippack-dir", required=True)
    parser.add_argument("--weights-path", required=True)
    parser.add_argument("--model-name", default="pippack_model_1")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--n-recycle", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--use-resample", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resample-args-json", default=None,
                        help="Optional JSON dict overriding PIPPack resample_args.")
    parser.add_argument("--pdb-list", required=True,
                        help="JSON file listing absolute PDB paths to pack.")
    args = parser.parse_args()

    pippack_dir = Path(args.pippack_dir).resolve()
    if not (pippack_dir / "inference.py").exists():
        print(f"PIPPack inference.py not found in {pippack_dir}",
              file=sys.stderr)
        return 2

    sys.path.insert(0, str(pippack_dir))

    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    import torch
    import hydra
    import lightning
    from utils.train_utils import load_checkpoint
    from data.protein import from_pdb_file
    from data.top2018_dataset import transform_structure, collate_fn
    from inference import sample_epoch, pdbs_from_prediction

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    lightning.seed_everything(args.seed)

    device = torch.device(
        "cuda:0" if args.device == "gpu" and torch.cuda.is_available() else "cpu")

    weights_path = Path(args.weights_path)
    config_path = weights_path / f"{args.model_name}_config.pickle"
    checkpoint_path = weights_path / f"{args.model_name}_ckpt.pt"
    if not config_path.exists() or not checkpoint_path.exists():
        print(f"Missing PIPPack weights for {args.model_name} in {weights_path}",
              file=sys.stderr)
        return 2

    with open(config_path, "rb") as handle:
        exp_cfg = pickle.load(handle)
    model = hydra.utils.instantiate(exp_cfg.model).to(device)
    load_checkpoint(str(checkpoint_path), model)
    model.eval()

    resample_args = dict(DEFAULT_RESAMPLE_ARGS)
    if args.resample_args_json:
        resample_args.update(json.loads(args.resample_args_json))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.pdb_list, "r", encoding="utf-8") as handle:
        pdb_paths = [Path(path) for path in json.load(handle)]

    failures = []
    for pdb_path in pdb_paths:
        name = pdb_path.stem
        try:
            protein = vars(from_pdb_file(str(pdb_path), mse_to_met=True))
            graph = transform_structure(protein,
                                        exp_cfg.model.n_chi_bins,
                                        sc_d_mask_from_seq=True)
            batch = collate_fn([graph])
            with torch.no_grad():
                results = sample_epoch(model,
                                       batch,
                                       temperature=args.temperature,
                                       device=device,
                                       n_recycle=args.n_recycle,
                                       resample=args.use_resample,
                                       resample_args=resample_args)
            packed = pdbs_from_prediction(results)[0]
            (output_dir / f"{name}.pdb").write_text(packed, encoding="utf-8")
        except Exception as exc:  # noqa: BLE001 - surface to parent via JSON
            failures.append({"name": name, "error": repr(exc)})
            print(f"[pippack_worker] FAILED {name}: {exc!r}",
                  file=sys.stderr,
                  flush=True)

    status_path = output_dir / "_worker_status.json"
    status_path.write_text(json.dumps({"failures": failures}), encoding="utf-8")
    return 1 if failures and len(failures) == len(pdb_paths) else 0


if __name__ == "__main__":
    raise SystemExit(main())
