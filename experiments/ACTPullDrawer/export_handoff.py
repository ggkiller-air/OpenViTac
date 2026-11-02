#!/usr/bin/env python3
"""Publish an atomic, self-contained ACTPullDrawer package for the independent 3090."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    if not (args.run / "COMPLETE.json").is_file() and not args.allow_partial:
        raise ValueError("Final handoff requires COMPLETE.json; partial snapshots must be explicit")
    checkpoint_dir = args.run / "checkpoints"
    if not (checkpoint_dir / "policy_last.ckpt").exists():
        raise FileNotFoundError("No trained ACT checkpoint exists yet")
    temporary = args.output.with_name(args.output.name + ".partial")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(exist_ok=False, parents=True)
    package = temporary / "ACTPullDrawer"
    package.mkdir()
    shutil.copytree(args.source / "runtime", package / "runtime", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for filename in ("deploy_policy.py", "official_source.json", "compatibility.patch"):
        shutil.copy2(args.source / filename, package / filename)
    shutil.copy2(args.run / "train_config.yaml", package / "train_config.yaml")
    (package / "checkpoints").mkdir()
    for filename in ("policy_best.ckpt", "policy_last.ckpt", "dataset_stats.pkl", "normalizer.npz", "normalizer.json"):
        src = checkpoint_dir / filename
        if src.exists():
            shutil.copy2(src, package / "checkpoints" / filename)
    weights = Path("/data/wangzihao/outputs/openvitac_act/torch_cache/hub/checkpoints/resnet18-f37072fd.pth")
    target = package / "torch_cache/hub/checkpoints"
    target.mkdir(parents=True)
    shutil.copy2(weights, target / weights.name)
    for filename in ("contract.json", "split.json", "runtime.json", "progress.json", "COMPLETE.json"):
        if (args.run / filename).exists():
            shutil.copy2(args.run / filename, package / filename)
    config = {"policy_name": "ACTPullDrawer", "seed": 0, "ckpt_setting": 100,
              "instruction_type": "seen", "instuction_file": None, "rollout_from_pre_move": True,
              "checkpoint_name": "policy_best.ckpt" if (checkpoint_dir / "policy_best.ckpt").exists() else "policy_last.ckpt",
              "device": "cuda:0", "swap_live_red_blue": False}
    (package / "deploy.yml").write_text(yaml.safe_dump(config, sort_keys=False))
    (package / "requirements_inference.txt").write_text(
        "torch==2.5.1\ntorchvision==0.20.1\nnumpy==1.26.4\nh5py\neinops\nPyYAML\nipython\npackaging\n")
    (package / "README.md").write_text(
        "# Pull Drawer official ACT baseline\n\n"
        "Copy ACTPullDrawer as a NEW directory under OpenViTac/policy/. Do not overwrite policy/ACT.\n"
        "Run from OpenViTac root: bash eval_policy.sh pull_drawer gelsight ACTPullDrawer/deploy 0\n"
        "Use the pinned official environment/task contract and reset policy every episode.\n"
        "Both head/wrist and both marker tactile inputs are required.\n"
        "qpos/action: seven joints in radians plus LEFT finger metres, not open fraction.\n"
        "Source observation period is nominally 1/60s; official controller defaults must be\n"
        "checked on this simulator version before interpreting closed-loop scores.\n"
        "Training HDF5 inverts the release RGB correction to match official ACT cv2-imdecode buffers.\n"
        "Live observations are passed as in official deployment; swap_live_red_blue is an\n"
        "explicit calibration option, not an assumed extra conversion.\n"
        "Normalizer fit scope: all100 episodes (official); train/validation split:80/20.\n"
        "Tactile pretrained weights are not distributed; this run uses official random fallback.\n"
        "This bundle may be an explicitly requested partial snapshot; check COMPLETE.json.\n")
    files = {p.relative_to(package).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in package.rglob("*") if p.is_file()}
    (package / "SHA256.json").write_text(json.dumps(files, indent=2))
    with tarfile.open(temporary / "ACTPullDrawer.tar.gz", "w:gz") as archive:
        archive.add(package, arcname="ACTPullDrawer")
    temporary.replace(args.output)
    print(json.dumps({"handoff": str(args.output / "ACTPullDrawer.tar.gz"), "files": len(files),
                      "final": (args.run / "COMPLETE.json").is_file()}))


if __name__ == "__main__":
    main()
