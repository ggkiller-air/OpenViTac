#!/usr/bin/env python3
"""Validate logs and frozen source before submitting one isolated ACT job."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import getpass
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


def ensure_log_directory(directory: Path) -> None:
    """Slurm opens stdout/stderr before executing the batch script."""
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=".submit_write_check_", dir=directory) as stream:
        stream.write(b"ACT submission log directory is writable\n")
        stream.flush()
        os.fsync(stream.fileno())


def time_seconds(value: str) -> int:
    """Parse a finite Slurm days-hours:minutes:seconds partition maximum."""
    if "-" in value:
        days, clock = value.split("-", 1)
    else:
        days, clock = "0", value
    hours, minutes, seconds = map(int, clock.split(":"))
    return int(days) * 86400 + hours * 3600 + minutes * 60 + seconds


def freeze_source(project: Path, destination: Path) -> int:
    """Keep previous snapshots intact and inventory this exact submission."""
    shutil.copytree(project, destination,
                    ignore=shutil.ignore_patterns(".git", "__pycache__", ".pytest_cache", "*.pyc"))
    records = {p.relative_to(destination).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
               for p in destination.rglob("*") if p.is_file() and p.name != "SOURCE_SHA256.json"}
    (destination / "SOURCE_SHA256.json").write_text(json.dumps(records, indent=2))
    return len(records)


def submit(project: Path, run_root: Path, log_dir: Path, run_command=subprocess.run) -> dict:
    """Never call sbatch if directory, source, scheduler or configuration checks fail."""
    # This deliberately precedes Git lookups and source preparation: a metadata
    # failure must never leave a submitted job without writable Slurm log paths.
    ensure_log_directory(log_dir)
    run_root.mkdir(parents=True, exist_ok=True)
    listing = run_command(["squeue", "--noheader", "--user", getpass.getuser(), "--format=%i|%j|%T"],
                          check=True, capture_output=True, text=True).stdout
    for line in listing.splitlines():
        fields = line.strip().split("|")
        if len(fields) == 3 and fields[1] == "openvitac_ACT_pull_drawer100":
            raise RuntimeError("An ACT Pull Drawer job already exists: " + fields[0])
    limits = {}
    for partition in ("gpu", "gpux"):
        text = run_command(["scontrol", "show", "partition", partition],
                           check=True, capture_output=True, text=True).stdout
        limits[partition] = next(v.split("=", 1)[1] for v in text.split() if v.startswith("MaxTime="))
    seconds = min(time_seconds(value) for value in limits.values())
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    walltime = f"{hours}:{minutes:02d}:{seconds:02d}"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    source = run_root / ("source_retry_" + timestamp)
    count = freeze_source(project, source)
    run_command(["bash", "-n", str(source / "train_h200.sbatch")], check=True)
    command = ["sbatch", "--parsable", "--partition=gpu,gpux", "--time=" + walltime,
               "--output=" + str(log_dir / "%x_%j.out"), "--error=" + str(log_dir / "%x_%j.err"),
               "--export=ALL,ACT_SOURCE_ROOT=" + str(source), str(source / "train_h200.sbatch")]
    result = run_command(command, check=True, capture_output=True, text=True)
    job_id = int(result.stdout.strip().split(";")[0])
    report = {"job_id": job_id, "submission_utc": datetime.now(timezone.utc).isoformat(),
              "replaces_failed_job": 412676, "partition_limits": limits, "time": walltime,
              "source": str(source), "source_files": count, "log_dir": str(log_dir),
              "stdout_pattern": str(log_dir / "%x_%j.out"), "stderr_pattern": str(log_dir / "%x_%j.err"),
              "formal_updates": 20000, "batch_size": 64, "sbatch_command": command}
    history = run_root / "submissions"
    history.mkdir(exist_ok=True)
    previous = run_root / "submission_plan.json"
    if previous.exists():
        old = json.loads(previous.read_text())
        shutil.copy2(previous, history / (str(old["job_id"]) + ".json"))
    (history / (str(job_id) + ".json")).write_text(json.dumps(report, indent=2))
    previous.write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=Path("/data/wangzihao/outputs/openvitac_act/pull_drawer_100_seed0"))
    parser.add_argument("--log-dir", type=Path, default=Path("/data/wangzihao/outputs/openvitac_act/slurm"))
    args = parser.parse_args()
    print(json.dumps(submit(Path(__file__).resolve().parent, args.run_root, args.log_dir), indent=2))
