"""Prevent a missing-log-directory launch failure before sbatch is invoked."""
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from submit_training import ensure_log_directory, submit


def test_submission_creates_writable_logs_before_scheduler_and_freezes_source(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    (project / "train_h200.sbatch").write_text("#!/bin/bash\nset -euo pipefail\n")
    logs, run_root = tmp_path / "missing/parent/logs", tmp_path / "run"
    calls = []

    def command(arguments, **kwargs):
        assert logs.is_dir()
        calls.append(arguments)
        if arguments[0] == "squeue":
            return SimpleNamespace(stdout="")
        if arguments[0] == "scontrol":
            limit = "4-04:00:00" if arguments[-1] == "gpu" else "2-00:00:00"
            return SimpleNamespace(stdout="MaxTime=" + limit)
        if arguments[0] == "bash":
            return subprocess.run(arguments, check=True)
        if arguments[0] == "sbatch":
            source = Path(arguments[-1]).parent
            assert (source / "SOURCE_SHA256.json").is_file()
            assert "--time=48:00:00" in arguments
            assert "--output=" + str(logs / "%x_%j.out") in arguments
            assert "--error=" + str(logs / "%x_%j.err") in arguments
            return SimpleNamespace(stdout="999999\n")
        raise AssertionError(arguments)

    report = submit(project, run_root, logs, run_command=command)
    assert report["job_id"] == 999999
    assert calls[-1][0] == "sbatch"
    assert not list(logs.iterdir())  # Write probe is removed.


def test_invalid_log_path_stops_before_any_scheduler_call(tmp_path):
    path = tmp_path / "file"
    path.write_text("not a directory")
    with pytest.raises(FileExistsError):
        submit(tmp_path, tmp_path / "run", path, run_command=lambda *a, **k: pytest.fail("must not submit"))
