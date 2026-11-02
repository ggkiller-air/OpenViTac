#!/usr/bin/env python3
"""Resumable runtime around the unmodified official ACT model and loss."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import pickle
import random
import signal
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler
import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "runtime/policy/ACT"))
from act_policy import ACTPolicy
from utils import OpenViTacDataset, get_norm_stats, worker_init_fn

STOP = None


def request_stop(signum: int, frame) -> None:
    global STOP
    STOP = signal.Signals(signum).name


def stop_requested() -> bool:
    """Observe this run's Slurm stop file without signalling any other job."""
    global STOP
    filename = os.environ.get("ACT_STOP_FILE")
    if filename and Path(filename).is_file():
        STOP = Path(filename).read_text().strip() or "SIGUSR1"
    return STOP is not None


def fingerprint(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def atomic_torch(path: Path, value) -> None:
    temporary = path.with_name(path.name + ".partial")
    torch.save(value, temporary)
    temporary.replace(path)


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def rng_state() -> dict:
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        torch.cuda.set_rng_state_all([v.cpu() for v in state["cuda"]])


def move(data: tuple, device: torch.device) -> tuple:
    return tuple(value.to(device, non_blocking=True) for value in data)


def forward(policy: ACTPolicy, data: tuple) -> dict:
    camera, tactile, qpos, action, pad = data
    return policy(qpos, camera, tactile, action, pad)


def worker_init(worker_id: int) -> None:
    worker_init_fn(worker_id)
    affinity = sorted(os.sched_getaffinity(0))
    torch.set_num_threads(1)
    os.sched_setaffinity(0, {affinity[worker_id % len(affinity)]})
    directory = os.environ.get("ACT_WORKER_AFFINITY_DIR")
    if directory:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        atomic_json(path / (str(os.getpid()) + ".json"), {"pid": os.getpid(), "worker": worker_id,
                    "inherited_affinity": affinity, "actual_affinity": sorted(os.sched_getaffinity(0))})


class ReplayBatchSampler(Sampler):
    """Persist the current epoch permutation and next batch for safe continuation."""

    def __init__(self, size: int, batch_size: int):
        self.order, self.start_batch, self.batch_size = list(range(size)), 0, batch_size

    def __iter__(self):
        for start in range(self.start_batch * self.batch_size, len(self.order), self.batch_size):
            yield self.order[start:start + self.batch_size]

    def __len__(self):
        return max(0, (len(self.order) + self.batch_size - 1) // self.batch_size - self.start_batch)


def make_data(args, cfg: dict) -> tuple:
    """Keep official seed1 80/20 split and official all-100 mean/std normalization."""
    content = (args.dataset / "conversion_manifest.json").read_text()
    completion = json.loads((args.dataset / "COMPLETE.json").read_text())
    if completion["status"] != "complete" or completion["manifest_sha256"] != hashlib.sha256(content.encode()).hexdigest():
        raise ValueError("Incomplete converted dataset")
    manifest = json.loads(content)
    if manifest["episodes"] != 100 or manifest["pairs"] != 24354:
        raise ValueError("Expected all 100 episodes and all valid next-state pairs")
    shuffled = np.random.RandomState(1).permutation(100)
    split = {"split_seed": 1, "train_episodes": shuffled[:80].tolist(), "validation_episodes": shuffled[80:].tolist(),
             "normalizer_fit_scope": "all 100 episodes, matching official get_norm_stats"}
    stats, maximum_length = get_norm_stats(str(args.dataset), 100)
    datasets = [OpenViTacDataset(split[key], str(args.dataset), cfg["camera_names"], cfg["tactile_names"], stats, cfg["chunk_size"])
                for key in ("train_episodes", "validation_episodes")]
    return datasets, stats, split, manifest


def write_stats(output: Path, stats: dict) -> None:
    output.mkdir(parents=True, exist_ok=True)
    with (output / "dataset_stats.pkl").open("wb") as stream:
        pickle.dump(stats, stream, protocol=4)
    np.savez(output / "normalizer.npz", **stats)
    atomic_json(output / "normalizer.json", {key: np.asarray(value).tolist() for key, value in stats.items()})


def evaluate(policy, loader, device, maximum_batches=None) -> dict:
    """Preserve official averaging of per-batch L1/KL/loss, including padding."""
    policy.eval()
    totals, count = {}, 0
    with torch.no_grad():
        for data in loader:
            losses = forward(policy, move(data, device))
            for key, value in losses.items():
                if not torch.isfinite(value):
                    raise FloatingPointError("Non-finite validation loss")
                totals[key] = totals.get(key, 0.) + float(value)
            count += 1
            if maximum_batches is not None and count >= maximum_batches:
                break
    if not count:
        raise ValueError("Empty validation dataset")
    return {key: value / count for key, value in totals.items()}


def preflight(args, cfg, datasets, stats, policy, optimizer) -> None:
    """Real full-size architecture: bounded updates, validation, weights and optimizer reload."""
    batch_size = args.smoke_batch_size
    loader = DataLoader(datasets[0], batch_size=batch_size, shuffle=False, num_workers=0)
    validation = DataLoader(datasets[1], batch_size=batch_size, shuffle=False, num_workers=0)
    device = torch.device(args.device)
    batches, losses_record = iter(loader), []
    start = time.monotonic()
    for step in range(args.preflight_steps):
        policy.train()
        optimizer.zero_grad(set_to_none=True)
        losses = forward(policy, move(next(batches), device))
        if not all(torch.isfinite(v) for v in losses.values()):
            raise FloatingPointError("Non-finite real-data ACT loss")
        losses["loss"].backward()
        if not all(torch.isfinite(p.grad).all() for p in policy.parameters() if p.grad is not None):
            raise FloatingPointError("Non-finite ACT gradient")
        optimizer.step()
        losses_record.append({key: float(value) for key, value in losses.items()})
    val = evaluate(policy, validation, device, maximum_batches=1)
    sample = move(next(iter(validation)), device)
    with torch.no_grad():
        before = policy(sample[2], sample[0], sample[1]).cpu()
    atomic_torch(args.output / "preflight_state.pt", {"model": policy.state_dict(), "optimizer": optimizer.state_dict()})
    saved = torch.load(args.output / "preflight_state.pt", map_location="cpu", weights_only=False)
    policy.load_state_dict(saved["model"], strict=True)
    optimizer.load_state_dict(saved["optimizer"])
    with torch.no_grad():
        after = policy(sample[2], sample[0], sample[1]).cpu()
    torch.testing.assert_close(before, after, rtol=0, atol=0)
    if before.shape != (batch_size, cfg["chunk_size"], 8):
        raise ValueError("Unexpected ACT action chunk shape")
    report = {"passed": True, "device": args.device, "architecture_unchanged": True,
              "parameters": sum(p.numel() for p in policy.parameters()), "optimizer_updates": args.preflight_steps,
              "batch_size": batch_size, "train_losses": losses_record, "validation": val,
              "inference_shape": list(before.shape), "seconds": time.monotonic() - start}
    atomic_json(args.output / "preflight_report.json", report)
    print(json.dumps(report), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "train_config.yaml")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--preflight-steps", type=int, default=0)
    parser.add_argument("--smoke-batch-size", type=int, default=2)
    args = parser.parse_args()
    if args.device == "cpu" and not args.preflight_steps:
        parser.error("Formal training requires an allocated CUDA GPU; CPU is only for bounded preflight")
    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGUSR1):
        signal.signal(signum, request_stop)
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoints = args.output / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    cfg = yaml.safe_load(args.config.read_text())
    cfg.update(device=args.device, ckpt_dir=str(checkpoints))
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    datasets, stats, split, manifest = make_data(args, cfg)
    source_files = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in sorted((ROOT / "runtime").rglob("*.py"))}
    contract = {"model_config": {k: v for k, v in cfg.items() if k not in ("device", "ckpt_dir")},
                "dataset_sha256": fingerprint(manifest), "split": split,
                "normalizer": {key: np.asarray(stats[key]).tolist() for key in ("qpos_mean", "qpos_std", "action_mean", "action_std")},
                "runtime_model_source": source_files,
                "training_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    contract_path = args.output / "contract.json"
    if contract_path.exists() and fingerprint(json.loads(contract_path.read_text())) != fingerprint(contract):
        raise ValueError("Run directory belongs to a different data/config/source contract")
    atomic_json(contract_path, contract)
    if (checkpoints / "normalizer.npz").exists():
        with np.load(checkpoints / "normalizer.npz", allow_pickle=False) as saved_stats:
            for key, value in stats.items():
                np.testing.assert_array_equal(saved_stats[key], value)
    else:
        write_stats(checkpoints, stats)
    atomic_json(args.output / "split.json", split)
    (args.output / "train_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    device = torch.device(args.device)
    if device.type == "cuda":
        if "H200" not in torch.cuda.get_device_name(device):
            raise RuntimeError("This launcher is reserved for the requested H200 run")
        print(json.dumps({"gpu": torch.cuda.get_device_name(device), "bytes": torch.cuda.get_device_properties(device).total_memory}), flush=True)
    policy = ACTPolicy(cfg).to(device)
    optimizer = policy.configure_optimizers()
    if args.preflight_steps:
        preflight(args, cfg, datasets, stats, policy, optimizer)
        return
    atomic_json(args.output / "runtime.json", {"gpu": torch.cuda.get_device_name(device), "torch": torch.__version__,
                "batch_size": cfg["batch_size"], "workers": args.workers, "cpu_affinity": sorted(os.sched_getaffinity(0)),
                "parameters": sum(p.numel() for p in policy.parameters()), "train_windows": len(datasets[0]),
                "validation_windows": len(datasets[1]), "tactile_initialization": "random; upstream optional checkpoint not published"})
    state = {"global_step": 0, "epoch": 0, "next_batch": 0, "order": None,
             "pending_validation": False, "best_validation_loss": None, "best_step": None}
    val_generator = torch.Generator().manual_seed(cfg["seed"] + 1000000)
    latest = checkpoints / "training_latest.pt"
    best_model = None
    if latest.exists():
        saved = torch.load(latest, map_location="cpu", weights_only=False)
        if saved["contract_sha256"] != fingerprint(contract):
            raise ValueError("Resume rejected: source, data, config, split or normalization changed")
        policy.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        for value in optimizer.state.values():
            for key, item in value.items():
                if isinstance(item, torch.Tensor) and key != "step":
                    value[key] = item.to(device)
        state = saved["state"]
        best_model = saved.get("best_model")
        if best_model is not None:
            atomic_torch(checkpoints / "policy_best.ckpt", best_model)
        val_generator.set_state(saved["val_generator_state"].cpu())
        restore_rng(saved["rng"])
        print(json.dumps({"event": "resumed", "global_step": state["global_step"], "epoch": state["epoch"]}), flush=True)
    sampler = ReplayBatchSampler(len(datasets[0]), cfg["batch_size"])
    options = dict(num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0, worker_init_fn=worker_init)
    if args.workers:
        options.update(prefetch_factor=2, timeout=180)
    training_loader = DataLoader(datasets[0], batch_sampler=sampler,
                                generator=torch.Generator().manual_seed(cfg["seed"] + 2000000), **options)
    validation_loader = DataLoader(datasets[1], batch_size=cfg["batch_size"], shuffle=True, generator=val_generator, **options)

    def save_checkpoint() -> None:
        atomic_torch(checkpoints / "policy_last.ckpt", policy.state_dict())
        atomic_torch(latest, {"contract_sha256": fingerprint(contract), "model": policy.state_dict(),
                             "optimizer": optimizer.state_dict(), "state": state, "rng": rng_state(),
                             "val_generator_state": val_generator.get_state(), "best_model": best_model})
        atomic_json(args.output / "progress.json", {key: value for key, value in state.items() if key != "order"})

    print(json.dumps({"event": "training_start", "global_step": state["global_step"], "target_steps": cfg["num_steps"]}), flush=True)
    window_start, wait_seconds, window_samples, window_steps = time.monotonic(), 0., 0, 0
    while state["global_step"] < cfg["num_steps"] or state["pending_validation"]:
        if state["order"] is None:
            generator = torch.Generator().manual_seed(cfg["seed"] + state["epoch"])
            state["order"] = torch.randperm(len(datasets[0]), generator=generator).tolist()
        sampler.order, sampler.start_batch = state["order"], state["next_batch"]
        if state["pending_validation"]:
            sampler.order, sampler.start_batch = [], 0
        policy.train()
        iterator = iter(training_loader)
        while True:
            wait_start = time.monotonic()
            try:
                data = next(iterator)
            except StopIteration:
                break
            wait_seconds += time.monotonic() - wait_start
            optimizer.zero_grad(set_to_none=True)
            losses = forward(policy, move(data, device))
            if not all(torch.isfinite(v) for v in losses.values()):
                raise FloatingPointError("Non-finite ACT training loss")
            losses["loss"].backward()
            if not torch.stack([torch.isfinite(p.grad).all() for p in policy.parameters() if p.grad is not None]).all():
                raise FloatingPointError("Non-finite ACT gradient")
            optimizer.step()
            state["global_step"] += 1
            state["next_batch"] += 1
            state["pending_validation"] = (state["global_step"] >= cfg["num_steps"] or
                state["next_batch"] >= (len(datasets[0]) + cfg["batch_size"] - 1) // cfg["batch_size"])
            window_steps += 1
            window_samples += len(data[2])
            if state["global_step"] == 1 or state["global_step"] % 20 == 0:
                torch.cuda.synchronize(device)
                seconds = time.monotonic() - window_start
                record = {"event": "train", "step": state["global_step"], "epoch": state["epoch"],
                          **{key: float(value) for key, value in losses.items()}, "samples_per_second": window_samples / seconds,
                          "data_wait_fraction": wait_seconds / seconds, "steps_in_window": window_steps,
                          "peak_cuda_bytes": torch.cuda.max_memory_allocated(device)}
                with (args.output / "metrics.jsonl").open("a") as stream:
                    stream.write(json.dumps(record, allow_nan=False) + "\n")
                print(json.dumps(record, allow_nan=False), flush=True)
                window_start, wait_seconds, window_samples, window_steps = time.monotonic(), 0., 0, 0
            if state["global_step"] == 1 or state["global_step"] % 1000 == 0 or stop_requested():
                save_checkpoint()
            if state["global_step"] % cfg["save_freq"] == 0:
                atomic_torch(checkpoints / ("policy_step_%06d.ckpt" % state["global_step"]), policy.state_dict())
            if stop_requested():
                atomic_json(args.output / "STOPPED.json", {"signal": STOP, "global_step": state["global_step"]})
                return
            if state["global_step"] >= cfg["num_steps"]:
                break
        validation = evaluate(policy, validation_loader, device)
        record = {"event": "validation", "step": state["global_step"], "epoch": state["epoch"], **validation}
        with (args.output / "validation.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
        if state["best_validation_loss"] is None or validation["loss"] < state["best_validation_loss"]:
            state["best_validation_loss"], state["best_step"] = validation["loss"], state["global_step"]
            best_model = {key: value.detach().cpu().clone() for key, value in policy.state_dict().items()}
            atomic_torch(checkpoints / "policy_best.ckpt", best_model)
        state["epoch"] += 1
        state["next_batch"], state["order"] = 0, None
        state["pending_validation"] = False
        save_checkpoint()
        if stop_requested():
            atomic_json(args.output / "STOPPED.json", {"signal": STOP, "global_step": state["global_step"]})
            return
    atomic_json(args.output / "COMPLETE.json", {"global_step": state["global_step"], "best_step": state["best_step"],
                "best_validation_loss": state["best_validation_loss"], "contract_sha256": fingerprint(contract)})


if __name__ == "__main__":
    main()
