#!/usr/bin/env python3
"""Convert verified Pull Drawer LeRobot episodes to the official ACT HDF5 contract."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import time

import av
import h5py
import numpy as np
import pyarrow.parquet as pq

REVISION = "5fb2757b88826917242adefd9196424ef8cf2d98"
STREAMS = {"cam_high": "head", "cam_wrist": "wrist", "tac_left": "tactile_left", "tac_right": "tactile_right"}
CODEC = {"state": "current measured arm q0..q6 radians + LEFT finger metres",
         "action": "next measured arm q0..q6 radians + LEFT finger metres",
         "recorded_commands_available": False, "target_offset": 1,
         "terminal_policy": "drop final source observation; no duplicated terminal target",
         "fps": 60, "source_step_delta": 2, "official_downsample": 1}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def convert_episode(arguments: tuple) -> dict:
    raw, output, episode, info, manifest = arguments
    raw, output = Path(raw), Path(output)
    source = raw / "Simulation/GelSight_mini/pull_drawer"
    stem = "episode_%06d" % episode
    data = pq.read_table(source / "data/chunk-000" / (stem + ".parquet")).to_pydict()
    if any("action" in key for key in data):
        raise ValueError("Unexpected recorded action; review codec instead of overwriting it")
    joint = np.asarray(data["observation.joint_position"], dtype=np.float32)
    fingers = np.asarray(data["observation.finger_position"], dtype=np.float32)
    qpos = np.concatenate((joint, fingers[:, :1]), axis=1)
    length = len(qpos)
    if manifest["episode_index"] != episode or manifest["source_step_gaps"] or manifest["frame_count"] != length:
        raise ValueError("Pull Drawer must have a continuous two-step source clock")
    np.testing.assert_array_equal(data["frame_index"], np.arange(length))
    np.testing.assert_allclose(data["timestamp"], np.arange(length) / 60, atol=2e-6)
    np.testing.assert_array_equal(np.asarray(data["observation.state"], dtype=np.float32)[:, :7], joint)
    if qpos.shape != (length, 8) or not np.isfinite(qpos).all():
        raise ValueError("Invalid 8D ACT qpos")
    path = output / ("episode_%d.hdf5" % episode)
    marker = path.with_suffix(".json")
    if marker.exists():
        record = json.loads(marker.read_text())
        if path.is_file() and sha256(path) == record["sha256"]:
            return record
        raise ValueError("Existing converted episode failed integrity check")
    temporary = path.with_suffix(".hdf5.partial")
    with h5py.File(temporary, "w") as target:
        target.attrs["sim"] = True
        target.attrs["source_revision"] = REVISION
        target.attrs["task"] = "pull_drawer"
        target.attrs["source_episode_index"] = episode
        target.attrs["nominal_fps"] = 60
        target.attrs["action_codec"] = json.dumps(CODEC)
        target.attrs["instruction"] = manifest["instruction"]
        target.create_dataset("observations/qpos", data=qpos[:-1])
        target.create_dataset("action", data=qpos[1:])
        target.create_dataset("timestamp", data=np.asarray(data["timestamp"], dtype=np.float32)[:-1])
        target.create_dataset("source_frame_index", data=np.arange(length - 1, dtype=np.int64))
        target.create_dataset("target_frame_index", data=np.arange(1, length, dtype=np.int64))
        for name, key in STREAMS.items():
            feature = "observation.images." + key
            height, width, channels = info["features"][feature]["shape"]
            images = target.create_dataset("observations/images/" + name,
                shape=(length - 1, height, width, channels), dtype=np.uint8,
                chunks=(1, height, width, channels), compression="lzf")
            video = source / "videos/chunk-000" / feature / (stem + ".mp4")
            count = 0
            with av.open(str(video)) as container:
                stream = container.streams.video[0]
                stream.thread_count = 1
                if stream.frames != length or float(stream.average_rate) != 60:
                    raise ValueError("Video/Parquet length or clock mismatch")
                for i, frame in enumerate(container.decode(stream)):
                    if i >= length or abs(float(frame.pts * frame.time_base) - i / 60) > 1e-5:
                        raise ValueError("Video PTS mismatch")
                    if i < length - 1:
                        # HF corrected cv2-decoded BGR to RGB during MP4 export.
                        # Official ACT's HDF5Handler keeps cv2.imdecode's array
                        # without a BGR->RGB swap; invert the release correction
                        # to reproduce that exact upstream tensor convention.
                        images[i] = frame.to_ndarray(format="bgr24")
                    count += 1
            if count != length:
                raise ValueError("Incomplete video decode")
        np.testing.assert_array_equal(target["observations/qpos"][:], qpos[:-1])
        np.testing.assert_array_equal(target["action"][:], qpos[1:])
    temporary.replace(path)
    record = {"episode": episode, "source_frames": length, "pairs": length - 1,
              "bytes": path.stat().st_size, "sha256": sha256(path), "video_streams_decoded": 4}
    marker.write_text(json.dumps(record, indent=2))
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    completion = json.loads((args.raw_root / "COMPLETE.json").read_text())
    if completion["revision"] != REVISION or completion.get("failures"):
        raise ValueError("Source release is incomplete or unpinned")
    source = args.raw_root / "Simulation/GelSight_mini/pull_drawer"
    info = json.loads((source / "meta/info.json").read_text())
    rows = [json.loads(line) for line in (source / "meta/conversion_manifest.jsonl").read_text().splitlines()]
    if info["total_episodes"] != 100 or len(rows) != 100:
        raise ValueError("Expected all 100 Pull Drawer episodes")
    args.output.mkdir(exist_ok=True, parents=True)
    if (args.output / "COMPLETE.json").exists():
        existing = json.loads((args.output / "conversion_manifest.json").read_text())
        if existing.get("release_correction_inverted_for_ACT") is not True or existing.get("action_codec") != CODEC:
            raise ValueError("Existing conversion has different action/color semantics; use a new output directory")
    start = time.monotonic()
    arguments = [(str(args.raw_root), str(args.output), i, info, rows[i]) for i in range(100)]
    records = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for record in pool.map(convert_episode, arguments):
            records.append(record)
            print(json.dumps({**record, "seconds": round(time.monotonic() - start, 1)}), flush=True)
    manifest = {"schema": "openvitac.official_act.pull_drawer.v1", "source_revision": REVISION,
                "episodes": 100, "source_frames": sum(r["source_frames"] for r in records),
                "pairs": sum(r["pairs"] for r in records), "bytes": sum(r["bytes"] for r in records),
                "action_codec": CODEC, "image_color_order": "official_ACT_cv2_imdecode_BGR_payload",
                "release_video_color_order": "RGB", "release_correction_inverted_for_ACT": True,
                "resize_in_converter": False,
                "source_streams": STREAMS, "episodes_data": records}
    content = json.dumps(manifest, indent=2)
    (args.output / "conversion_manifest.json").write_text(content)
    (args.output / "COMPLETE.json").write_text(json.dumps({"status": "complete", "manifest_sha256": hashlib.sha256(content.encode()).hexdigest(),
                                                         "episodes": 100, "pairs": manifest["pairs"]}, indent=2))
    print(json.dumps({k: v for k, v in manifest.items() if k != "episodes_data"}), flush=True)


if __name__ == "__main__":
    main()
