"""OpenViTac plugin: official ACT inference with strict artifact checks and matching transforms."""
from pathlib import Path
import os
import sys

import numpy as np
import torch
import yaml

from policy._base_policy import BasePolicy


class Policy(BasePolicy):
    """Use native qpos8, official temporal aggregation and original take_action(qpos)."""

    def __init__(self, args):
        super().__init__(args)
        package = Path(__file__).resolve().parent
        checkpoint_dir = Path(args.get("checkpoint_dir", package / "checkpoints"))
        checkpoint_name = args.get("checkpoint_name", "policy_best.ckpt")
        if not (checkpoint_dir / checkpoint_name).is_file() or not (checkpoint_dir / "normalizer.npz").is_file():
            raise FileNotFoundError("ACT trained checkpoint and matching normalizer are required")
        # Preserve official pretrained-backbone construction without network access.
        os.environ["TORCH_HOME"] = str(package / "torch_cache")
        runtime = package / "runtime/policy/ACT"
        sys.path.insert(0, str(runtime))
        from act_policy import ACT
        from utils import OpenViTacDataset
        self.camera_transform = OpenViTacDataset.cam_image_trans
        self.tactile_transform = OpenViTacDataset.tac_image_trans
        cfg = yaml.safe_load((package / "train_config.yaml").read_text())
        cfg.update(ckpt_dir=str(checkpoint_dir), checkpoint_name=checkpoint_name,
                   device=args.get("device", "cuda:0"), num_epochs=1)
        self.model = ACT(cfg)
        # NPZ avoids numpy-version-dependent pickle decoding on the 3090.
        with np.load(checkpoint_dir / "normalizer.npz", allow_pickle=False) as stats:
            self.model.stats = {key: stats[key].copy() for key in stats.files}
        self.swap_live_red_blue = bool(args.get("swap_live_red_blue", False))
        self.camera_names, self.tactile_names = cfg["camera_names"], cfg["tactile_names"]

    def _array(self, value):
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        value = np.asarray(value)
        if value.ndim != 3 or value.shape[-1] not in (3, 4):
            raise ValueError("Expected native HWC image")
        value = value[..., :3]
        if value.dtype != np.uint8:
            raise ValueError("Expected official uint8 observation; do not silently double-normalize")
        if self.swap_live_red_blue:
            value = value[..., ::-1]
        return np.ascontiguousarray(value)

    def encode_obs(self, observation):
        joint = observation["embodiment"]["joint"]
        if isinstance(joint, torch.Tensor):
            joint = joint.detach().cpu().numpy()
        joint = np.asarray(joint, dtype=np.float32).reshape(-1)
        if joint.shape[0] < 9:
            raise ValueError("Expected Franka joint7 plus left/right finger positions")
        result = {"qpos": joint[:8].copy()}
        for name, source in (("cam_high", "head"), ("cam_wrist", "wrist")):
            if name in self.camera_names:
                result[name] = self.camera_transform(self._array(observation["observation"][source]["rgb"]))
        tactile = observation["tactile"]
        for name, keys in (("tac_left", ("left_tactile", "left_gsmini")), ("tac_right", ("right_tactile", "right_gsmini"))):
            if name in self.tactile_names:
                key = next((k for k in keys if k in tactile), None)
                if key is None:
                    raise KeyError("Missing tactile stream: " + str(keys))
                result[name] = self.tactile_transform(self._array(tactile[key]["rgb_marker"]))
        return result

    def eval(self, task, observation):
        action = self.model.get_action(self.encode_obs(observation)).reshape(-1)
        if action.shape != (8,) or not np.isfinite(action).all():
            raise ValueError("Invalid denormalized ACT action")
        # Preserve the official deployment controller. No inferred velocity,
        # extra action repetition or force-setting change is introduced here.
        return task.take_action(torch.from_numpy(action).to(task.device).float(), action_type="qpos")

    def reset(self):
        self.model.reset()
