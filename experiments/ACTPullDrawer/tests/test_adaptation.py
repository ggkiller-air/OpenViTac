"""Verify the actual ACT transport contract and unchanged official architecture."""
from pathlib import Path
import hashlib
import json
import sys

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from train_baseline import ReplayBatchSampler


def test_official_source_intact_and_only_device_compatibility_changes():
    source = json.loads((ROOT / "official_source.json").read_text())
    for name, digest in source["sha256"].items():
        assert hashlib.sha256((ROOT / "official" / name).read_bytes()).hexdigest() == digest
    changed = []
    for path in (ROOT / "official/policy/ACT").rglob("*.py"):
        relative = path.relative_to(ROOT / "official/policy/ACT")
        if path.read_bytes() != (ROOT / "runtime/policy/ACT" / relative).read_bytes():
            changed.append(relative.as_posix())
    assert changed == ["detr/main.py"]
    assert (ROOT / "compatibility.patch").read_text().count('+    model.to(') == 2


def test_resume_batch_order_and_tail():
    sampler = ReplayBatchSampler(17, 8)
    sampler.order = list(reversed(range(17)))
    full = list(sampler)
    sampler.start_batch = 1
    assert list(sampler) == full[1:]
    assert full[-1] == [0]
    assert sorted(sum(full, [])) == list(range(17))


def test_converted_data_next_state_alignment_and_finger_units():
    root = Path('/data/wangzihao/datasets/openvitac/act_pull_drawer_100_official_v1')
    manifest = json.loads((root / 'conversion_manifest.json').read_text())
    assert manifest['episodes'] == 100 and manifest['pairs'] == 24354
    assert manifest['release_correction_inverted_for_ACT']
    source = Path('/data/wangzihao/datasets/openvitac/converted/uvip_v1_5fb2757b/pull_drawer')
    fingers = np.load(source / 'finger_position.npy', mmap_mode='r')
    state = np.load(source / 'state.npy', mmap_mode='r')
    ends = np.load(source / 'episode_ends.npy', mmap_mode='r')
    start = 0
    for i in range(100):
        stop = int(ends[i])
        with h5py.File(root / f'episode_{i}.hdf5', 'r') as f:
            np.testing.assert_array_equal(f['observations/qpos'][:, :7], state[start:stop - 1, :7])
            np.testing.assert_array_equal(f['observations/qpos'][:, 7], fingers[start:stop - 1, 0])
            np.testing.assert_array_equal(f['action'][:, :7], state[start + 1:stop, :7])
            np.testing.assert_array_equal(f['action'][:, 7], fingers[start + 1:stop, 0])
            assert set(f['observations/images']) == {'cam_high','cam_wrist','tac_left','tac_right'}
            assert f['observations/images/cam_high'].shape[0] == stop - start - 1
        start = stop


def test_official_dataset_chunk_padding_and_normalization():
    sys.path.insert(0, str(ROOT / 'runtime/policy/ACT'))
    from utils import OpenViTacDataset, get_norm_stats, worker_clear_fn
    root = Path('/data/wangzihao/datasets/openvitac/act_pull_drawer_100_official_v1')
    stats, maximum = get_norm_stats(str(root), 100)
    assert np.all(stats['qpos_std'] >= .01) and np.all(stats['action_std'] >= .01)
    dataset = OpenViTacDataset([0], str(root), ['cam_high','cam_wrist'], ['tac_left','tac_right'], stats, 50)
    camera,tactile,qpos,actions,pad = dataset[len(dataset)-1]
    assert camera.shape == tactile.shape == (2,3,256,256)
    assert qpos.shape == (8,) and actions.shape == (50,8)
    assert pad.tolist() == [False]+[True]*49
    with h5py.File(root/'episode_0.hdf5','r') as f:
        np.testing.assert_allclose(actions[0].numpy()*stats['action_std']+stats['action_mean'],f['action'][-1],atol=1e-6)
    worker_clear_fn()


def test_3090_adapter_matches_official_training_transforms(monkeypatch):
    import importlib.util
    import types
    fake_policy = types.ModuleType('policy')
    fake_base = types.ModuleType('policy._base_policy')
    fake_base.BasePolicy = type('BasePolicy', (), {})
    monkeypatch.setitem(sys.modules, 'policy', fake_policy)
    monkeypatch.setitem(sys.modules, 'policy._base_policy', fake_base)
    spec = importlib.util.spec_from_file_location('drawer_deploy_test', ROOT/'deploy_policy.py')
    module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    from utils import OpenViTacDataset
    p = module.Policy.__new__(module.Policy)
    p.camera_names,p.tactile_names = ['cam_high','cam_wrist'],['tac_left','tac_right']
    p.camera_transform,p.tactile_transform = OpenViTacDataset.cam_image_trans,OpenViTacDataset.tac_image_trans
    p.swap_live_red_blue = False
    with h5py.File('/data/wangzihao/datasets/openvitac/act_pull_drawer_100_official_v1/episode_0.hdf5','r') as f:
        qpos = f['observations/qpos'][0]
        images = {key:f['observations/images/'+key][0] for key in ['cam_high','cam_wrist','tac_left','tac_right']}
    observation = {'embodiment':{'joint':torch.from_numpy(np.r_[qpos,qpos[-1]])},
                   'observation':{'head':{'rgb':torch.from_numpy(images['cam_high'])},'wrist':{'rgb':torch.from_numpy(images['cam_wrist'])}},
                   'tactile':{'left_tactile':{'rgb_marker':torch.from_numpy(images['tac_left'])},'right_tactile':{'rgb_marker':torch.from_numpy(images['tac_right'])}}}
    encoded = p.encode_obs(observation)
    np.testing.assert_array_equal(encoded['qpos'],qpos)
    for key in images:
        transform = p.camera_transform if key.startswith('cam_') else p.tactile_transform
        torch.testing.assert_close(encoded[key],transform(images[key]),rtol=0,atol=0)
