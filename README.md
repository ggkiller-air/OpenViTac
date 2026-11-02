# OpenViTac Pull Drawer official ACT baseline

This is an isolated single-task baseline. No UViP DP, ManiFeel, or UniVTAC
source, checkpoint, normalizer or running job is modified.

Official source is pinned to FVL-Repo/OpenViTac commit
5220aaf6a68dec5d2a6d284a76cffe337ef234c9. The49 downloaded official files
are SHA256 inventoried in official_source.json and retained unchanged.
runtime/policy/ACT is the executable copy. Its only upstream code patch changes
two hardcoded model.cuda() calls to model.to(args.device); model architecture,
ACTPolicy, CVAE, Transformer, ResNet sharing, padding loss and KL are unchanged.

## Data and action alignment

HF revision:5fb2757b88826917242adefd9196424ef8cf2d98.
All100 GelSight_mini/pull_drawer episodes are converted from LeRobot Parquet+
MP4 to the official ACT HDF5 layout:

    /observations/qpos            [T-1,8] float32
    /action                       [T-1,8] float32
    /observations/images/cam_high [T-1,270,480,3] uint8
    /observations/images/cam_wrist[T-1,270,480,3] uint8
    /observations/images/tac_left [T-1,240,320,3] uint8
    /observations/images/tac_right[T-1,240,320,3] uint8

The public simulation release has no recorded action commands. The official
HDF5Handler.batch_gather_hdf5 pairs joint[:-1] with joint[1:] and images[:-1].
This converter follows that same rule with downsample1. Both qpos and action
use seven arm joints in radians plus LEFT finger position in metres. The
LeRobot observation.state's final open-fraction value is not used for ACT.
24,454 source rows yield24,354 valid pairs; the final source observation is
dropped per episode. All400 videos are fully decoded, with frame count and
PTS checked against the numeric60Hz clock. Pull Drawer has no source step gaps.

Color transport is explicit: HF export swaps cv2-decoded source BGR to RGB.
Official ACT preprocessing keeps cv2.imdecode's payload without swapping.
MP4 decoding to bgr24 restores that upstream ACT convention. No resize or
additional JPEG loss is introduced in conversion; HDF5 uses frame-wise LZF.
The shared official dataset transforms resize to256x256, apply ImageNet
normalization to visual images, and keep tactile images in[0,1]. The3090
adapter reuses those exact transforms, avoiding the stock deployment's uint8
resize-before-division difference. Live input is kept in the same memory order
as stock deployment; swap_live_red_blue is an explicit calibration option.

Dataset: /data/wangzihao/datasets/openvitac/act_pull_drawer_100_official_v1
Total HDF5 storage:13,402,496,385bytes. Per-episode hashes and source/action
provenance are in conversion_manifest.json. No other benchmark dataset is changed.

## Official settings and disclosed release limitation

train_config.yaml is based on official policy/ACT/train_config.yml:
batch64,20,000updates, chunk50, state/action8, KL10, hidden512, FF3200,
4encoder/7decoder layers,8heads, dropout.1, ResNet18, temporal aggregation,
AdamW LR1e-5/weight-decay1e-4, vision/tactile LR1e-5, save interval5,000.
There is no added scheduler, EMA, architecture change, or altered padding loss.
Two visual streams share the official visual backbone; both tactile streams
share the official tactile backbone. Model size is95,577,929parameters.

The release references encoder/checkpoints/resnet18/20251128-125750/best.pth,
but does not publish that tactile checkpoint. The official code silently skips
it if absent. This run explicitly sets tactile_ckpt:null and records random
tactile initialization. This is an official-code baseline with the released
fallback initialization, not a claim to reproduce unpublished pretrained results.
The vision ImageNet ResNet18 weights are cached and included in the final bundle.

The official seed1 split uses80train/20validation episodes. Official statistics
are fitted over all100episodes using population mean/std with std>=.01; this
all-data normalization scope is intentionally disclosed and preserved. Seed0
is used for model training. Single-task ACT has no language-model input.

## Runtime and artifacts

Environment:/public/home/wangzihao/.venvs/openvitac-act (isolated venv, read-only
base act-py310). Python3.10, PyTorch2.5.1+cu124, torchvision0.20.1, numpy1.26.4.
This modern runtime differs from the upstream old conda specification; the
model code is unchanged apart from device placement.

train_baseline.py uses the official model, optimizer parameter groups, dataset,
normalization and losses. Infrastructure adds JSONL logs, finite checks,
deterministic replayable batch permutations, atomic weights/full-state saves,
RNG/sampler continuation, time-limit safe stops, and source/data contracts.
It runs exactly20,000updates, correcting the upstream loop's extra-step bug.
Validation retains official averaging of per-batch losses. Best/last raw
state_dict files are compatible with official ACT loading.

Run root:/data/wangzihao/outputs/openvitac_act/pull_drawer_100_seed0

- checkpoints/policy_best.ckpt and policy_last.ckpt: inference weights.
- checkpoints/policy_step_005000.ckpt etc.: fixed update snapshots.
- checkpoints/training_latest.pt: optimizer/RNG/permutation/cursor/best-model resume.
- checkpoints/dataset_stats.pkl,normalizer.npz,normalizer.json: same statistics.
- metrics.jsonl,validation.jsonl,progress.json,split.json,contract.json,config:
  training records and exact provenance.
- handoff/final/ACTPullDrawer.tar.gz: automatically created only after COMPLETE.

The Slurm job requests one H200 on gpu,gpux,16CPUs/64GiB/common48h, with a
per-run lock and safe requeue on time-limit warning. Existing jobs are untouched.
The job prints GPU/process status before model work and records actual worker
CPU affinities. First optimizer update is checkpointed; subsequent complete
state saves are every1,000steps and after validation. GPU training is not
replaced with CPU training when quota is unavailable.

Submit through the checked entrypoint:

    /public/home/wangzihao/.venvs/openvitac-act/bin/python submit_training.py

It creates and write-tests the Slurm stdout/stderr parent directory before any
submission, rejects duplicate ACT jobs, queries the current common partition
walltime, freezes a fresh source snapshot, and records the submission. Slurm
opens its log files before the batch script runs; creating their directory
inside the batch script is too late.

The full official architecture passed real-data CPU batch2 checks: two optimizer
updates, validation, [2,50,8] inference, and exact checkpoint/optimizer reload.
These are preflight artifacts, not trained Baseline weights for evaluation.

## 3090 handoff

After training completes, extract ACTPullDrawer.tar.gz and place ACTPullDrawer
as a NEW folder under OpenViTac/policy/. Do not overwrite policy/ACT or any
existing evaluation installation. The bundle includes the exact runtime code,
best/last checkpoints, portable normalizer, config, hashes and offline ImageNet
initialization cache. Required PyTorch/torchvision versions are provided.

    bash eval_policy.sh pull_drawer gelsight ACTPullDrawer/deploy 0

The plugin supports both left_tactile/right_tactile and older left_gsmini/
right_gsmini observation keys, validates artifact existence strictly, and resets
the official temporal aggregator per episode. It retains stock take_action(qpos)
behavior. Collection/control clock and contact behavior must be checked on the
independent simulator before interpreting success rates: source nominal60Hz,
legacy simulator take_action commonly advances one120Hz physics step.

Checkpoint existence and offline loss do not imply a closed-loop success rate.
Actual3090 rollout is a separate evaluation. Training logs and COMPLETE status
must be checked before calling the Baseline trained.
