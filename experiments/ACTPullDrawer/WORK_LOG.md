# Pull Drawer ACT preparation, 2026-11-01

- Official OpenViTac5220aaf source49files retained with hashes. Runtime ACT95,577,929parameters; only device-placement patch, no architecture/loss changes.
- All100GelSight Pull Drawer demonstrations converted from pinned HF5fb2757b:24,454source frames ->24,354next-state pairs. qpos/action use joint7+left finger metres. All400videos fully decoded/PTS checked; HF channel correction inverted to match the upstream ACT HDF5 decoder convention. Final HDF5 bytes13,402,496,385.
- Official settings retained:2V+2T,chunk50,batch64,20,000updates,LR1e-5,KL10,enc4/dec7,hidden512/FF3200,AdamW1e-4. Seed0; official seed1 episode split80/20,19,469/4,885windows. Official all100-episode normalizer scope preserved.
- Missing unpublished tactile checkpoint explicitly null: same upstream random fallback, not a pretrained-result reproduction claim. ImageNet vision weights are cached locally.
- Real full-model CPU preflight passed two batch2 optimizer updates, validation, [2,50,8] inference and checkpoint/optimizer reload. Loss100.1496 ->73.2774; these are smoke artifacts, not trained baseline checkpoints. Evidence:/data/wangzihao/outputs/openvitac_act/pull_drawer_cpu_preflight/preflight_report.json.
- Five adaptation tests passed, including source architecture integrity, all100 numeric alignments, tail padding and exact deployment/training transform parity. Actual batch64/6worker loader passed two batches in2.49s; worker affinities separately recorded.
- Isolated project and venv. Existing ManiFeel/UniVTAC/UViP code, normalizers, checkpoints and jobs remain unchanged.
- New GPU job uses one H200,16CPUs/64GiB,gpu,gpux/common48h,per-run lock,nice10000,time-limit stop and requeue. Full resumable state includes model,Adam,RNG,current permutation/cursor and best model. Best/last raw weights and pkl/NPZ/JSON normalizers remain ACT-compatible.
- Final3090 bundle is generated after COMPLETE, in run/handoff/final/ACTPullDrawer.tar.gz, with runtime,weights,normalizers,config,hashes and offline vision initialization. No trained baseline checkpoint is claimed before the queued GPU job runs.

- Submitted formal H200 job412676; current state PENDING/AssocGrpGRES. Source snapshot106files verifies with no mismatches; frozen implementation commit09de90e. The frozen-source adaptation suite also passed5tests. Official normalizers are ready; no formal training checkpoint exists before GPU allocation. Automatic final bundle path:/data/wangzihao/outputs/openvitac_act/pull_drawer_100_seed0/handoff/final/ACTPullDrawer.tar.gz.

## 2026-11-02 launch failure repair

- 412676 failed after1s on server45 at2026-11-02 09:19:31, ExitCode0:53; no batch logs or trained checkpoint were produced. The configured stdout/stderr parent directory did not exist. The original preparation command had failed during Git metadata collection, skipped subsequent directory creation, yet its shell continued to sbatch.
- Added a checked Python submission entrypoint: create/fsync-test log directory first, reject duplicate ACT jobs, query both partition limits, freeze a new source snapshot, check the batch script, then call sbatch with explicit log paths and source export. Scheduler/submission errors now stop the sequence.
- Batch launcher accepts ACT_SOURCE_ROOT; original frozen snapshot, datasets and normalizers remain intact. Training/model source and hyperparameters are unchanged. Regression coverage for the missing-directory failure and the existing adaptation suite:7passed in10.44s. No existing experiment is cancelled or reprioritized.
- Resubmitted as414991 using source_retry_20261102_022154_265085 (109hash-verified files). Current state PENDING/MaxGRESPerAccount, ExitCode0:0. Both resolved stdout/stderr paths have an existing writable parent. Common partition limit remains48h. Training/model/config bytes match the previous snapshot, and no existing experiment was modified.

## 2026-11-02 workspace organization

- Imported standalone main d6967a5 into the existing ggkiller-air/OpenViTac
  main at experiments/ACTPullDrawer. The import tree exactly matched all120
  original tracked files. The complete original repository remains archived
  outside projects, alongside the other retired workspaces.
- Source, data, run roots and frozen training snapshots retain their existing
  contracts. Added repository navigation and stopped tracking11generated
  Python bytecode files; their source remains unchanged and local caches stay
  ignored. No model or training hyperparameter changed.
- New-location adaptation/submission suite:7passed in10.70s, including49official
  file hashes, all100episode numeric alignments and deployment transform parity.
  Batch shell syntax and Git whitespace checks passed.
- This server has all upstream source/configuration files checked out.103large
  simulation resources are deferred after repeated GitHub transfer failures;
  docs/FORK_WORKSPACE.md explains git sparse-checkout disable before simulation.
