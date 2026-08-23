# π0.5 LIBERO SFT runs

Both launchers use `python` from the active DreamerVLA π0.5 environment; they do
not source or inspect another repository's virtual environment. Runtime paths,
W&B credentials, and CUDA/NCCL settings must be provided by the caller or job
environment. They run eight-process DDP with an effective batch of 32 per GPU
and 256 globally. The full-data recipe
uses micro-batch 1 with gradient accumulation 32; the one-episode-per-task recipe
uses micro-batch 32 with no gradient accumulation. They use a static loopback
rendezvous, so all eight worker processes remain inside one node and do not depend
on cluster host-name resolution.

Run the full 1693-episode dataset with the baseline learning rate `2.5e-5`:

```bash
bash experiments/pi05_sft/train_full.sh
```

Run one complete episode for each of the 40 tasks with learning rate `5e-6`.
This joint four-suite checkpoint uses action horizon 50:

```bash
bash experiments/pi05_sft/train_one_episode_per_task.sh
```

The full-data recipe trains for 30,000 optimizer steps; the joint four-suite
one-episode recipe trains for 80,000 optimizer steps. Hydra overrides can be
appended to either command. Set `WANDB_MODE=offline` before the command when
online W&B logging is not available.

The full dataset is
`/jfs/oss-import/xinglei/datasets/physical-intelligence/libero`. The deterministic
subset is
`/jfs/oss-import/xinglei/datasets/pi05_libero_one_episode_per_task/physical-intelligence/libero`;
it selects the lowest source episode index for each task and records the mapping
in `subset_manifest.json`. It intentionally uses the full dataset's normalization
statistics so the two runs differ only in episode selection and learning rate.
