# Online KIR Findings

- RLinf Wan KIR uses a downloaded static `.npy` initialization pool. Runtime code
  only loads the first reference frame and the final four trajectory frames.
- In a 25-frame `_kir.npy`, the temporal context is frames 21–24 and frame 24 is
  the keyframe/current state from which imagination continues.
- RLinf does not detect semantic keyframes online; `_kir` naming and trajectory-end
  placement encode the convention.
- DreamerVLA `OnlineReplay` stores full episodes with `success`, `finish_step`,
  `obs_embedding`, `action`, `lang_emb`, and `proprio` information.
- Current `sample_initial_conditions()` always selects the episode's first step.
- Current chunk-WM bootstrap repeats a single latent to fill history and creates a
  zero action history; true KIR must replace both with aligned real history.
- Mainline cotrain re-encodes each current-step real batch and calls
  `ReplayWorker.replace_real_trajectories()`, so online KIR can operate directly on
  step-local replay. Startup collection data remains the cold-start seed.
