Archived 2026-05-27.

This directory keeps the deprecated in-process multi-env experiment:

- train_online_pi0_action_hidden_dreamervla_multienv.py
- run_online_dreamervla_wmpo_alltasks_g67_multienv.sh
- smoke/wait_and_run_multienv_smoke_g45.sh

Reason: it runs multiple LIBERO environments inside one learner process and
does not give the desired collector/learner split. It worked functionally, but
the speedup was small because env stepping and encoder use were still coupled.

Use the multiprocess collector prototype instead:

- scripts/training/train_online_pi0_action_hidden_dreamervla_multiproc.py
- scripts/smoke/run_multiproc_collector_smoke_g45.sh
