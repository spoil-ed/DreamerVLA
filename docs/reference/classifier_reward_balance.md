# WMPO Reward Model Balance Notes

Source: arXiv 2511.09515, `sections/03method.tex`, Reward Model subsection.

WMPO trains a lightweight reward model from real trajectories. For a trajectory
`tau = {I_0:N}`, a clip has length `L` and is written as `c_i = I_{i-L:i}`.

Positive samples:
- The terminal clip `c_N` of a successful trajectory.

Negative samples:
- Earlier clips from successful trajectories, `c_i` for `L <= i <= N-L`.
- Arbitrary clips from failed trajectories.

Class balance:
- WMPO balances the number of positive and negative samples within each
  training batch.
- The paper describes this as sample-count balancing at batch construction
  time, not as class-weighted BCE.

Inference:
- Apply the reward model over a trajectory with a sliding window.
- Classify the trajectory as successful if any clip probability exceeds a
  validation-selected threshold.

Implication for DreamerVLA token adaptation:
- The closest match is a balanced batch sampler: each train batch should contain
  equal positive and negative token clips.
- The token-WMPO experiment now sets `data.sampling_protocol=wmpo`,
  `data.balance_batches=true`, `training.loss_type=bce`, and
  `classifier.output_dim=1`, preserving token inputs while matching the WMPO
  reward-model sampling/loss contract.
