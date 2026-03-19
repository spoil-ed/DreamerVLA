# Architecture Notes

Planned high-level pipeline:

1. Encode `(image, proprio, text)` into a fixed-length latent `z`.
2. Compress `z` into a lower-dimensional bottleneck state `z'`.
3. Use a world model to imagine rollouts from `z'`.
4. Score rollouts with critic and choose actions with the actor / VLA policy.
