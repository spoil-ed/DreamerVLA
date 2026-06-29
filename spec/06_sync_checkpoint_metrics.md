# Sync Checkpoint Metrics

本文件保留旧路径兼容。当前权重同步、warmup checkpoint bridge、manual checkpoint 和 metrics
namespace 说明见 [`04_complete_loop.md`](04_complete_loop.md)。

checkpoint 必须能恢复 actor、rollout、world model、classifier、replay cursor 和 global step 的一致状态。
