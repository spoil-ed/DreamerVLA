import math
import os
import re

_UNSAFE_METRIC_CHARS_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def format_metric_checkpoint_name(
    *,
    epoch: int,
    metric_name: str,
    metric_value: float,
) -> str:
    """Return the canonical flat filename for a metric-selected checkpoint."""

    completed_epoch = int(epoch)
    if completed_epoch < 0:
        raise ValueError("checkpoint epoch must be non-negative")
    value = float(metric_value)
    if not math.isfinite(value):
        raise ValueError("checkpoint metric value must be finite")
    safe_name = _UNSAFE_METRIC_CHARS_RE.sub("_", str(metric_name)).strip("._-")
    if not safe_name:
        raise ValueError("checkpoint metric name must not be empty")
    return f"epoch={completed_epoch:04d}-{safe_name}={value:.6f}.ckpt"


class TopKCheckpointManager:
    def __init__(
        self,
        save_dir,
        monitor_key: str,
        metric_name: str | None = None,
        mode="min",
        k=1,
    ):
        assert mode in ["max", "min"]
        assert k >= 0

        self.save_dir = save_dir
        self.monitor_key = monitor_key
        self.metric_name = str(metric_name or monitor_key)
        self.mode = mode
        self.k = k
        self.path_value_map = dict()

    def get_ckpt_path(self, data: dict[str, float]) -> str | None:
        if self.k == 0:
            return None

        value = float(data[self.monitor_key])
        ckpt_path = os.path.join(
            self.save_dir,
            format_metric_checkpoint_name(
                epoch=int(data["epoch"]),
                metric_name=self.metric_name,
                metric_value=value,
            ),
        )

        if len(self.path_value_map) < self.k:
            # under-capacity
            self.path_value_map[ckpt_path] = value
            return ckpt_path

        # at capacity
        sorted_map = sorted(self.path_value_map.items(), key=lambda x: x[1])
        min_path, min_value = sorted_map[0]
        max_path, max_value = sorted_map[-1]

        delete_path = None
        if self.mode == "max":
            if value > min_value:
                delete_path = min_path
        else:
            if value < max_value:
                delete_path = max_path

        if delete_path is None:
            return None
        else:
            del self.path_value_map[delete_path]
            self.path_value_map[ckpt_path] = value

            if not os.path.exists(self.save_dir):
                os.mkdir(self.save_dir)

            if os.path.exists(delete_path):
                os.remove(delete_path)
            return ckpt_path
