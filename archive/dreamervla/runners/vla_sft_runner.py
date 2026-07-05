from __future__ import annotations

from dreamervla.runners.pretokenize_vla_runner import PretokenizeVLARunner


class VLASFTRunner(PretokenizeVLARunner):
    """Route-specific VLA SFT runner.

    The training loop lives in ``PretokenizeVLARunner`` for compatibility
    with existing checkpoints.  This class gives the route an explicit home so
    configs can distinguish VLA SFT from generic pretokenized data plumbing.
    """

    runner_name = "vla_sft"
    runner_status = "current"
    runner_family = "vla"


__all__ = ["VLASFTRunner"]
