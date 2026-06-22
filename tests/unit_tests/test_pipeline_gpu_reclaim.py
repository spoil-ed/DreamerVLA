"""Pipeline GPU reclamation: park the idle frozen encoder off-GPU during warmup.

The frozen OFT encoder (~15GB) is unused during WM/classifier warmup (warmup reads
pre-seeded sidecar embeddings), so the pipeline runner shuttles it to CPU for the
warmup and back to the device before the online phase that needs it.
"""

from dreamervla.runners.online_cotrain_pipeline_runner import OnlineCotrainPipelineRunner


class _SpyEncoder:
    def __init__(self) -> None:
        self.moved_to = None

    def to(self, device):
        self.moved_to = device
        return self


def _bare_runner(encoder):
    runner = OnlineCotrainPipelineRunner.__new__(OnlineCotrainPipelineRunner)
    runner.encoder = encoder
    return runner


def test_set_encoder_device_moves_the_encoder():
    spy = _SpyEncoder()
    runner = _bare_runner(spy)

    runner._set_encoder_device("cpu")

    assert spy.moved_to == "cpu"


def test_set_encoder_device_tolerates_missing_encoder():
    # No-op when there is no encoder (e.g. total_env_steps<=0 builds encoder=None).
    _bare_runner(None)._set_encoder_device("cpu")  # must not raise
