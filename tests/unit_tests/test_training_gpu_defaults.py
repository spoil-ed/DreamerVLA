"""Training and distributed smoke default to eight GPUs; other routes opt out."""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir


def _compose(experiment: str, profile: str = "production"):
    with initialize_config_dir(
        config_dir=str(Path(__file__).resolve().parents[2] / "configs"), version_base=None
    ):
        return compose(
            config_name="train", overrides=[f"experiment={experiment}", f"profile={profile}"]
        )


@pytest.mark.parametrize("profile", ["production", "debug", "smoke"])
@pytest.mark.parametrize(
    "experiment",
    [
        "wm_pi05_collected_train",
        "wm_pi05_vjepa2_latent_train",
        "wm_full_dataset_train",
        "pi05_libero_sft",
        "pi05_pixel_decoder",
        "pi05_pixel_decoder_collected",
        "classifier_official_upper_bound",
        "openvla_libero",
        "openvla_onetraj_libero_cotrain",
    ],
)
def test_training_defaults_to_eight_gpus(experiment: str, profile: str) -> None:
    cfg = _compose(experiment, profile)
    assert cfg.launch.ngpu == 8
    if experiment.startswith("openvla"):
        assert cfg.launch.distributed is False  # Ray owns its eight ranks.
        assert cfg.cluster.num_gpus == cfg.manual_cotrain.ngpu == 8
    else:
        assert cfg.launch.distributed is True


def test_smoke_preserves_complete_eight_gpu_geometry() -> None:
    from dreamervla.config import validate_cfg

    cfg = _compose("openvla_libero", "smoke")
    validate_cfg(cfg)
    assert cfg.manual_cotrain.real_env_workers == 1
    assert (
        cfg.manual_cotrain.wm_rollout_target_trajectories
        == 7 * cfg.manual_cotrain.wm_envs_per_worker
    )
    assert cfg.manual_cotrain.global_steps == 1


@pytest.mark.parametrize("experiment", ["pi05_libero_sft", "pi05_pixel_decoder"])
def test_generic_launcher_defaults_to_eight_training_processes(experiment: str) -> None:
    from dreamervla.launchers.train import build_launch

    launch = build_launch(["--config", experiment])
    assert launch.ngpu == 8
    assert "--nproc-per-node=8" in launch.command


def test_collection_and_evaluation_keep_their_explicit_resource_contracts() -> None:
    collection = _compose("collect_rollouts")
    evaluation = _compose("eval_libero_vla")
    assert collection.launch.distributed is evaluation.launch.distributed is False
    assert collection.launch.ngpu == evaluation.launch.ngpu == 1
