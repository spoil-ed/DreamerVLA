from __future__ import annotations

from pathlib import Path


def test_model_and_algorithm_layout_is_converged() -> None:
    root = Path(__file__).resolve().parents[2]
    models = root / "dreamervla" / "models"
    algorithms = root / "dreamervla" / "algorithms"

    assert (models / "embodiment").is_dir()
    assert (models / "embodiment" / "world_model").is_dir()
    assert not (models / "encoder").exists()
    assert not (models / "world_model").exists()
    assert not (models / "actor").exists()
    assert not (models / "critic").exists()
    assert not (models / "reward").exists()

    assert (algorithms / "actor").is_dir()
    assert (algorithms / "critic").is_dir()


def test_converged_public_imports() -> None:
    from dreamervla.algorithms.actor import OpenVLADiscreteTokenActor
    from dreamervla.algorithms.critic import LatentSuccessClassifier, TwohotCritic
    from dreamervla.models.embodiment import OpenVLAOFTPolicy, RynnVLAEncoder
    from dreamervla.models.embodiment.world_model import ChunkAwareWorldModel, WorldModel

    assert OpenVLAOFTPolicy is not None
    assert RynnVLAEncoder is not None
    assert WorldModel is not None
    assert ChunkAwareWorldModel is not None
    assert OpenVLADiscreteTokenActor is not None
    assert TwohotCritic is not None
    assert LatentSuccessClassifier is not None
