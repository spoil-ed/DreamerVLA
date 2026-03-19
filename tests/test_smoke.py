from dreamer_vla.config import DreamerVLAConfig
from dreamer_vla.pipeline import DreamerVLAPipeline
from dreamer_vla.trainer.main_ppo import build_trainer


def test_pipeline_smoke():
    config = DreamerVLAConfig()
    config.data.train_num_sequences = 4
    config.data.val_num_sequences = 2
    config.data.rollout_num_sequences = 2
    config.data.train_batch_size = 2
    config.data.val_batch_size = 2
    config.data.rollout_batch_size = 2
    config.trainer.total_epochs = 1

    pipeline = DreamerVLAPipeline(config)
    assert pipeline.summary()["modules"]["world_model"] == "WorldModelAdapter"

    trainer = build_trainer(config=config)
    history = trainer.fit()
    assert len(history) == 1
    assert "world_model/loss" in history[0]
