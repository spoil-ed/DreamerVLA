def test_inference_worker_can_disable_obs_embedding_sidecar():
    from dreamervla.workers.inference.inference_worker import InferenceWorker

    worker = InferenceWorker(
        {
            "encoder": {"target": "dreamervla.diagnostics.fixtures.inference_models:TinyEncoder"},
            "world_model": {
                "target": "dreamervla.diagnostics.fixtures.actor_models:TinyLumosWorldModel",
                "kwargs": {"hidden_dim": 4, "action_dim": 7},
            },
            "policy": {
                "target": "dreamervla.diagnostics.fixtures.actor_models:TinyLumosPolicy",
                "kwargs": {"hidden_dim": 4, "action_dim": 7, "chunk_size": 1},
            },
            "device": "cpu",
            "emit_hidden_sidecar": False,
        },
        {},
        num_envs=1,
    )
    worker.init()

    out = worker.forward_batch([{"step": 0, "env_id": 0, "is_first": True}], [0])

    assert "actions" in out
    assert "obs_embedding" not in out
