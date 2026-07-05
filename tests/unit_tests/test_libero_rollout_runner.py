from dreamervla.runners.libero_rollout_runner import run_vectorized_rollout


class _FakeVecEnv:
    """K slots; each episode ends after `ep_len` steps; success = (task_id + episode_id) even."""
    def __init__(self, num_envs, ep_len=2):
        self.num_envs = num_envs
        self.ep_len = ep_len
        self._task = [-1] * num_envs
        self._ep = [-1] * num_envs
        self._t = [0] * num_envs

    def set_task(self, task_ids, env_ids=None):
        ids = env_ids if env_ids is not None else range(self.num_envs)
        for e, t in zip(ids, task_ids, strict=False):
            self._task[e] = t
        return [f"task-{t}" for t in task_ids]

    def reset(self, task_ids, episode_ids, env_ids=None):
        ids = list(env_ids) if env_ids is not None else list(range(self.num_envs))
        for e, t, ep in zip(ids, task_ids, episode_ids, strict=False):
            self._task[e] = t
            self._ep[e] = ep
            self._t[e] = 0
        return [{"rec": e} for e in ids]

    def step(self, actions, env_ids=None):
        ids = list(env_ids) if env_ids is not None else list(range(self.num_envs))
        out = []
        for e in ids:
            self._t[e] += 1
            term = self._t[e] >= self.ep_len
            success = term and (self._task[e] + self._ep[e]) % 2 == 0
            out.append((0.0, term, False, {"success": success}, {"rec": e}))
        return out


class _StubExtractor:
    def reset(self): pass
    def prepare(self, obs, desc): return obs


def _stub_infer(preps):
    # decode contract: object with .action_chunk (list of actions)
    return [type("O", (), {"action_chunk": [0.0]})() for _ in preps]


def test_core_runs_every_work_item_once():
    vec = _FakeVecEnv(num_envs=3, ep_len=2)
    seen = []
    work = [(t, e) for t in range(4) for e in range(3)]  # 12 items
    run_vectorized_rollout(
        vec, [_StubExtractor() for _ in range(3)], _stub_infer, work,
        episode_horizon=10, on_episode=lambda t, e, steps, ok: seen.append((t, e, ok)),
    )
    assert sorted((t, e) for t, e, _ in seen) == sorted(work)
    assert {(t, e): ok for t, e, ok in seen} == {  # success = (t+e) even
        (t, e): (t + e) % 2 == 0 for t, e in work
    }
