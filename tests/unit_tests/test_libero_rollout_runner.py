from dreamervla.runners.eval_metrics import summarize_libero_task_success
from dreamervla.runners.libero_rollout_runner import (
    SuccessTally,
    build_grid_work_list,
    run_vectorized_rollout,
)


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


def test_grid_matches_sequential_episode_order():
    # sequential loop is: for task in task_ids: for episode_idx in range(n): (task, episode_idx)
    assert build_grid_work_list([5, 2, 9], num_episodes_per_task=3) == [
        (5, 0), (5, 1), (5, 2), (2, 0), (2, 1), (2, 2), (9, 0), (9, 1), (9, 2)
    ]


def test_success_tally_macro_average_matches_eval_metrics():
    tally = SuccessTally()
    # task 5: 2/3 ; task 2: 0/3
    for ep, ok in enumerate([True, True, False]):
        tally.on_episode(5, ep, [], ok)
    for ep in range(3):
        tally.on_episode(2, ep, [], False)
    metrics = tally.summarize(episodes_per_task=3)
    expected = summarize_libero_task_success(
        [{"task_id": 5, "episodes": 3, "successes": 2},
         {"task_id": 2, "episodes": 3, "successes": 0}],
        episodes_per_task=3,
    )
    assert metrics["eval_success_rate"] == expected["eval_success_rate"]
    assert metrics["eval_task_5_success_rate"] == expected["eval_task_5_success_rate"]
