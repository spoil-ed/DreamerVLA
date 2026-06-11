#!/usr/bin/env python
# ruff: noqa: E402
from __future__ import annotations

import argparse
import copy
import json
import pathlib
from typing import Any

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf, open_dict
from PIL import Image
from torch.utils.data import DataLoader

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]

from dreamer_vla.envs import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    quat2axisangle,
)
from dreamer_vla.models.actor import RynnVLAActionHiddenActor
from dreamer_vla.runners.eval_libero_vla_runner import EvalLiberoVLARunner


def _load_eval_cfg(overrides: list[str]) -> DictConfig:
    with hydra.initialize_config_dir(
        config_dir=str(PROJECT_ROOT / "configs"),
        version_base=None,
    ):
        return hydra.compose(config_name="eval_libero_vla", overrides=overrides)


def _hydra_quote(value: str) -> str:
    return json.dumps(str(value))


def _merge_train_eval_cfg(
    ws: EvalLiberoVLARunner, eval_cfg: DictConfig, ckpt_path: str
) -> tuple[DictConfig, dict[str, Any]]:
    payload = ws._load_checkpoint_payload(
        str(pathlib.Path(ckpt_path).expanduser().resolve())
    )
    train_cfg = copy.deepcopy(payload.get("cfg"))
    if train_cfg is None:
        raise RuntimeError(f"{ckpt_path} has no saved cfg")
    with open_dict(train_cfg):
        train_cfg.eval = copy.deepcopy(eval_cfg.eval)
        if OmegaConf.select(train_cfg, "encoder", default=None) is None:
            train_cfg.encoder = copy.deepcopy(eval_cfg.encoder)
        eval_vla_path = OmegaConf.select(eval_cfg, "init.vla_ckpt_path", default=None)
        if eval_vla_path is not None:
            train_cfg.init.vla_ckpt_path = eval_vla_path
            if OmegaConf.select(train_cfg, "encoder", default=None) is not None:
                train_cfg.encoder.model_path = eval_vla_path
        eval_encoder_ckpt = OmegaConf.select(
            eval_cfg, "init.encoder_state_ckpt", default=None
        )
        if eval_encoder_ckpt is not None:
            train_cfg.init.encoder_state_ckpt = eval_encoder_ckpt
        eval_horizon = OmegaConf.select(eval_cfg, "encoder.time_horizon", default=None)
        if (
            eval_horizon is not None
            and OmegaConf.select(train_cfg, "encoder", default=None) is not None
        ):
            train_cfg.encoder.time_horizon = eval_horizon
        train_cfg.training.distributed_strategy = "ddp"
        train_cfg.training.enable_activation_checkpointing = False
        train_cfg.trainer.device = str(eval_cfg.trainer.device)
    return train_cfg, payload


def _init_runner(
    train_cfg: DictConfig, payload: dict[str, Any], output_dir: str
) -> EvalLiberoVLARunner:
    ws = EvalLiberoVLARunner(train_cfg, output_dir=output_dir)
    ws.cfg = train_cfg
    ws.config = train_cfg
    ws._dreamer_eval = True
    ws._dreamer_deterministic = True
    ws._dreamer_action_repeat = int(
        OmegaConf.select(train_cfg, "eval.dreamer_action_repeat", default=1)
    )
    ws._dreamer_clip_actions = bool(
        OmegaConf.select(train_cfg, "eval.dreamer_clip_actions", default=True)
    )
    ws._dreamer_rollout_mode = str(
        OmegaConf.select(train_cfg, "eval.dreamer_rollout_mode", default="online_rssm")
    )
    ws._dreamer_actor_input_source = str(
        OmegaConf.select(train_cfg, "eval.dreamer_actor_input_source", default="rssm")
    )
    ws._dreamer_policy_source = str(
        OmegaConf.select(train_cfg, "eval.dreamer_policy_source", default="ckpt")
    )
    ws._hidden_noise_std = 0.0
    ws._hidden_noise_seed = 0
    ws._hidden_noise_generator = torch.Generator(device=ws.device)
    ws._hidden_noise_generator.manual_seed(0)
    ws._hidden_action_compare_enabled = False
    ws._hidden_action_compare_count = 0
    ws._init_policy_trace(train_cfg)
    ws._build_dreamer_modules(train_cfg, payload)
    return ws


class Collector:
    def __init__(self) -> None:
        self.values: dict[str, list[torch.Tensor]] = {}

    def add(self, key: str, value: torch.Tensor) -> None:
        x = value.detach().float().cpu()
        if x.ndim > 2:
            x = x.reshape(-1, x.shape[-1])
        elif x.ndim == 1:
            x = x[None, :]
        self.values.setdefault(key, []).append(x)

    def summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, chunks in sorted(self.values.items()):
            x = torch.cat(chunks, dim=0)
            row_norm = x.norm(dim=-1)
            centered = x - x.mean(dim=0, keepdim=True)
            out[key] = {
                "shape": list(x.shape),
                "mean": float(x.mean()),
                "std": float(x.std(unbiased=False)),
                "abs_mean": float(x.abs().mean()),
                "row_norm_mean": float(row_norm.mean()),
                "row_norm_std": float(row_norm.std(unbiased=False)),
                "row_norm_min": float(row_norm.min()),
                "row_norm_max": float(row_norm.max()),
                "centered_row_norm_mean": float(centered.norm(dim=-1).mean()),
                "min": float(x.min()),
                "max": float(x.max()),
            }
            if x.ndim == 2 and 1 < x.shape[1] <= 16:
                out[key]["per_dim_mean"] = [float(v) for v in x.mean(dim=0)]
                out[key]["per_dim_std"] = [
                    float(v) for v in x.std(dim=0, unbiased=False)
                ]
                out[key]["per_dim_abs_mean"] = [float(v) for v in x.abs().mean(dim=0)]
                out[key]["per_dim_min"] = [float(v) for v in x.min(dim=0).values]
                out[key]["per_dim_max"] = [float(v) for v in x.max(dim=0).values]
        return out


class PairCollector:
    def __init__(self) -> None:
        self.rows: dict[str, list[torch.Tensor]] = {}

    def add(self, key: str, left: torch.Tensor, right: torch.Tensor) -> None:
        a = left.detach().float().cpu()
        b = right.detach().float().cpu()
        if a.ndim > 2:
            a = a.reshape(-1, a.shape[-1])
        elif a.ndim == 1:
            a = a[None, :]
        if b.ndim > 2:
            b = b.reshape(-1, b.shape[-1])
        elif b.ndim == 1:
            b = b[None, :]
        n = min(a.shape[0], b.shape[0])
        if n <= 0:
            return
        a = a[:n]
        b = b[:n]
        mse = (a - b).square().mean(dim=-1, keepdim=True)
        l2 = (a - b).norm(dim=-1, keepdim=True)
        cos = torch.nn.functional.cosine_similarity(a, b, dim=-1).reshape(-1, 1)
        self.rows.setdefault(f"{key}/mse", []).append(mse)
        self.rows.setdefault(f"{key}/l2", []).append(l2)
        self.rows.setdefault(f"{key}/cos", []).append(cos)

    def summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, chunks in sorted(self.rows.items()):
            x = torch.cat(chunks, dim=0).reshape(-1)
            out[key] = {
                "count": int(x.numel()),
                "mean": float(x.mean()),
                "std": float(x.std(unbiased=False)),
                "min": float(x.min()),
                "max": float(x.max()),
            }
        return out


def _add_latent_stats(
    prefix: str, collector: Collector, world_model: torch.nn.Module, latent: Any
) -> None:
    collector.add(f"{prefix}/deter_h", latent.deter)
    collector.add(
        f"{prefix}/stoch_z_flat", latent.stoch.reshape(*latent.stoch.shape[:-2], -1)
    )
    collector.add(f"{prefix}/feature_hz", latent.feature())
    actor_in = world_model({"mode": "actor_input", "latent": latent})
    collector.add(f"{prefix}/actor_input", actor_in)
    logits = getattr(latent, "logits", None)
    if logits is not None:
        probs = torch.softmax(logits.float(), dim=-1)
        entropy = -(probs.clamp_min(1e-8) * probs.clamp_min(1e-8).log()).sum(dim=-1)
        collector.add(f"{prefix}/post_entropy_per_stoch", entropy)
        collector.add(f"{prefix}/post_max_prob_per_stoch", probs.max(dim=-1).values)


def _make_original_rynnvla_actor(
    ws: EvalLiberoVLARunner,
) -> RynnVLAActionHiddenActor:
    cfg = ws.cfg
    actor = RynnVLAActionHiddenActor(
        hidden_dim=OmegaConf.select(cfg, "policy.hidden_dim", default=None),
        action_hidden_dim=int(
            OmegaConf.select(cfg, "policy.action_hidden_dim", default=1024)
        ),
        action_dim=int(OmegaConf.select(cfg, "policy.action_dim", default=7)),
        time_horizon=int(OmegaConf.select(cfg, "policy.time_horizon", default=5)),
        adapter_type="identity",
        freeze_output_projection=True,
        init_action_head_ckpt=str(OmegaConf.select(cfg, "init.encoder_state_ckpt")),
    ).to(ws.device)
    target_dtype = next(ws.policy.parameters()).dtype
    actor = actor.to(dtype=target_dtype)
    actor.eval()
    return actor


@torch.no_grad()
def _actor_action_chunk(actor: torch.nn.Module, hidden: torch.Tensor) -> torch.Tensor:
    action, _, _ = actor(
        {
            "mode": "sample",
            "hidden": hidden,
            "deterministic": True,
            "return_chunk": True,
        }
    )
    return action.detach().float()


def _env_action_chunk(
    ws: EvalLiberoVLARunner, raw_chunk: torch.Tensor
) -> torch.Tensor:
    raw = raw_chunk.detach().float().cpu().numpy().reshape(-1, raw_chunk.shape[-1])
    env = [
        ws._dreamer_policy_raw_to_env_action(np.asarray(row[:7], dtype=np.float32))
        for row in raw
    ]
    return torch.from_numpy(np.stack(env, axis=0).astype(np.float32))


def _add_action_stats(
    prefix: str,
    collector: Collector,
    ws: EvalLiberoVLARunner,
    raw_chunk: torch.Tensor,
) -> None:
    raw = raw_chunk.detach().float().cpu()
    env = _env_action_chunk(ws, raw_chunk)
    collector.add(f"{prefix}/raw_action", raw)
    collector.add(f"{prefix}/env_action", env)
    sat = (env.abs() > 0.95).float()
    collector.add(f"{prefix}/env_action_saturation_gt_0p95", sat)


@torch.no_grad()
def collect_offline(
    ws: EvalLiberoVLARunner, train_cfg: DictConfig, batches: int, batch_size: int
) -> dict[str, Any]:
    dataset_cfg = copy.deepcopy(train_cfg.dataset)
    dataset = hydra.utils.instantiate(
        dataset_cfg, max_windows=max(batches * batch_size, batch_size)
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    collector = Collector()
    for idx, batch in enumerate(loader):
        if idx >= batches:
            break
        model_batch = {
            "obs_embedding": batch["obs_embedding"].to(ws.device),
            "actions": batch["actions"].to(ws.device),
            "is_first": batch["is_first"].to(ws.device),
        }
        collector.add("offline/obs_embedding", model_batch["obs_embedding"])
        observed = ws.world_model({"mode": "observe_sequence", **model_batch})
        _add_latent_stats(
            "offline/posterior", collector, ws.world_model, observed["latent"]
        )
    original_actor = _make_original_rynnvla_actor(ws)
    pair = PairCollector()
    for idx, batch in enumerate(loader):
        if idx >= batches:
            break
        obs_embedding = batch["obs_embedding"].to(ws.device)
        model_batch = {
            "obs_embedding": obs_embedding,
            "actions": batch["actions"].to(ws.device),
            "is_first": batch["is_first"].to(ws.device),
        }
        observed = ws.world_model({"mode": "observe_sequence", **model_batch})
        actor_input = ws.world_model(
            {"mode": "actor_input", "latent": observed["latent"]}
        )
        live = obs_embedding.reshape(-1, obs_embedding.shape[-1])
        recon = actor_input.reshape(-1, actor_input.shape[-1])
        original_live = _actor_action_chunk(original_actor, live)
        original_recon = _actor_action_chunk(original_actor, recon)
        trained_live = _actor_action_chunk(ws.policy, live)
        trained_recon = _actor_action_chunk(ws.policy, recon)
        _add_action_stats("offline_actor/original_live", collector, ws, original_live)
        _add_action_stats("offline_actor/original_recon", collector, ws, original_recon)
        _add_action_stats("offline_actor/trained_live", collector, ws, trained_live)
        _add_action_stats("offline_actor/trained_recon", collector, ws, trained_recon)
        pair.add("offline_pair/hidden_live_vs_recon", live, recon)
        pair.add(
            "offline_pair/action_original_live_vs_trained_recon_raw",
            original_live,
            trained_recon,
        )
        pair.add(
            "offline_pair/action_original_live_vs_original_recon_raw",
            original_live,
            original_recon,
        )
        pair.add(
            "offline_pair/action_original_live_vs_trained_live_raw",
            original_live,
            trained_live,
        )
    summary = collector.summary()
    summary.update(pair.summary())
    return summary


@torch.no_grad()
def collect_online(
    ws: EvalLiberoVLARunner,
    task_ids: list[int],
    episodes_per_task: int,
    max_steps: int,
) -> dict[str, Any]:
    from libero.libero import benchmark as libero_benchmark

    eval_cfg = OmegaConf.select(ws.cfg, "eval")
    task_suite_name = str(
        OmegaConf.select(eval_cfg, "task_suite_name", default="libero_goal")
    )
    resolution = int(OmegaConf.select(ws.cfg, "encoder.resolution", default=256))
    history_length = int(OmegaConf.select(eval_cfg, "history_length", default=2))
    action_steps = int(OmegaConf.select(eval_cfg, "action_steps", default=5))
    item_processor = ws.encoder._build_processor(ws.device)
    task_suite = libero_benchmark.get_benchmark_dict()[task_suite_name]()
    collector = Collector()
    action_collector = Collector()
    pair_collector = PairCollector()
    original_actor = _make_original_rynnvla_actor(ws)

    for task_id in task_ids:
        task = task_suite.get_task(int(task_id))
        initial_states = task_suite.get_task_init_states(int(task_id))
        env, task_description = get_libero_env(task, resolution=resolution)
        for episode_idx in range(min(int(episodes_per_task), len(initial_states))):
            ws._dreamer_online_reset()
            env.reset()
            obs = env.set_init_state(initial_states[episode_idx])
            done = False
            for _ in range(10):
                obs, _, done, _ = env.step(get_libero_dummy_action())
                if done:
                    break
            frame_history: list[tuple[Image.Image, Image.Image]] = []
            env_actions_buffer: list[np.ndarray] = []
            rssm_actions_buffer: list[np.ndarray] = []
            for _step_idx in range(int(max_steps)):
                img = get_libero_image(obs, resolution)
                wrist_img = get_libero_image(
                    obs, resolution, "robot0_eye_in_hand_image"
                )
                state = np.concatenate(
                    (
                        obs["robot0_eef_pos"],
                        quat2axisangle(obs["robot0_eef_quat"]),
                        obs["robot0_gripper_qpos"],
                    )
                )
                frame_history.append((Image.fromarray(img), Image.fromarray(wrist_img)))
                if len(frame_history) > history_length:
                    frame_history = frame_history[-history_length:]
                padded = [frame_history[0]] * (
                    history_length - len(frame_history)
                ) + frame_history
                ws._libero_current_raw_obs = obs
                obs_embedding, input_ids = ws._dreamer_obs_embedding_from_eval_inputs(
                    item_processor,
                    padded,
                    state,
                    task_description,
                )
                collector.add("online/obs_embedding", obs_embedding)
                latent = ws._dreamer_online_update_latent(obs_embedding)
                _add_latent_stats("online/posterior", collector, ws.world_model, latent)
                actor_input = ws.world_model(
                    {"mode": "actor_input", "latent": latent}
                ).float()
                live_flat = obs_embedding.reshape(obs_embedding.shape[0], -1)
                recon_flat = actor_input.reshape(actor_input.shape[0], -1)
                original_live = _actor_action_chunk(original_actor, live_flat)
                original_recon = _actor_action_chunk(original_actor, recon_flat)
                trained_live = _actor_action_chunk(ws.policy, live_flat)
                trained_recon = _actor_action_chunk(ws.policy, recon_flat)
                _add_action_stats(
                    "online_actor/original_live", collector, ws, original_live
                )
                _add_action_stats(
                    "online_actor/original_recon", collector, ws, original_recon
                )
                _add_action_stats(
                    "online_actor/trained_live", collector, ws, trained_live
                )
                _add_action_stats(
                    "online_actor/trained_recon", collector, ws, trained_recon
                )
                pair_collector.add(
                    "online_pair/hidden_live_vs_recon", live_flat, recon_flat
                )
                pair_collector.add(
                    "online_pair/action_original_live_vs_trained_recon_raw",
                    original_live,
                    trained_recon,
                )
                pair_collector.add(
                    "online_pair/action_original_live_vs_original_recon_raw",
                    original_live,
                    original_recon,
                )
                pair_collector.add(
                    "online_pair/action_original_live_vs_trained_live_raw",
                    original_live,
                    trained_live,
                )
                if not env_actions_buffer:
                    env_actions_buffer, rssm_actions_buffer = (
                        ws._dreamer_action_chunk_from_latent(
                            latent,
                            input_ids=input_ids,
                            action_steps=action_steps,
                            live_hidden=obs_embedding,
                        )
                    )
                if not env_actions_buffer:
                    break
                action = env_actions_buffer.pop(0)
                rssm_action = (
                    rssm_actions_buffer.pop(0) if rssm_actions_buffer else action
                )
                action_collector.add(
                    "online/env_action",
                    torch.from_numpy(np.asarray(action, dtype=np.float32)),
                )
                obs, _, done, _ = env.step(action.tolist())
                ws._dreamer_online_prev_action = (
                    torch.from_numpy(np.asarray(rssm_action, dtype=np.float32))
                    .to(ws.device)
                    .reshape(1, -1)
                )
                if done:
                    break
    summary = collector.summary()
    summary.update(action_collector.summary())
    summary.update(pair_collector.summary())
    return summary


def _compare(offline: dict[str, Any], online: dict[str, Any]) -> dict[str, Any]:
    pairs = [
        ("obs_embedding", "offline/obs_embedding", "online/obs_embedding"),
        ("deter_h", "offline/posterior/deter_h", "online/posterior/deter_h"),
        ("stoch_z", "offline/posterior/stoch_z_flat", "online/posterior/stoch_z_flat"),
        ("feature_hz", "offline/posterior/feature_hz", "online/posterior/feature_hz"),
        (
            "actor_input",
            "offline/posterior/actor_input",
            "online/posterior/actor_input",
        ),
    ]
    out: dict[str, Any] = {}
    for name, off_key, on_key in pairs:
        if off_key not in offline or on_key not in online:
            continue
        off = offline[off_key]
        on = online[on_key]
        out[name] = {
            "mean_delta": on["mean"] - off["mean"],
            "std_ratio_online_over_offline": on["std"] / max(off["std"], 1e-12),
            "row_norm_ratio_online_over_offline": on["row_norm_mean"]
            / max(off["row_norm_mean"], 1e-12),
            "centered_norm_ratio_online_over_offline": on["centered_row_norm_mean"]
            / max(off["centered_row_norm_mean"], 1e-12),
        }
    return out


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def _metric(section: dict[str, Any], key: str, field: str, default: Any = None) -> Any:
    return section.get(key, {}).get(field, default)


def _pair(section: dict[str, Any], key: str, suffix: str) -> Any:
    return _metric(section, f"{key}/{suffix}", "mean")


def _write_markdown_report(
    result: dict[str, Any], out_path: pathlib.Path
) -> pathlib.Path:
    report_path = out_path.with_suffix(".md")
    offline = result.get("offline", {})
    online = result.get("online", {})
    compare = result.get("compare", {})
    action_labels = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "grip"]

    lines: list[str] = []
    lines.append("# DreamerVLA Actor/Input Diagnostic Report")
    lines.append("")
    lines.append(f"- ckpt: `{result.get('ckpt')}`")
    lines.append(f"- encoder_ckpt: `{result.get('encoder_ckpt')}`")
    lines.append(f"- tasks: `{result.get('tasks')}`")
    lines.append("")
    lines.append("## 1. Input And Latent Alignment")
    lines.append("")
    lines.append(
        "| signal | mean_delta | std_ratio online/offline | norm_ratio online/offline | centered_norm_ratio |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for name in ["obs_embedding", "deter_h", "stoch_z", "feature_hz", "actor_input"]:
        row = compare.get(name, {})
        lines.append(
            f"| {name} | {_fmt(row.get('mean_delta'))} | "
            f"{_fmt(row.get('std_ratio_online_over_offline'))} | "
            f"{_fmt(row.get('row_norm_ratio_online_over_offline'))} | "
            f"{_fmt(row.get('centered_norm_ratio_online_over_offline'))} |"
        )
    lines.append("")
    lines.append("## 2. Action Hidden Reconstruction")
    lines.append("")
    lines.append(
        "| split | hidden live-vs-recon cos | hidden live-vs-recon mse | original actor live-vs-recon cos | original actor live-vs-recon mse |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for split, section in [("offline", offline), ("online", online)]:
        lines.append(
            f"| {split} | "
            f"{_fmt(_pair(section, f'{split}_pair/hidden_live_vs_recon', 'cos'))} | "
            f"{_fmt(_pair(section, f'{split}_pair/hidden_live_vs_recon', 'mse'))} | "
            f"{_fmt(_pair(section, f'{split}_pair/action_original_live_vs_original_recon_raw', 'cos'))} | "
            f"{_fmt(_pair(section, f'{split}_pair/action_original_live_vs_original_recon_raw', 'mse'))} |"
        )
    lines.append("")
    lines.append("## 3. Original Actor Vs Trained Actor")
    lines.append("")
    lines.append(
        "| split | comparison | raw_action cos | raw_action mse | raw_action l2 |"
    )
    lines.append("| --- | --- | ---: | ---: | ---: |")
    for split, section in [("offline", offline), ("online", online)]:
        for label, key in [
            (
                "original_live vs trained_live",
                f"{split}_pair/action_original_live_vs_trained_live_raw",
            ),
            (
                "original_live vs trained_recon",
                f"{split}_pair/action_original_live_vs_trained_recon_raw",
            ),
        ]:
            lines.append(
                f"| {split} | {label} | "
                f"{_fmt(_pair(section, key, 'cos'))} | "
                f"{_fmt(_pair(section, key, 'mse'))} | "
                f"{_fmt(_pair(section, key, 'l2'))} |"
            )
    lines.append("")
    lines.append("## 4. Action Distribution")
    lines.append("")
    lines.append(
        "| split | actor/input | env_abs_mean | env_std | env_norm | saturation_gt_0.95 |"
    )
    lines.append("| --- | --- | ---: | ---: | ---: | ---: |")
    for split, section in [("offline", offline), ("online", online)]:
        for actor_name in [
            "original_live",
            "original_recon",
            "trained_live",
            "trained_recon",
        ]:
            prefix = f"{split}_actor/{actor_name}"
            lines.append(
                f"| {split} | {actor_name} | "
                f"{_fmt(_metric(section, f'{prefix}/env_action', 'abs_mean'))} | "
                f"{_fmt(_metric(section, f'{prefix}/env_action', 'std'))} | "
                f"{_fmt(_metric(section, f'{prefix}/env_action', 'row_norm_mean'))} | "
                f"{_fmt(_metric(section, f'{prefix}/env_action_saturation_gt_0p95', 'mean'))} |"
            )
    lines.append("")
    lines.append("## 5. Online Per-Dimension Env Action Means")
    lines.append("")
    lines.append("| actor/input | " + " | ".join(action_labels) + " |")
    lines.append("| --- | " + " | ".join(["---:"] * len(action_labels)) + " |")
    for actor_name in [
        "original_live",
        "original_recon",
        "trained_live",
        "trained_recon",
    ]:
        values = _metric(
            online, f"online_actor/{actor_name}/env_action", "per_dim_mean", []
        )
        lines.append(
            "| "
            + actor_name
            + " | "
            + " | ".join(_fmt(v, 3) for v in values[:7])
            + " |"
        )
    lines.append("")
    lines.append("## Required Interpretation")
    lines.append("")
    lines.append(
        "- Input alignment is acceptable when `obs_embedding` std/norm ratios are near 1.0 and prompt/history settings match the sidecar attrs."
    )
    lines.append(
        "- Action hidden reconstruction is acceptable when offline live-vs-recon hidden cosine is high and original actor live-vs-recon action MSE is low."
    )
    lines.append(
        "- A trained actor is suspect when its raw actions diverge strongly from original actor outputs or its env actions saturate/freeze dimensions."
    )
    lines.append("")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--encoder-ckpt", required=True)
    parser.add_argument(
        "--out",
        default=str(
            PROJECT_ROOT
            / "data/outputs/diagnostics/dreamervla_latent_distribution.json"
        ),
    )
    parser.add_argument("--tasks", default="0,3,6,8")
    parser.add_argument("--episodes-per-task", type=int, default=1)
    parser.add_argument("--online-steps", type=int, default=20)
    parser.add_argument("--offline-batches", type=int, default=4)
    parser.add_argument("--offline-batch-size", type=int, default=4)
    args = parser.parse_args()

    overrides = [
        f"eval.ckpt_path={_hydra_quote(args.ckpt)}",
        "eval.ckpt_kind=dreamer",
        f"init.encoder_state_ckpt={_hydra_quote(args.encoder_ckpt)}",
        "+encoder.action_head_type=legacy",
        "eval.dreamer_rollout_mode=online_rssm",
        "eval.dreamer_actor_input_source=rssm",
        "eval.dreamer_unnorm_actions=auto",
        "eval.dreamer_rssm_action_source=env",
        "+eval.history_length=2",
        "eval.action_steps=5",
        "eval.dreamer_wm_history_length=2",
        "eval.dreamer_wm_rotate_images=true",
        "eval.obs_hidden_source=auto",
        "eval.log_action_stats=false",
    ]
    eval_cfg = _load_eval_cfg(overrides)
    output_dir = str(
        pathlib.Path(args.out).expanduser().resolve().parent / "_runner"
    )
    ws0 = EvalLiberoVLARunner(eval_cfg, output_dir=output_dir)
    train_cfg, payload = _merge_train_eval_cfg(ws0, eval_cfg, args.ckpt)
    ws = _init_runner(train_cfg, payload, output_dir=output_dir)
    tasks = [int(x) for x in str(args.tasks).split(",") if x.strip()]

    result = {
        "ckpt": str(pathlib.Path(args.ckpt).expanduser().resolve()),
        "encoder_ckpt": str(pathlib.Path(args.encoder_ckpt).expanduser().resolve()),
        "tasks": tasks,
        "offline": collect_offline(
            ws, train_cfg, args.offline_batches, args.offline_batch_size
        ),
        "online": collect_online(ws, tasks, args.episodes_per_task, args.online_steps),
    }
    result["compare"] = _compare(result["offline"], result["online"])

    out_path = pathlib.Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    report_path = _write_markdown_report(result, out_path)
    print(json.dumps(result["compare"], indent=2))
    print(f"[latent-diagnostic] wrote {out_path}")
    print(f"[latent-diagnostic] wrote {report_path}")


if __name__ == "__main__":
    main()
