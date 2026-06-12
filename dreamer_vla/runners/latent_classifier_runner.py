"""LatentSuccessClassifier training runner (WMPO-aligned).

Launch path:
    bash scripts/train_wm.sh experiment=latent_classifier_libero_goal_chunk
        → python -m dreamer_vla.train --config-name train experiment=latent_classifier_libero_goal_chunk
            → dreamer_vla.runners.LatentClassifierRunner.run()
                → dreamer_vla.dataset.wmpo_aligned_latent_dataset
                → dreamer_vla.models.reward.LatentSuccessClassifier

Why a dedicated runner, not another standalone script:
  * The existing v3 / wm_replay classifier scripts are 500+ lines each and bypass
    BaseRunner, so resume / checkpoint / Hydra-override semantics don't
    transfer. A runner fixes that.
  * Decouples sampling protocol (dataset), model (LatentSuccessClassifier head_type),
    and training loop (this runner) — so head_type ablation is a 1-line config
    override, not a script fork.

The training loop is step-based (matches WMPO's ``MAX_STEPS=200_000`` paradigm):
  * Infinite resampled train stream → ``cfg.training.max_steps`` total optimizer steps
  * Eval every ``cfg.training.eval_every`` steps; window F1 + (optional) episode F1
  * Best ckpt saved by val window F1 (sigmoid + threshold sweep, WMPO protocol)
  * Final ckpt saved at ``max_steps``

Window-level F1 uses sigmoid + threshold sweep to mirror WMPO's
``_evaluate_terminal_model`` (note: WMPO sweep is [0.3, 1.0]; we expose the bounds
via cfg). Episode-level F1 mirrors ``predict_success`` (stride-1 sliding window +
``any-positive`` aggregation).

The runner owns resume, checkpointing, logging, and Hydra override behavior so
classifier training follows the same contract as WM and DreamerVLA routes.
"""

from __future__ import annotations

import json
import pathlib
import time
from typing import Any

import hydra
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader

from dreamer_vla.dataset.wmpo_aligned_latent_dataset import (
    WMPOAlignedLatentTrainDataset,
    WMPOAlignedLatentValDataset,
)
from dreamer_vla.models.reward import LatentSuccessClassifier, LatentSuccessClassifierConfig
from dreamer_vla.runners.base_runner import BaseRunner

# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class LatentClassifierRunner(BaseRunner):
    """Single-GPU step-based trainer for LatentSuccessClassifier.

    Lifecycle: setup() → run() → teardown() (via BaseRunner.execute).
    """

    runner_name = "latent_classifier"
    runner_status = "current"
    runner_family = "reward"
    include_keys = ("_output_dir",)

    def __init__(self, config: DictConfig, output_dir: str | None = None) -> None:
        super().__init__(config, output_dir)
        self.device = torch.device(
            str(OmegaConf.select(self.cfg, "training.device") or "cuda")
        )
        self.train_ds: WMPOAlignedLatentTrainDataset | None = None
        self.val_ds: WMPOAlignedLatentValDataset | None = None
        self.train_loader: DataLoader | None = None
        self.val_loader: DataLoader | None = None
        self.model: LatentSuccessClassifier | None = None
        self.optim: torch.optim.Optimizer | None = None
        self.best_window_f1: float = -1.0
        self.best_episode_f1: float = -1.0
        self._log_path: pathlib.Path | None = None

    # --------------------------- setup ---------------------------------

    def setup(self) -> None:
        super().setup()
        torch.manual_seed(int(OmegaConf.select(self.cfg, "training.seed") or 0))
        torch.backends.cudnn.benchmark = True

        ckpt_dir = self.get_checkpoint_dir()
        log_dir = self.get_log_dir()
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
        self._log_path = log_dir / "train_log.jsonl"
        # truncate at startup so reruns don't pollute the file
        self._log_path.write_text("")

        self._log(
            {
                "event": "setup_begin",
                "output_dir": str(self.output_dir),
                "device": str(self.device),
                "head_type": str(
                    OmegaConf.select(self.cfg, "classifier.head_type") or "linear"
                ),
            }
        )

        # ----- datasets ---------------------------------------------------
        d = self.cfg.data
        # Chunk-level (Type B) classifier: each window frame is pooled from
        # K = chunk_subsample env-step frames. Defaults reduce to env-step
        # (action) granularity for backwards compatibility.
        chunk_subsample = int(OmegaConf.select(d, "chunk_subsample") or 1)
        chunk_pool = str(OmegaConf.select(d, "chunk_pool") or "last")
        self.train_ds = WMPOAlignedLatentTrainDataset(
            success_dir_raw=d.success_dir_raw,
            success_dir_hidden=d.success_dir_hidden,
            failure_dir_raw=OmegaConf.select(d, "failure_dir_raw"),
            failure_dir_hidden=OmegaConf.select(d, "failure_dir_hidden"),
            window=int(d.window),
            stride=int(d.stride_train),
            seed=int(OmegaConf.select(self.cfg, "training.seed") or 0),
            chunk_subsample=chunk_subsample,
            chunk_pool=chunk_pool,
        )
        self.val_ds = WMPOAlignedLatentValDataset(
            success_dir_raw=d.success_dir_raw,
            success_dir_hidden=d.success_dir_hidden,
            failure_dir_raw=OmegaConf.select(d, "failure_dir_raw"),
            failure_dir_hidden=OmegaConf.select(d, "failure_dir_hidden"),
            window=int(d.window),
            stride=int(d.stride_val),
            chunk_subsample=chunk_subsample,
            chunk_pool=chunk_pool,
        )

        # ----- dataloaders -----------------------------------------------
        tr = self.cfg.training
        self.train_loader = DataLoader(
            self.train_ds,
            batch_size=int(tr.batch_size),
            num_workers=int(OmegaConf.select(tr, "num_workers") or 0),
            pin_memory=True,
            collate_fn=self.train_ds.collate_fn,
            drop_last=True,
        )
        self.val_loader = DataLoader(
            self.val_ds,
            batch_size=int(OmegaConf.select(tr, "val_batch_size") or 256),
            num_workers=0,  # val data is in-RAM; workers add overhead
            pin_memory=True,
            collate_fn=self.val_ds.collate_fn,
            shuffle=False,
            drop_last=False,
        )

        # ----- model -----------------------------------------------------
        cfg_dict = OmegaConf.to_container(self.cfg.classifier, resolve=True)
        # only keep keys LatentSuccessClassifierConfig accepts
        valid_keys = LatentSuccessClassifierConfig.__dataclass_fields__.keys()
        cfg_dict = {k: v for k, v in cfg_dict.items() if k in valid_keys}
        cls_cfg = LatentSuccessClassifierConfig(**cfg_dict)
        if int(cls_cfg.window) != int(d.window):
            raise ValueError(
                f"classifier.window ({cls_cfg.window}) != data.window ({d.window})"
            )
        # Chunk granularity consistency: classifier.cfg.chunk_size must match
        # data.chunk_subsample, otherwise the windows produced by the dataset
        # have a different time-coverage than what the classifier expects at
        # inference time (predict_success internally subsamples by chunk_size).
        if str(getattr(cls_cfg, "granularity", "action")) == "chunk":
            if int(cls_cfg.chunk_size) != chunk_subsample:
                raise ValueError(
                    f"classifier.chunk_size ({cls_cfg.chunk_size}) != "
                    f"data.chunk_subsample ({chunk_subsample})"
                )
            if str(cls_cfg.chunk_pool) != chunk_pool:
                raise ValueError(
                    f"classifier.chunk_pool ({cls_cfg.chunk_pool!r}) != "
                    f"data.chunk_pool ({chunk_pool!r})"
                )
        elif chunk_subsample != 1:
            raise ValueError(
                f"data.chunk_subsample={chunk_subsample} requires classifier.granularity='chunk'"
            )
        classifier_target = OmegaConf.select(
            self.cfg, "classifier._target_", default=None
        )
        if classifier_target:
            self.model = hydra.utils.instantiate(self.cfg.classifier).to(self.device)
        else:
            self.model = LatentSuccessClassifier(cls_cfg).to(self.device)
        n_params = sum(p.numel() for p in self.model.parameters())
        n_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self._log(
            {
                "event": "model_built",
                "head_type": str(cls_cfg.head_type),
                "n_params": int(n_params),
                "n_trainable": int(n_trainable),
            }
        )

        # ----- optimizer + (optional) scheduler --------------------------
        self.optim = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(tr.lr),
            weight_decay=float(OmegaConf.select(tr, "weight_decay") or 1e-4),
        )

        # ----- resume (BaseRunner.resume) -----------------------------
        if bool(OmegaConf.select(tr, "resume") or False):
            try:
                self.resume(self.cfg)
            except Exception as exc:  # noqa: BLE001
                self._log({"event": "resume_failed", "error": repr(exc)})

        self._log({"event": "setup_done"})

    # --------------------------- run -----------------------------------

    def run(self) -> dict[str, float]:
        assert self.model is not None and self.optim is not None
        assert self.train_loader is not None and self.val_loader is not None

        tr = self.cfg.training
        max_steps = int(tr.max_steps)
        eval_every = int(OmegaConf.select(tr, "eval_every") or 500)
        ckpt_every = int(OmegaConf.select(tr, "ckpt_every") or eval_every)
        log_every = int(OmegaConf.select(tr, "log_every") or 50)
        label_smoothing = float(OmegaConf.select(tr, "label_smoothing") or 0.0)

        # class-balanced CE (matches WMPO `nn.CrossEntropyLoss()` *unweighted* by
        # default; user can flip via cfg.training.class_balanced)
        class_balanced = bool(OmegaConf.select(tr, "class_balanced") or False)
        if class_balanced:
            n_succ = sum(1 for d in self.train_ds._demos if d.complete)
            n_fail = len(self.train_ds._demos) - n_succ
            n_pos = n_succ
            n_neg = n_succ + 2 * n_fail
            cw = torch.tensor([1.0, n_neg / max(n_pos, 1)], device=self.device)
            self._log(
                {
                    "event": "class_balanced",
                    "class_weight": [1.0, float(n_neg / max(n_pos, 1))],
                }
            )
        else:
            cw = None
        loss_fn = nn.CrossEntropyLoss(weight=cw, label_smoothing=label_smoothing)

        train_iter = iter(self.train_loader)
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        t0 = time.time()
        while self.global_step < max_steps:
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_loader)
                batch = next(train_iter)
            xs, ys = batch
            xs = xs.to(self.device, non_blocking=True)
            ys = ys.to(self.device, non_blocking=True)

            self.model.train()
            logits = self.model(xs)
            loss = loss_fn(logits, ys)
            self.optim.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            )
            self.optim.step()

            with torch.no_grad():
                pred = logits.argmax(dim=-1)
                running_correct += int((pred == ys).sum().item())
                running_total += int(ys.numel())
            running_loss += float(loss.item())

            self.global_step += 1

            if self.global_step % log_every == 0:
                self._log(
                    {
                        "event": "train_step",
                        "step": self.global_step,
                        "loss": running_loss / log_every,
                        "acc": running_correct / max(running_total, 1),
                        "grad_norm": grad_norm,
                        "wall_s": time.time() - t0,
                    }
                )
                running_loss = 0.0
                running_correct = 0
                running_total = 0

            # ---- periodic eval ---------------------------------------
            if self.global_step % eval_every == 0:
                w_metrics = self._evaluate_window_level()
                self._log(
                    {"event": "val_window", "step": self.global_step, **w_metrics}
                )
                if w_metrics["best_f1"] > self.best_window_f1:
                    self.best_window_f1 = float(w_metrics["best_f1"])
                    self._save_named(
                        f"best_window_f1{w_metrics['best_f1']:.4f}_th{w_metrics['best_thresh']:.2f}",
                        extra={"val_window": w_metrics},
                    )

                if bool(OmegaConf.select(tr, "episode_eval_enabled") or False):
                    e_metrics = self._evaluate_episode_level()
                    self._log(
                        {"event": "val_episode", "step": self.global_step, **e_metrics}
                    )
                    if e_metrics["best_f1"] > self.best_episode_f1:
                        self.best_episode_f1 = float(e_metrics["best_f1"])
                        self._save_named(
                            f"best_episode_f1{e_metrics['best_f1']:.4f}_th{e_metrics['best_thresh']:.2f}",
                            extra={"val_episode": e_metrics},
                        )

            if self.global_step % ckpt_every == 0:
                self.save_checkpoint(tag="latest")

        # ---- final ckpt + summary ------------------------------------
        self.save_checkpoint(tag="final")
        summary = {
            "best_window_f1": self.best_window_f1,
            "best_episode_f1": self.best_episode_f1,
            "total_steps": int(self.global_step),
            "wall_s": time.time() - t0,
        }
        self._log({"event": "done", **summary})
        with open(pathlib.Path(self.output_dir) / "summary.json", "w") as fh:
            json.dump(summary, fh, indent=2)
        return summary

    # --------------------------- evaluation ----------------------------

    @torch.no_grad()
    def _evaluate_window_level(self) -> dict[str, Any]:
        """Sigmoid + threshold sweep over softmax(logits)[:, 1].

        Mirrors WMPO's _evaluate_terminal_model: P(success) = sigmoid(logit_class_1).
        (Equivalent decision boundary at high thresholds to softmax-based, but
        we use sigmoid here for 1:1 parity with WMPO's eval protocol.)
        """
        assert self.model is not None and self.val_loader is not None
        self.model.eval()
        probs_l: list[float] = []
        ys_l: list[int] = []
        for xs, ys, _ in self.val_loader:
            xs = xs.to(self.device, non_blocking=True)
            logits = self.model(xs)
            # WMPO uses sigmoid(logits)[:, 1] — see WMPO/verl/.../fsdp_workers.py:847
            probs = torch.sigmoid(logits)[:, 1].detach().cpu().numpy()
            probs_l.extend(probs.tolist())
            ys_l.extend(ys.tolist())
        probs = np.asarray(probs_l, dtype=np.float32)
        ys = np.asarray(ys_l, dtype=np.int64)

        tr = self.cfg.training
        thresholds = np.linspace(
            float(OmegaConf.select(tr, "thresh_min") or 0.3),
            float(OmegaConf.select(tr, "thresh_max") or 1.0),
            int(OmegaConf.select(tr, "thresh_steps") or 20),
        )
        return _sweep_metrics(probs, ys, thresholds, tag="window")

    @torch.no_grad()
    def _evaluate_episode_level(self) -> dict[str, Any]:
        """WMPO predict_success protocol — stride-1 sliding + any-positive.

        For each demo, scan stride-1 windows over the full trajectory (from
        ``min_steps + W`` to ``finish_step``). Use ``max`` over windows as the
        episode-level score, sweep thresholds, return best F1.

        Unit convention: ``episode_eval_min_steps`` and
        ``episode_eval_stride`` are in the classifier's NATIVE unit
        (env-step for action granularity, chunk for chunk granularity).
        The dataset's pooling K only affects how the env-step obs is folded
        to chunks for the sliding window; the gate values themselves are
        already chunk-unit in chunk configs.
        """
        assert self.model is not None and self.val_ds is not None
        tr = self.cfg.training
        W = int(self.cfg.data.window)
        min_steps = int(OmegaConf.select(tr, "episode_eval_min_steps") or 0)
        stride = int(OmegaConf.select(tr, "episode_eval_stride") or 1)
        ep_batch = int(OmegaConf.select(tr, "episode_eval_batch") or 256)

        K = int(getattr(self.val_ds, "K", 1))
        chunk_pool = str(getattr(self.val_ds, "chunk_pool", "last"))

        self.model.eval()
        ep_max_prob: list[float] = []
        ep_true: list[int] = []

        # Collect all windows from all episodes into flat batches, tagged
        # with episode idx; aggregate max per episode.
        flat_xs: list[np.ndarray] = []
        flat_ep: list[int] = []
        for ep_idx, (obs, complete, finish_step, _eid) in enumerate(
            self.val_ds.trajectories()
        ):
            T_env = int(min(finish_step, obs.shape[0]))
            if K > 1:
                T_chunk = T_env // K
                if T_chunk < 1:
                    obs_pooled = None
                    T = 0
                else:
                    trailing_shape = obs.shape[1:]
                    reshaped = obs[: T_chunk * K].reshape(
                        T_chunk, K, *trailing_shape
                    )
                    if chunk_pool == "last":
                        obs_pooled = reshaped[:, -1]
                    elif chunk_pool == "first":
                        obs_pooled = reshaped[:, 0]
                    else:
                        obs_pooled = reshaped.mean(axis=1)
                    T = T_chunk
            else:
                obs_pooled = obs
                T = T_env
            ep_true.append(int(bool(complete)))
            first_end = max(W, min_steps + W)
            if T < first_end or obs_pooled is None:
                ep_max_prob.append(0.0)
                continue
            ep_max_prob.append(-1.0)  # placeholder; updated below
            for end in range(first_end, T + 1, stride):
                flat_xs.append(obs_pooled[end - W : end])
                flat_ep.append(ep_idx)

        if flat_xs:
            i = 0
            n = len(flat_xs)
            while i < n:
                chunk = np.stack(flat_xs[i : i + ep_batch])
                logits = self.model(torch.from_numpy(chunk).float().to(self.device))
                p = torch.sigmoid(logits)[:, 1].detach().cpu().numpy()
                for j, pj in enumerate(p):
                    eid = flat_ep[i + j]
                    if pj > ep_max_prob[eid]:
                        ep_max_prob[eid] = float(pj)
                i += ep_batch

        # placeholder -1.0 → 0.0 (too-short episodes)
        ep_max_prob = [max(0.0, p) for p in ep_max_prob]

        probs = np.asarray(ep_max_prob, dtype=np.float32)
        ys = np.asarray(ep_true, dtype=np.int64)
        thresholds = np.linspace(
            float(OmegaConf.select(tr, "thresh_min") or 0.3),
            float(OmegaConf.select(tr, "thresh_max") or 1.0),
            int(OmegaConf.select(tr, "thresh_steps") or 20),
        )
        return _sweep_metrics(probs, ys, thresholds, tag="episode")

    # --------------------------- io helpers ----------------------------

    def _save_named(self, name: str, *, extra: dict | None = None) -> None:
        """Save in the format consumed by the online WMPO training script.

        Schema (matches the old v2/v3 trainer + WMPO predict_success consumer):
            model      : nn.Module.state_dict()
            threshold  : float — best operating point from the val sweep
            f1         : float — F1 at that threshold
            step       : int   — global_step at save time
            config     : { classifier: {…LatentSuccessClassifierConfig…} }
            extra      : the originating sweep dict (kept for offline analysis)
        """
        ckpt_dir = self.get_checkpoint_dir()
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        path = ckpt_dir / f"{name}.ckpt"
        # pick best F1/threshold out of the sweep dict that called us
        f1 = 0.0
        threshold = 0.5
        if isinstance(extra, dict):
            for k in ("val_episode", "val_window"):
                v = extra.get(k)
                if isinstance(v, dict):
                    f1 = float(v.get("best_f1", f1))
                    threshold = float(v.get("best_thresh", threshold))
                    break
        torch.save(
            {
                "model": self.model.state_dict(),
                "threshold": threshold,
                "f1": f1,
                "step": int(self.global_step),
                "config": {
                    "classifier": OmegaConf.to_container(
                        self.cfg.classifier, resolve=True
                    ),
                },
                "extra": extra or {},
            },
            path,
        )
        self._log(
            {"event": "ckpt_named", "path": str(path), "f1": f1, "threshold": threshold}
        )

    def _log(self, payload: dict) -> None:
        payload = {"ts": time.strftime("%H:%M:%S"), **payload}
        event = str(payload.get("event", ""))
        metric_prefix = "eval" if event.startswith("val_") else None
        if event == "train_step":
            metric_prefix = "train"
        step_value = payload.get("step", payload.get("global_step", self.global_step))
        self.log_metrics(payload, step=int(step_value), prefix=metric_prefix)
        print(json.dumps(payload), flush=True)
        if self._log_path is not None:
            with open(self._log_path, "a") as fh:
                fh.write(json.dumps(payload) + "\n")

    # ----------------------- BaseRunner exclusions ------------------

    # Datasets carry the in-memory demo cache (multi-GB); keep them out of
    # the BaseRunner state_dict serializer so save_checkpoint doesn't try
    # to pickle them.
    exclude_keys = ("train_ds", "val_ds", "train_loader", "val_loader")


# ---------------------------------------------------------------------------
# Threshold sweep
# ---------------------------------------------------------------------------


def _sweep_metrics(
    probs: np.ndarray, ys: np.ndarray, thresholds: np.ndarray, tag: str
) -> dict[str, Any]:
    best_f1 = -1.0
    best_thresh = float(thresholds[0])
    rows: dict[str, dict[str, float]] = {}
    for th in thresholds:
        preds = (probs >= th).astype(np.int64)
        f1 = float(f1_score(ys, preds, zero_division=0))
        rows[f"th_{th:.2f}"] = {
            "f1": f1,
            "acc": float(accuracy_score(ys, preds)),
            "prec": float(precision_score(ys, preds, zero_division=0)),
            "rec": float(recall_score(ys, preds, zero_division=0)),
        }
        if f1 > best_f1:
            best_f1, best_thresh = f1, float(th)
    return {
        "best_f1": best_f1,
        "best_thresh": best_thresh,
        "n": int(len(ys)),
        "n_pos": int((ys == 1).sum()),
        "n_neg": int((ys == 0).sum()),
        "tag": tag,
        # full sweep retained for offline analysis; small dict, ok to log
        "per_thresh": rows,
    }


__all__ = ["LatentClassifierRunner"]
