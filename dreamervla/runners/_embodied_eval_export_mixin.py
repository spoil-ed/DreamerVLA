"""Eval debug-export helpers for EmbodiedEvalRunner.

Cohesive, self-contained group extracted from embodied_eval_runner.py (P3 god-file
split, mixin route): real-rollout relabel export + policy-trace export. These methods
only touch ``self`` attributes/config and the pure helpers in ``_embodied_eval_helpers``
(no calls into other runner methods), so they live cleanly on a sibling mixin the runner
inherits — zero call-site change. Behaviour-preserving.
"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
from omegaconf import DictConfig, OmegaConf

from dreamervla.runners import _embodied_eval_helpers as _eh


class EmbodiedEvalExportMixin:
    def _init_real_relabel_export(self, cfg: DictConfig) -> None:
        self._real_relabel_enabled = bool(
            OmegaConf.select(cfg, "eval.export_real_relabel", default=False)
        )
        self._real_relabel_records: list[dict[str, Any]] = []
        self._real_relabel_success_rate = 0.0
        relabel_dir = OmegaConf.select(cfg, "eval.real_relabel_dir", default=None)
        if relabel_dir is None:
            relabel_dir = os.path.join(self.output_dir, "real_relabel")
        self._real_relabel_dir = str(relabel_dir)
        self._real_relabel_jsonl_path = os.path.join(
            self._real_relabel_dir, "real_rollout_relabel_records.jsonl"
        )
        self._real_relabel_summary_path = os.path.join(
            self._real_relabel_dir, "real_rollout_relabel_summary.json"
        )
        if self._real_relabel_enabled and self.distributed.is_main_process:
            os.makedirs(self._real_relabel_dir, exist_ok=True)
            with open(self._real_relabel_jsonl_path, "w"):
                pass

    _real_relabel_sparse_rewards = staticmethod(_eh.real_relabel_sparse_rewards)

    def _append_real_relabel_record(self, record: dict[str, Any]) -> None:
        if not bool(getattr(self, "_real_relabel_enabled", False)):
            return
        self._real_relabel_records.append(record)
        with open(self._real_relabel_jsonl_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def _write_real_relabel_summary(self) -> None:
        records = list(getattr(self, "_real_relabel_records", []))
        successes = int(sum(int(bool(row.get("complete", False))) for row in records))
        success_rate = successes / max(len(records), 1)
        self._real_relabel_success_rate = float(success_rate)
        groups: dict[str, list[dict[str, Any]]] = {}
        for row in records:
            groups.setdefault(str(row.get("prompt_key", "")), []).append(row)
        group_rows = []
        for prompt_key, rows in sorted(groups.items()):
            acc = (
                float(np.mean([float(row.get("acc", 0.0)) for row in rows]))
                if rows
                else 0.0
            )
            group_rows.append(
                {
                    "prompt_key": prompt_key,
                    "num_samples": len(rows),
                    "successes": int(
                        sum(int(bool(row.get("complete", False))) for row in rows)
                    ),
                    "acc_mean": acc,
                    "keep_by_accuracy_band": bool(0.01 <= acc <= 0.99),
                }
            )
        summary = {
            "num_records": len(records),
            "successes": successes,
            "success_rate": float(success_rate),
            "records_jsonl": str(getattr(self, "_real_relabel_jsonl_path", "")),
            "wmpo_style_filter": {
                "accuracy_lower_bound": 0.01,
                "accuracy_upper_bound": 0.99,
                "num_prompt_groups": len(group_rows),
                "num_kept_prompt_groups": int(
                    sum(int(row["keep_by_accuracy_band"]) for row in group_rows)
                ),
                "num_records": len(records),
                "num_kept_records": int(
                    sum(
                        len(groups[row["prompt_key"]])
                        for row in group_rows
                        if row["keep_by_accuracy_band"]
                    )
                ),
                "groups": group_rows,
            },
        }
        os.makedirs(self._real_relabel_dir, exist_ok=True)
        with open(self._real_relabel_summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(
            f"  [Eval] wrote real relabel summary -> {self._real_relabel_summary_path}",
            flush=True,
        )

    def _init_policy_trace(self, cfg: DictConfig) -> None:
        self._policy_trace_enabled = bool(
            OmegaConf.select(cfg, "eval.trace_policy_debug", default=False)
        )
        self._policy_trace_limit = int(
            OmegaConf.select(cfg, "eval.trace_policy_debug_limit", default=64)
        )
        self._policy_trace_count = 0
        self._policy_trace_dir = os.path.join(self.output_dir, "policy_trace_arrays")
        self._policy_trace_path = os.path.join(self.output_dir, "policy_trace.jsonl")
        if self._policy_trace_enabled and self.distributed.is_main_process:
            os.makedirs(self._policy_trace_dir, exist_ok=True)
            with open(self._policy_trace_path, "w"):
                pass

    _to_numpy_array = staticmethod(_eh.to_numpy_array)

    _array_summary = staticmethod(_eh.array_summary)

    def _write_policy_trace(
        self,
        *,
        source: str,
        state: np.ndarray,
        action_chunk_raw: np.ndarray,
        action_chunk_env: np.ndarray,
        action_hidden: Any | None = None,
        wm_style_action_hidden: Any | None = None,
        live_action_hidden: Any | None = None,
        recon_action_hidden: Any | None = None,
        obs_embedding: Any | None = None,
        actor_input: Any | None = None,
        rssm_latent: Any | None = None,
        input_ids: Any | None = None,
    ) -> None:
        if not bool(getattr(self, "_policy_trace_enabled", False)):
            return
        index = int(getattr(self, "_policy_trace_count", 0))
        if index >= int(getattr(self, "_policy_trace_limit", 64)):
            return

        arrays: dict[str, np.ndarray] = {
            "state": np.asarray(state, dtype=np.float32).reshape(-1),
            "action_chunk_raw": np.asarray(action_chunk_raw, dtype=np.float32),
            "action_chunk_env": np.asarray(action_chunk_env, dtype=np.float32),
        }
        optional_arrays = {
            "action_hidden": self._to_numpy_array(action_hidden),
            "wm_style_action_hidden": self._to_numpy_array(wm_style_action_hidden),
            "live_action_hidden": self._to_numpy_array(live_action_hidden),
            "recon_action_hidden": self._to_numpy_array(recon_action_hidden),
            "obs_embedding": self._to_numpy_array(obs_embedding),
            "actor_input": self._to_numpy_array(actor_input),
            "input_ids": self._to_numpy_array(input_ids),
        }
        if rssm_latent is not None:
            for attr in ("deter", "stoch", "logits", "mean", "std", "h"):
                if hasattr(rssm_latent, attr):
                    optional_arrays[f"rssm_{attr}"] = self._to_numpy_array(
                        getattr(rssm_latent, attr)
                    )
        for key, value in optional_arrays.items():
            if value is not None:
                arrays[key] = np.asarray(value, dtype=np.float32)

        array_path = os.path.join(
            self._policy_trace_dir, f"step_{index:06d}_{source}.npz"
        )
        np.savez_compressed(array_path, **arrays)
        context = dict(getattr(self, "_libero_current_eval_context", {}) or {})
        raw_chunk = arrays["action_chunk_raw"].reshape(
            -1, arrays["action_chunk_raw"].shape[-1]
        )
        env_chunk = arrays["action_chunk_env"].reshape(
            -1, arrays["action_chunk_env"].shape[-1]
        )
        record = {
            "index": index,
            "source": str(source),
            "context": context,
            "array_path": array_path,
            "state": arrays["state"].tolist(),
            "first_action_raw": raw_chunk[0].tolist(),
            "first_action_env": env_chunk[0].tolist(),
            "summaries": {
                key: self._array_summary(value) for key, value in arrays.items()
            },
        }
        if self.distributed.is_main_process:
            with open(self._policy_trace_path, "a") as f:
                f.write(json.dumps(record) + "\n")
        self._policy_trace_count = index + 1
