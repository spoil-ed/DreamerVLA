import os
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from argparse import ArgumentParser
import json
import math
import pickle
import copy

from src.preprocess.convertsation import Conversation
from src.preprocess.item_processor import FlexARItemProcessor_Action_State
from src.preprocess.paths import DEFAULT_TOKENIZER_PATH


class ItemProcessor(FlexARItemProcessor_Action_State):
    def __init__(
        self,
        tokenizer=str(DEFAULT_TOKENIZER_PATH),
        conv_template=Conversation,
        target_size=512,
    ):
        super().__init__(tokenizer, conv_template, target_size)
        print(self.crop_size_list)

    def process_item(self, raw_item, training_mode=False, out_flatten=True):

        # Add custom codes here to convert raw_item to the standard format
        # The standard format contains the "conversations" and "image" keys

        # ********* <start>  Add your custom codes here *******

        # *********  <end>   Add your custom codes here *******

        conversations = copy.deepcopy(raw_item["conversations"])
        if not conversations:
            conversations = [{"from": "human", "value": ""}, {"from": "gpt", "value": ""}]

        task_name = str(raw_item.get("task_name", "")).strip()
        next_obs = raw_item.get("next_obs", {})
        next_obs_images = []
        next_obs_states = []
        if isinstance(next_obs, dict):
            next_obs_images = list(next_obs.get("image", []) or [])
            next_obs_states = list(next_obs.get("state", []) or [])

        extra_prefix = ""
        if task_name:
            extra_prefix += f"Task name: {task_name}. "
        if next_obs_images:
            extra_prefix += "<|image|>" * len(next_obs_images)
        if next_obs_states:
            extra_prefix += "<|state|>" * len(next_obs_states)

        if extra_prefix:
            conversations[0]["value"] = extra_prefix + str(conversations[0].get("value", ""))

        item = {
            "conversations": conversations,
            "image": list(raw_item["image"]) + next_obs_images,
            "action": raw_item["action"],
            "state": list(raw_item["state"]) + next_obs_states,
        }

        return super(ItemProcessor, self).process_item(item, training_mode, out_flatten)


# ─────────────────────────────────────────────────────────────────────────────
# next_obs derivation from image/state file paths.
#
# The upstream conv JSON may leave ``next_obs`` empty (older generator) but the
# filesystem layout encodes the frame index in the filename:
#     .../imgs_third_view/image_{N}.png
#     .../imgs_wrist/image_{N}.png
#     .../eef_gripper_state/eef_gripper_state_{N}.npy
# After executing a length-H action chunk starting at frame N the next
# observation is frame N+H. We bump the index in each path and keep the file
# only when it exists on disk (so trajectory-end samples correctly fall back
# to empty next_obs).
# ─────────────────────────────────────────────────────────────────────────────


_IMG_INDEX_RE = re.compile(r"(?P<stem>.*/image_)(?P<idx>\d+)(?P<ext>\.\w+)$")
_STATE_INDEX_RE = re.compile(r"(?P<stem>.*/eef_gripper_state_)(?P<idx>\d+)(?P<ext>\.\w+)$")


def _bump_indexed_path(path: str, offset: int, regex: re.Pattern) -> str | None:
    match = regex.match(path)
    if match is None:
        return None
    new_idx = int(match.group("idx")) + int(offset)
    if new_idx < 0:
        return None
    candidate = f"{match.group('stem')}{new_idx}{match.group('ext')}"
    return candidate if os.path.isfile(candidate) else None


def derive_next_obs_from_paths(
    raw_item: dict,
    image_views_per_frame: int | None = None,
) -> dict:
    """Derive next_obs and the effective horizon by bumping frame indices.

    Behaviour:
      * First tries the full action-chunk horizon (H = len(actions)).
      * If image_{N+H} is missing (end-of-trajectory), walks H down step-by-step
        and picks the **largest h ≤ H** for which image_{N+h} exists on disk.
        The returned next_obs is then clamped to frame N+h and the WM side is
        expected to mask out action positions h..H-1 so the "unused" tail
        behaves like no-motion padding.
      * If even h=1 has no file, returns an empty next_obs (truly dead sample).

    Returned dict:
        {
            "image":              list[str],   # paths at frame N+h (empty when h=0)
            "state":              list[str],   # state path at frame N+h if any
            "effective_horizon":  int,         # h in [0, H]; H means full rollout
            "full_horizon":       int,         # H = len(actions); for reference
        }
    """
    obs_images = list(raw_item.get("image", []) or [])
    obs_states = list(raw_item.get("state", []) or [])
    actions = list(raw_item.get("action", []) or [])
    full_horizon = len(actions)
    empty_result = {
        "image": [],
        "state": [],
        "effective_horizon": 0,
        "full_horizon": full_horizon,
    }
    if full_horizon <= 0 or not obs_images:
        return empty_result

    if image_views_per_frame is None or image_views_per_frame <= 0:
        image_views_per_frame = len(obs_images)
    image_views_per_frame = min(image_views_per_frame, len(obs_images))
    current_frame_paths = obs_images[-image_views_per_frame:]
    current_state_path = obs_states[-1] if obs_states else None

    for horizon in range(full_horizon, 0, -1):
        next_image_paths: list[str] = []
        ok = True
        for p in current_frame_paths:
            bumped = _bump_indexed_path(p, horizon, _IMG_INDEX_RE)
            if bumped is None:
                ok = False
                break
            next_image_paths.append(bumped)
        if not ok:
            continue

        next_state_paths: list[str] = []
        if current_state_path is not None:
            bumped_state = _bump_indexed_path(current_state_path, horizon, _STATE_INDEX_RE)
            if bumped_state is not None:
                next_state_paths.append(bumped_state)

        return {
            "image": next_image_paths,
            "state": next_state_paths,
            "effective_horizon": horizon,
            "full_horizon": full_horizon,
        }

    return empty_result


def ensure_next_obs(raw_item: dict, image_views_per_frame: int | None = None) -> dict:
    """Return a next_obs dict (image + state + effective_horizon + full_horizon).

    Uses the upstream ``raw_item.next_obs`` when it already carries image paths,
    otherwise derives one from paths with the variable-horizon strategy above.
    A derived-from-paths result is always accompanied by ``effective_horizon``;
    for upstream-supplied dicts we assume the full horizon unless they say
    otherwise.
    """
    existing = raw_item.get("next_obs")
    if isinstance(existing, dict):
        has_images = bool(list(existing.get("image", []) or []))
        has_states = bool(list(existing.get("state", []) or []))
        if has_images or has_states:
            actions = list(raw_item.get("action", []) or [])
            merged = dict(existing)
            merged.setdefault("effective_horizon", len(actions))
            merged.setdefault("full_horizon", len(actions))
            return merged
    return derive_next_obs_from_paths(raw_item, image_views_per_frame=image_views_per_frame)


def build_wm_action_mask(effective_horizon: int, full_horizon: int) -> list[bool]:
    """Per-step mask: first ``effective_horizon`` positions are real, rest padded."""
    effective_horizon = max(int(effective_horizon), 0)
    full_horizon = max(int(full_horizon), 0)
    effective_horizon = min(effective_horizon, full_horizon)
    return [True] * effective_horizon + [False] * (full_horizon - effective_horizon)


def _build_wm_token_sequences(raw_item: dict, item_processor: ItemProcessor) -> tuple[list[int], list[int]]:
    task_name = str(raw_item.get("task_name", "")).strip()
    task_prefix = f"Task name: {task_name}. " if task_name else ""
    obs_images = list(raw_item.get("image", []) or [])
    obs_states = list(raw_item.get("state", []) or [])
    next_obs = raw_item.get("next_obs", {})
    if not isinstance(next_obs, dict):
        next_obs = {}
    next_images = list(next_obs.get("image", []) or [])
    next_states = list(next_obs.get("state", []) or [])

    obs_human = task_prefix + ("<|state|>" * len(obs_states)) + ("<|image|>" * len(obs_images))
    next_human = task_prefix + ("<|state|>" * len(next_states)) + ("<|image|>" * len(next_images))
    obs_item = {
        "conversations": [
            {"from": "human", "value": obs_human},
            {"from": "gpt", "value": ""},
        ],
        "image": obs_images,
        "action": [],
        "state": obs_states,
    }
    next_item = {
        "conversations": [
            {"from": "human", "value": next_human},
            {"from": "gpt", "value": ""},
        ],
        "image": next_images,
        "action": [],
        "state": next_states,
    }
    obs_input_ids = list(item_processor.process_item(obs_item, training_mode=False))
    next_input_ids = list(item_processor.process_item(next_item, training_mode=False))
    return obs_input_ids, next_input_ids


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument(
        "--splits",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--in_filename",
        type=str,
    )
    parser.add_argument(
        "--out_dir",
        type=str,
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=str(DEFAULT_TOKENIZER_PATH),
    )
    parser.add_argument("--target_size", type=int, default=512)
    parser.add_argument(
        "--image_views_per_frame",
        type=int,
        default=2,
        help=(
            "Number of image views per frame (e.g. third_view + wrist = 2). "
            "Used to pick the *current* frame's view paths out of a history "
            "observation when deriving next_obs. Set <=0 to disable and treat "
            "the full image list as one frame."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-tokenize samples even when the output pkl already exists.",
    )
    args = parser.parse_args()

    item_processor = ItemProcessor(target_size=args.target_size, tokenizer=args.tokenizer)

    with open(args.in_filename) as f:
        ori_contents = json.load(f)

    num = len(ori_contents)

    splits = args.splits
    rank = args.rank
    output_dir = args.out_dir
    save_dir = os.path.join(output_dir, "files")
    os.makedirs(save_dir, exist_ok=True)

    num_per_rank = math.ceil(num / splits)

    rank_start_idx = num_per_rank * rank
    end_idx = min(num_per_rank * (rank + 1), len(ori_contents))
    progress_path = os.path.join(output_dir, f"{rank}-of-{splits}-progress.txt")
    try:
        with open(progress_path, "r") as f:
            progress = f.read().strip()
        if progress == "finished":
            print(f"rank {rank}: progress is finished; scan existing pkl only")
        else:
            print(f"rank {rank}: previous progress={progress}; scan existing pkl for holes")
    except Exception:
        print(f"rank {rank}: no progress file; scan existing pkl from {rank_start_idx}")

    derived_count = 0
    skipped_existing = 0
    for i in range(rank_start_idx, end_idx):
        if i % 10 == 0:
            print(f"{i}/{end_idx}  (next_obs derived so far: {derived_count})")

        record = None
        pkl_path = os.path.join(save_dir, f"{i}.pkl")
        if os.path.exists(pkl_path) and not args.overwrite:
            skipped_existing += 1
            continue

        try:
            raw_item = ori_contents[i]

            # Patch raw_item.next_obs in place before any downstream consumer
            # sees it, so both the VLA item_processor and the WM token builder
            # agree on the same observation/next_obs pair.
            original_next_obs = raw_item.get("next_obs")
            patched_next_obs = ensure_next_obs(
                raw_item,
                image_views_per_frame=args.image_views_per_frame,
            )
            if (not isinstance(original_next_obs, dict)
                or not (list((original_next_obs or {}).get("image", []) or [])
                        or list((original_next_obs or {}).get("state", []) or []))):
                if patched_next_obs.get("image") or patched_next_obs.get("state"):
                    derived_count += 1
            raw_item["next_obs"] = patched_next_obs

            full_horizon = int(patched_next_obs.get("full_horizon", len(raw_item.get("action", []) or [])))
            effective_horizon = int(patched_next_obs.get("effective_horizon", full_horizon))
            if effective_horizon <= 0:
                # truly end-of-trajectory; no future frame reachable -> skip.
                with open(os.path.join(output_dir, f"{rank}-of-{splits}-progress.txt"), "w") as f:
                    if i == end_idx - 1:
                        f.write("finished")
                    else:
                        f.write(f"{i}")
                continue
            wm_action_mask = build_wm_action_mask(effective_horizon, full_horizon)

            tokens, labels = item_processor.process_item(raw_item, training_mode=True)
            wm_obs_input_ids, wm_next_obs_input_ids = _build_wm_token_sequences(raw_item, item_processor)
            meta = {
                "task_name": raw_item.get("task_name"),
                "task_text": raw_item.get("task_text"),
                "prompt_text": raw_item.get("prompt_text"),
                "num_images": len(raw_item.get("image", [])),
                "num_actions": len(raw_item.get("action", [])),
                "num_states": len(raw_item.get("state", [])),
                "num_next_images": len(patched_next_obs.get("image", [])),
                "num_next_states": len(patched_next_obs.get("state", [])),
                "reward": raw_item.get("reward"),
                "next_obs": patched_next_obs,
                "next_obs_derived": original_next_obs != patched_next_obs,
                "effective_horizon": effective_horizon,
                "full_horizon": full_horizon,
                "is_eot_padded": effective_horizon < full_horizon,
            }
            new_item = {
                "token": tokens,
                "label": labels,
                "id": i,
                "meta": meta,
                "task_name": raw_item.get("task_name"),
                "image": raw_item.get("image", []),
                "action": raw_item.get("action", []),
                "state": raw_item.get("state", []),
                "reward": raw_item.get("reward"),
                "next_obs": patched_next_obs,
                "wm_obs_input_ids": wm_obs_input_ids,
                "wm_next_obs_input_ids": wm_next_obs_input_ids,
                "wm_action_mask": wm_action_mask,
                "effective_horizon": effective_horizon,
                "full_horizon": full_horizon,
            }
            with open(pkl_path, "wb") as f:
                pickle.dump(new_item, f)

            record = {
                "file": pkl_path,
                "len": len(tokens),
                "id": i,
                "meta": meta,
                "reward": raw_item.get("reward"),
                "next_obs": patched_next_obs,
            }

        except Exception:
            from traceback import format_exc

            print(f"item {i} error: \n{ori_contents[i]}")
            print(format_exc())

        if record is not None:
            with open(os.path.join(output_dir, f"{rank}-of-{splits}-record.jsonl"), "a") as f:
                record_str = json.dumps(record) + "\n"
                f.write(record_str)

        with open(os.path.join(output_dir, f"{rank}-of-{splits}-progress.txt"), "w") as f:
            if i == end_idx - 1:
                f.write("finished")
            else:
                f.write(f"{i}")

    missing_after = [
        i
        for i in range(rank_start_idx, end_idx)
        if not os.path.exists(os.path.join(save_dir, f"{i}.pkl"))
    ]
    with open(progress_path, "w") as f:
        f.write("finished" if not missing_after else str(max(rank_start_idx - 1, missing_after[0] - 1)))

    print(
        f"rank {rank}: done. skipped existing {skipped_existing}; "
        f"remaining missing {len(missing_after)}; derived next_obs for {derived_count} samples."
    )
