import os
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

    try:
        with open(os.path.join(output_dir, f"{rank}-of-{splits}-progress.txt"), "r") as f:
            start_idx = int(f.read()) + 1
        print(f"resume from {start_idx}")
    except:
        start_idx = num_per_rank * rank
        print(f"start from {start_idx}")

    end_idx = min(num_per_rank * (rank + 1), len(ori_contents))
    for i in range(start_idx, end_idx):
        if i % 10 == 0:
            print(f"{i}/{end_idx}")

        record = None
        pkl_path = os.path.join(save_dir, f"{i}.pkl")
        try:
            tokens, labels = item_processor.process_item(ori_contents[i], training_mode=True)
            wm_obs_input_ids, wm_next_obs_input_ids = _build_wm_token_sequences(ori_contents[i], item_processor)
            meta = {
                "task_name": ori_contents[i].get("task_name"),
                "task_text": ori_contents[i].get("task_text"),
                "prompt_text": ori_contents[i].get("prompt_text"),
                "num_images": len(ori_contents[i].get("image", [])),
                "num_actions": len(ori_contents[i].get("action", [])),
                "num_states": len(ori_contents[i].get("state", [])),
                "reward": ori_contents[i].get("reward"),
                "next_obs": ori_contents[i].get("next_obs"),
            }
            new_item = {
                "token": tokens,
                "label": labels,
                "id": i,
                "meta": meta,
                "task_name": ori_contents[i].get("task_name"),
                "image": ori_contents[i].get("image", []),
                "action": ori_contents[i].get("action", []),
                "state": ori_contents[i].get("state", []),
                "reward": ori_contents[i].get("reward"),
                "next_obs": ori_contents[i].get("next_obs"),
                "wm_obs_input_ids": wm_obs_input_ids,
                "wm_next_obs_input_ids": wm_next_obs_input_ids,
            }
            with open(pkl_path, "wb") as f:
                pickle.dump(new_item, f)

            record = {
                "file": pkl_path,
                "len": len(tokens),
                "id": i,
                "meta": meta,
                "reward": ori_contents[i].get("reward"),
                "next_obs": ori_contents[i].get("next_obs"),
            }

        except Exception as e:
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
