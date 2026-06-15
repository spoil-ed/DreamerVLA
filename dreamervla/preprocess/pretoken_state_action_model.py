# ruff: noqa: E402
import argparse  # 导入 argparse 模块
import os
import subprocess
import sys
from multiprocessing import Process
from pathlib import Path

from dreamervla.preprocess.paths import (
    DEFAULT_CONVS_DIR,
    DEFAULT_TOKENIZER_PATH,
    DEFAULT_TOKENS_DIR,
)


def run_script(
    rank,
    all_ranks,
    resolution,
    in_filename_path,
    out_dir,
    with_state,
    tokenizer_path,
    image_views_per_frame,
    gpu_pool,
    overwrite,
):  # 添加 task 参数
    # gpu_pool is the list of *physical* GPU indices the launcher was told to
    # use (defaults to [0,1,2,3] for backwards compatibility). Each worker
    # pins itself to one GPU via round-robin over that pool.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_pool[rank % len(gpu_pool)])
    print(
        f"Starting running on rank={rank}, CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}."
    )

    module_name = (
        "dreamervla.preprocess.pre_tokenize_action_state_local"
        if with_state
        else "dreamervla.preprocess.pre_tokenize_action_local"
    )
    command = [
        sys.executable,
        "-u",
        "-m",
        module_name,
        f"--splits={all_ranks}",
        f"--rank={rank}",
        "--in_filename",
        str(in_filename_path),
        "--out_dir",
        str(out_dir),
        "--tokenizer",
        str(tokenizer_path),
        "--target_size",
        str(resolution),
        "--image_views_per_frame",
        str(int(image_views_per_frame)),
    ]
    if overwrite:
        command.append("--overwrite")
    subprocess.run(command, check=True)


if __name__ == "__main__":
    # 1. 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(
        description="Run parallel data processing scripts with a customizable spatial task."
    )

    # 2. 添加命令行参数
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="dataset name (e.g., 'spatial', 'object', 'goal', '10').",
    )
    parser.add_argument(
        "--resolution", type=int, required=True, help="resolution (e.g., 256, 512)."
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=str(DEFAULT_TOKENIZER_PATH),
        help="tokenizer path inside DreamerVLA/data/checkpoints",
    )
    parser.add_argument(
        "--in_filename_dir",
        type=str,
        default=str(DEFAULT_CONVS_DIR),
        help="directory containing generated conversation json files",
    )
    parser.add_argument(
        "--out_root",
        type=str,
        default=str(DEFAULT_TOKENS_DIR),
        help="directory where tokenized outputs will be written",
    )
    parser.add_argument(
        "--his",
        "-H",
        type=int,
        default=2,
        help="The number of historical image frames to include in each conversation (for observation history).",
    )
    parser.add_argument(
        "--len_action",
        "-L",
        type=int,
        default=5,
        help="The number of future action steps to predict.",
    )
    parser.add_argument(
        "--with_state", action="store_true", help="If True, with state."
    )
    parser.add_argument(
        "--img_names",
        nargs="+",
        default=["imgs_third_view"],
        choices=["imgs_wrist", "imgs_third_view"],
        help="List of image names to include (imgs_wrist and/or imgs_third_view)",
    )
    parser.add_argument(
        "--num_procs",
        type=int,
        default=32,
        help="Number of worker processes for tokenization.",
    )
    parser.add_argument(
        "--gpu_devices",
        type=str,
        default=os.environ.get("PREPROCESS_GPU_DEVICES", "0,1,2,3"),
        help=(
            "Comma-separated physical GPU indices to pin workers to, round-robin. "
            "Defaults to env var PREPROCESS_GPU_DEVICES, else '0,1,2,3'. "
            "Use e.g. '4,5,6,7' when GPUs 0-3 are busy with training."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Forward --overwrite to workers and re-tokenize existing pkl files.",
    )

    # 3. 解析命令行参数
    args = parser.parse_args()

    data_type = ["val_ind", "val_ood", "train"]

    img_item = "_".join([item.replace("imgs_", "") for item in args.img_names])
    state_item = "w_state" if args.with_state else "wo_state"

    in_filename_dir = Path(args.in_filename_dir)
    out_root = Path(args.out_root)

    # Each frame contributes one image path per view (e.g. third_view + wrist = 2).
    # The launcher forwards this count to the workers so they can slice the
    # *current* frame out of a history observation when deriving next_obs.
    image_views_per_frame = max(len(args.img_names), 1)

    gpu_pool = [int(x) for x in str(args.gpu_devices).split(",") if x.strip() != ""]
    if not gpu_pool:
        gpu_pool = [0, 1, 2, 3]
    print(f"Worker GPU pool: {gpu_pool}")

    for data_t in data_type:
        in_filename_path = (
            in_filename_dir
            / f"libero_{args.task}_his_{args.his}_{data_t}_{img_item}_{state_item}_{args.len_action}_{args.resolution}.json"
        )
        out_dir = (
            out_root
            / f"libero_{args.task}_his_{args.his}_{data_t}_{img_item}_{state_item}_{args.len_action}_{args.resolution}"
        )

        processes = []
        all_ranks = args.num_procs
        for i in range(all_ranks):
            # 将解析到的 task 传递给 run_script 函数
            p = Process(
                target=run_script,
                args=(
                    i,
                    all_ranks,
                    args.resolution,
                    in_filename_path,
                    out_dir,
                    args.with_state,
                    args.tokenizer_path,
                    image_views_per_frame,
                    gpu_pool,
                    args.overwrite,
                ),
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
            if p.exitcode != 0:
                raise SystemExit(f"worker process failed with exit code {p.exitcode}")
