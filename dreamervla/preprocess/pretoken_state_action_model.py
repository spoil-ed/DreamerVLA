# ruff: noqa: E402
import os
import subprocess
import sys
from multiprocessing import Process
from pathlib import Path

from dreamervla.utils.hydra_config import script_namespace


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
    args = script_namespace("pretoken_state_action_model")

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
