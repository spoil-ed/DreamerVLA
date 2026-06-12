import json
import os
import re
import warnings
from argparse import ArgumentParser
from pathlib import Path


def find_sub_records(directory: str):
    pattern = re.compile(r"\d+-of-\d+-record\.json(l)?")

    sub_record_files = [f for f in os.listdir(directory) if pattern.match(f)]
    sorted_files = sorted(
        sub_record_files, key=lambda filename: int(filename.split("-of")[0])
    )
    return sorted_files


def concat_records(sub_record_dir: str | Path, save_path: str | Path) -> None:
    sub_record_dir = Path(sub_record_dir)
    save_path = Path(save_path)
    l_sub_records = find_sub_records(str(sub_record_dir))

    print(f"find {len(l_sub_records)} sub-records in {sub_record_dir}")
    print(str(l_sub_records) + "\n\n")

    complete_record_by_id = {}
    complete_record_without_id = []
    for sub_record in l_sub_records:
        with open(sub_record_dir / sub_record) as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                try:
                    l_item = json.loads(line)
                    record_id = l_item.get("id")
                    if record_id is None:
                        complete_record_without_id.append(l_item)
                    else:
                        complete_record_by_id[int(record_id)] = l_item
                except Exception:
                    if i == len(lines) - 1:
                        print(
                            f"{sub_record} seems still writing, skip last incomplete record"
                        )
                    else:
                        warnings.warn(f"read line failed: {line}", stacklevel=2)

    complete_record = complete_record_without_id + [
        complete_record_by_id[k] for k in sorted(complete_record_by_id)
    ]
    duplicate_count = sum(
        len(open(sub_record_dir / f).readlines()) for f in l_sub_records
    ) - len(complete_record)
    if duplicate_count > 0:
        print(f"deduplicated {duplicate_count} records by id")

    with open(save_path, "w") as f:
        json.dump(complete_record, f)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--sub_record_dir",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
    )
    args = parser.parse_args()

    concat_records(args.sub_record_dir, args.save_path)
