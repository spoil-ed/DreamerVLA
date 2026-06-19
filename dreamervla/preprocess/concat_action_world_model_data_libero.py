# ruff: noqa: E402
import json
from pathlib import Path

from dreamervla.utils.hydra_config import script_namespace


def read_json_file(file_path: Path):
    try:
        with open(file_path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: File not found at {file_path}. Skipping.")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {file_path}. Skipping.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while reading {file_path}: {e}")
        return None


def write_json_file(data, file_path: Path):
    file_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
        print(f"Successfully wrote concatenated data to {file_path}")
    except Exception as e:
        print(f"Error: Could not write data to {file_path}: {e}")


def concatenate_records(input_file_paths: list[Path]):
    all_records = []
    for file_path in input_file_paths:
        content = read_json_file(file_path)
        if content is not None:
            if isinstance(content, list):
                all_records.extend(content)
            else:
                print(
                    f"Warning: Content of {file_path} is not a list. Appending it as a single item."
                )
                all_records.append(content)
    return all_records


def main(args):
    base_processed_data_dir = Path(args.processed_data_root)
    local_processed_data_dir = base_processed_data_dir / "tokens"
    concat_output_dir = base_processed_data_dir / "concate_tokens"

    # Use command line arguments for patterns
    SOURCE_DIR_PATTERNS = args.source_dir_patterns
    ALL_PATTERNS = args.all_patterns
    RECORD_FILENAME = "record.json"

    SPLITS = ["train", "val_ind", "val_ood"]
    concat_output_dir.mkdir(parents=True, exist_ok=True)
    all_inputs = []

    for split in SPLITS:
        print(f"\n--- Processing split: {split} ---")
        input_paths_for_split = []

        for pattern in SOURCE_DIR_PATTERNS:
            source_subdir_name = pattern.format(split)
            print(source_subdir_name)
            input_path = local_processed_data_dir / source_subdir_name / RECORD_FILENAME
            input_paths_for_split.append(input_path)
            all_inputs.append(input_path)
            print(f"  Looking for input: {input_path}")

        # base_output_name_from_pattern = SOURCE_DIR_PATTERNS[1].format(split)

        # output_filename_base = base_output_name_from_pattern[:-4] + f'_a2i_{args.resolution}'

        # output_filename = f"{output_filename_base}.json"
        # output_file_path = CONCAT_OUTPUT_DIR / output_filename

        # concatenated_data = concatenate_records(input_paths_for_split)

        # if concatenated_data:
        #     write_json_file(concatenated_data, output_file_path)
        # else:
        #     print(f"No valid data found to concatenate for split {split}. Output file {output_file_path} not created.")

    output_filename = f"{ALL_PATTERNS}.json"
    output_file_path = concat_output_dir / output_filename

    concatenated_data = concatenate_records(all_inputs)

    if concatenated_data:
        write_json_file(concatenated_data, output_file_path)
    else:
        print(
            f"No valid data found to concatenate. Output file {output_file_path} not created."
        )


if __name__ == "__main__":
    args = script_namespace("concat_action_world_model_data_libero")
    main(args)
