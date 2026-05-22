from __future__ import annotations

from pathlib import Path


def test_docs_and_smoke_script_do_not_point_at_removed_entrypoints() -> None:
    project_root = Path(__file__).resolve().parents[1]

    readme = (project_root / "README.md").read_text(encoding="utf-8")
    smoke_script = (project_root / "scripts" / "preprocess" / "smoke_pretokenize.sh").read_text(
        encoding="utf-8"
    )
    dreamervla_data_script = (project_root / "scripts" / "prepare_dreamervla_data.sh").read_text(
        encoding="utf-8"
    )

    assert "eval_wm.sh" not in readme
    assert "pretokenize_sft_wm_vla_smoke" not in smoke_script
    assert "prepare_latent_data.sh" not in dreamervla_data_script
    assert (project_root / "LIBERO" / "benchmark_scripts" / "download_libero_datasets.py").is_file()
