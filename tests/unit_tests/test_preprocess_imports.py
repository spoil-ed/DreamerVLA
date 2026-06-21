from __future__ import annotations

import importlib


def test_preprocess_item_processor_imports_from_preprocess_helpers() -> None:
    module = importlib.import_module("dreamervla.preprocess.item_processor")

    assert hasattr(module, "FlexARItemProcessorAction")
    assert hasattr(module, "FlexARItemProcessorActionState")
    assert module.FlexARItemProcessor_Action is module.FlexARItemProcessorAction
    assert module.FlexARItemProcessor_Action_State is module.FlexARItemProcessorActionState
