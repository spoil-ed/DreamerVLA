import glob
import importlib
import logging
import os.path as osp

from .model import DreamerVLA


def _discover_model_filenames():
    model_folder = osp.dirname(osp.abspath(__file__))

    filenames = {
        osp.splitext(osp.basename(path))[0]
        for path in glob.glob(osp.join(model_folder, "*_model.py"))
    }

    base_model_path = osp.join(model_folder, "model.py")
    if osp.isfile(base_model_path):
        filenames.add("model")

    return sorted(filenames)


_model_modules = [
    importlib.import_module(f"{__name__}.{file_name}")
    for file_name in _discover_model_filenames()
]


def create_model(opt):
    """Create model.

    Args:
        opt (dict): Configuration. It contains:
            model_type (str): Model type.
    """
    model_type = opt["model_type"]
    model_cls = None

    for module in _model_modules:
        model_cls = getattr(module, model_type, None)
        if model_cls is not None:
            break

    if model_cls is None:
        raise ValueError(f"Model {model_type} is not found.")

    model = model_cls(opt)

    logger = logging.getLogger("base")
    logger.info(f"Model [{model.__class__.__name__}] is created.")
    return model


__all__ = ["DreamerVLA", "create_model"]
