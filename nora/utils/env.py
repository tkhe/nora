# Copyright (c) Facebook, Inc. and its affiliates.

import importlib
import importlib.util
import logging
import os
import random
import sys
from datetime import datetime
from typing import Optional

import numpy as np
import torch

__all__ = ["seed_all_rng"]

TORCH_VERSION = tuple(int(x) for x in torch.__version__.split(".")[:2])
"""
PyTorch version as a tuple of 2 ints. Useful for comparison.
"""


def seed_all_rng(seed: Optional[int] = None):
    """
    Set the random seed for the RNG in torch, numpy and python.

    Args:
        seed (int): if None, will use a strong random seed.
    """
    if seed is None:
        seed = os.getpid() + int(datetime.now().strftime("%S%f")) + int.from_bytes(os.urandom(2), "big")

        logger = logging.getLogger(__name__)
        logger.info(f"Using a generated random seed {seed}")

    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(str(seed))
    os.environ["PYTHONHASHSEED"] = str(seed)


# from https://stackoverflow.com/questions/67631/how-to-import-a-module-given-the-full-path
def _import_file(module_name, file_path, make_importable=False):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if make_importable:
        sys.modules[module_name] = module
    return module


def _configure_libraries():
    """
    Configurations for some libraries.
    """
    # An environment option to disable `import cv2` globally,
    # in case it leads to negative performance impact
    disable_cv2 = int(os.environ.get("NORA_DISABLE_CV2", False))
    if disable_cv2:
        sys.modules["cv2"] = None
    else:
        # Disable opencl in opencv since its interaction with cuda often has negative effects
        # This envvar is supported after OpenCV 3.4.0
        os.environ["OPENCV_OPENCL_RUNTIME"] = "disabled"
        try:
            import cv2

            if int(cv2.__version__.split(".")[0]) >= 3:
                cv2.ocl.setUseOpenCL(False)
        except ModuleNotFoundError:
            # Other types of ImportError, if happened, should not be ignored.
            # Because a failed opencv import could mess up address space
            # https://github.com/skvark/opencv-python/issues/381
            pass

    def get_version(module, digit=2):
        return tuple(map(int, module.__version__.split(".")[:digit]))

    assert get_version(torch) >= (1, 8), "Requires torch>=1.8"


_ENV_SETUP_DONE = False


def setup_environment():
    """Perform environment setup work. The default setup is a no-op, but this
    function allows the user to specify a Python source file or a module in
    the $NORA_ENV_MODULE environment variable, that performs
    custom setup work that may be necessary to their computing environment.
    """
    global _ENV_SETUP_DONE

    if _ENV_SETUP_DONE:
        return

    _ENV_SETUP_DONE = True

    _configure_libraries()

    custom_module_path = os.environ.get("NORA_ENV_MODULE")

    if custom_module_path:
        setup_custom_environment(custom_module_path)
    else:
        # The default setup is a no-op
        pass


def setup_custom_environment(custom_module):
    """
    Load custom environment setup by importing a Python source file or a
    module, and run the setup function.
    """
    if custom_module.endswith(".py"):
        module = _import_file("nora.utils.env.custom_module", custom_module)
    else:
        module = importlib.import_module(custom_module)

    assert hasattr(module, "setup_environment") and callable(module.setup_environment), f"Custom environment module defined in {custom_module} does not have the required callable attribute 'setup_environment'."

    module.setup_environment()
