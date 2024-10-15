# Copyright (c) Facebook, Inc. and its affiliates.

import glob
import os
import shutil
from typing import List

import torch
from torch.utils.cpp_extension import CUDAExtension
from torch.utils.cpp_extension import CUDA_HOME
from torch.utils.cpp_extension import CppExtension
from setuptools import find_packages
from setuptools import setup

torch_ver = [int(x) for x in torch.__version__.split(".")[:2]]
assert torch_ver >= [1, 8], "Requires PyTorch >= 1.8"


def get_version():
    init_py_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "nora", "__init__.py")
    init_py = open(init_py_path, "r").readlines()
    version_line = [l.strip() for l in init_py if l.startswith("__version__")][0]
    version = version_line.split("=")[-1].strip().strip("'\"")

    # The following is used to build release packages.
    # Users should never use it.
    suffix = os.getenv("NORA_VERSION_SUFFIX", "")
    version = version + suffix
    if os.getenv("BUILD_NIGHTLY", "0") == "1":
        from datetime import datetime

        date_str = datetime.today().strftime("%y%m%d")
        version = version + ".dev" + date_str

        new_init_py = [l for l in init_py if not l.startswith("__version__")]
        new_init_py.append(f'__version__ = "{version}"\n')
        with open(init_py_path, "w") as f:
            f.write("".join(new_init_py))
    return version


def get_extensions():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(this_dir, "nora", "ops", "csrc")

    main_source = os.path.join(extensions_dir, "vision.cpp")
    sources = glob.glob(os.path.join(extensions_dir, "**", "*.cpp"))

    from torch.utils.cpp_extension import ROCM_HOME

    is_rocm_pytorch = True if ((torch.version.hip is not None) and (ROCM_HOME is not None)) else False
    if is_rocm_pytorch:
        assert torch_ver >= [1, 8], "ROCM support requires PyTorch >= 1.8!"

    # common code between cuda and rocm platforms, for hipify version [1,0,0] and later.
    source_cuda = glob.glob(os.path.join(extensions_dir, "**", "*.cu")) + glob.glob(os.path.join(extensions_dir, "*.cu"))
    sources = [main_source] + sources

    extension = CppExtension

    extra_compile_args = {"cxx": []}
    define_macros = []

    if (torch.cuda.is_available() and ((CUDA_HOME is not None) or is_rocm_pytorch)) or os.getenv("FORCE_CUDA", "0") == "1":
        extension = CUDAExtension
        sources += source_cuda

        if not is_rocm_pytorch:
            define_macros += [("WITH_CUDA", None)]
            extra_compile_args["nvcc"] = [
                "-O3",
                "-DCUDA_HAS_FP16=1",
                "-D__CUDA_NO_HALF_OPERATORS__",
                "-D__CUDA_NO_HALF_CONVERSIONS__",
                "-D__CUDA_NO_HALF2_OPERATORS__",
            ]
        else:
            define_macros += [("WITH_HIP", None)]
            extra_compile_args["nvcc"] = []

        nvcc_flags_env = os.getenv("NVCC_FLAGS", "")
        if nvcc_flags_env != "":
            extra_compile_args["nvcc"].extend(nvcc_flags_env.split(" "))

        if torch_ver < [1, 7]:
            # supported by https://github.com/pytorch/pytorch/pull/43931
            CC = os.environ.get("CC", None)
            if CC is not None:
                extra_compile_args["nvcc"].append(f"-ccbin={CC}")

    include_dirs = [extensions_dir]

    ext_modules = [
        extension(
            "nora._C",
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]

    return ext_modules


def get_model_zoo_configs() -> List[str]:
    """
    Return a list of configs to include in package for model zoo. Copy over these configs inside
    nora/model_zoo.
    """

    # Use absolute paths while symlinking.
    source_configs_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "configs")
    destination = os.path.join(os.path.dirname(os.path.realpath(__file__)), "nora", "model_zoo", "configs")
    # Symlink the config directory inside package to have a cleaner pip install.

    # Remove stale symlink/directory from a previous build.
    if os.path.exists(source_configs_dir):
        if os.path.islink(destination):
            os.unlink(destination)
        elif os.path.isdir(destination):
            shutil.rmtree(destination)

    if not os.path.exists(destination):
        try:
            os.symlink(source_configs_dir, destination)
        except OSError:
            # Fall back to copying if symlink fails: ex. on Windows.
            shutil.copytree(source_configs_dir, destination)

    config_paths = glob.glob("configs/**/*.py", recursive=True)
    return config_paths


setup(
    name="nora",
    version=get_version(),
    author="tkhe",
    url="https://github.com/tkhe/nora",
    description="nora is an open source toolbox for visual recognition tasks.",
    packages=find_packages(exclude=("configs", "tests*")),
    package_data={"nora.model_zoo": get_model_zoo_configs()},
    python_requires=">=3.7",
    install_requires=[
        "Pillow>=7.1",
        "black",
        "cloudpickle",
        "dataclasses; python_version<'3.7'",
        "fairscale",
        "hydra-core>=1.1",
        "matplotlib",
        "omegaconf>=2.1,<2.4",
        "opencv-python",
        "packaging",
        "portalocker",
        "pycocotools>=2.0.2",
        "scipy>1.5.1",
        "tabulate",
        "tensorboard",
        "termcolor>=1.1",
        "tqdm>4.29.0",
    ],
    extras_require={
        # optional dependencies, required by some features
        "all": [
            "panopticapi @ https://github.com/cocodataset/panopticapi/archive/master.zip",
            "psutil",
            "pygments>=2.2",
            "shapely",
        ],
        # dev dependencies. Install them by `pip install 'nora[dev]'`
        "dev": [
            "black==22.3.0",
            "flake8==3.8.1",
            "flake8-bugbear",
            "flake8-comprehensions",
            "isort==4.3.21",
        ],
    },
    ext_modules=get_extensions(),
    cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension},
)
