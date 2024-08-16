# Copyright (c) Facebook, Inc. and its affiliates.

"""
This file contains components with some default boilerplate logic user may need
in training / testing. They will not work for everyone, but many users may find them useful.

The behavior of functions/classes in this file is subject to change,
since they are meant to represent the "common default behavior" people need in their projects.
"""

import argparse
import copy
import logging
import os
import sys
import weakref
from collections import OrderedDict
from typing import Optional

import torch
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel

from nora.checkpoint import Checkpointer
from nora.config import LazyConfig
from nora.config import instantiate
from nora.data import transforms as T
from nora.evaluation import DatasetEvaluator
from nora.evaluation import inference_on_dataset
from nora.evaluation import print_csv_format
from nora.evaluation import verify_results
from nora.utils import comm
from nora.utils.collect_env import collect_env_info
from nora.utils.env import seed_all_rng
from nora.utils.events import CommonMetricPrinter
from nora.utils.events import JSONWriter
from nora.utils.events import TensorboardXWriter
from nora.utils.file_io import PathManager
from nora.utils.logger import setup_logger
from nora.utils.precise_bn import get_bn_modules
from . import hooks
from .train_loop import AMPTrainer
from .train_loop import SimpleTrainer
from .train_loop import Trainer

__all__ = [
    "DefaultPredictor",
    "DefaultTrainer",
    "create_ddp_model",
    "default_argument_parser",
    "default_setup",
    "default_writers",
]


def create_ddp_model(model, *, fp16_compression=False, **kwargs):
    """
    Create a DistributedDataParallel model if there are >1 processes.

    Args:
        model: a torch.nn.Module
        fp16_compression: add fp16 compression hooks to the ddp object.
            See more at https://pytorch.org/docs/stable/ddp_comm_hooks.html#torch.distributed.algorithms.ddp_comm_hooks.default_hooks.fp16_compress_hook
        kwargs: other arguments of :module:`torch.nn.parallel.DistributedDataParallel`.
    """
    if comm.get_world_size() == 1:
        return model

    if "device_ids" not in kwargs:
        kwargs["device_ids"] = [comm.get_local_rank()]
    ddp = DistributedDataParallel(model, **kwargs)
    if fp16_compression:
        from torch.distributed.algorithms.ddp_comm_hooks import default as comm_hooks

        ddp.register_comm_hook(state=None, hook=comm_hooks.fp16_compress_hook)
    return ddp


def default_argument_parser(epilog=None):
    """
    Create a parser with some common arguments used by nora users.

    Args:
        epilog (str): epilog passed to ArgumentParser describing the usage.

    Returns:
        argparse.ArgumentParser:
    """
    parser = argparse.ArgumentParser(
        epilog=epilog
        or f"""
Examples:

Run on single machine:
    $ {sys.argv[0]} --num-gpus 8 --config-file cfg.py

Change some config options:
    $ {sys.argv[0]} --config-file cfg.py a=b c.d.e=f

Run on multiple machines:
    (machine0)$ {sys.argv[0]} --machine-rank 0 --num-machines 2 --dist-url <URL> [--other-flags]
    (machine1)$ {sys.argv[0]} --machine-rank 1 --num-machines 2 --dist-url <URL> [--other-flags]
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument("--resume", action="store_true", help="Whether to attempt to resume from the checkpoint directory.")
    parser.add_argument("--eval-only", action="store_true", help="perform evaluation only")
    parser.add_argument("--num-gpus", type=int, default=1, help="number of gpus *per machine*")
    parser.add_argument("--num-machines", type=int, default=1, help="total number of machines")
    parser.add_argument("--machine-rank", type=int, default=0, help="the rank of this machine (unique per machine)")

    # PyTorch still may leave orphan processes in multi-gpu training.
    # Therefore we use a deterministic way to obtain port,
    # so that users are aware of orphan processes by seeing the port occupied.
    port = 2 ** 15 + 2 ** 14 + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
    parser.add_argument(
        "--dist-url",
        default=f"tcp://127.0.0.1:{port}",
        help="initialization URL for pytorch distributed backend. See https://pytorch.org/docs/stable/distributed.html for details.",
    )
    parser.add_argument(
        "opts",
        help='Modify config options at the end of the command. For python-based LazyConfig, use "path.key=value".',
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


def _try_get_key(cfg, *keys, default=None):
    """
    Try select keys from cfg until the first key that exists. Otherwise return default.
    """
    for k in keys:
        none = object()
        p = OmegaConf.select(cfg, k, default=none)
        if p is not none:
            return p
    return default


def _highlight(code, filename):
    try:
        import pygments
    except ImportError:
        return code

    from pygments.lexers import Python3Lexer
    from pygments.lexers import YamlLexer
    from pygments.formatters import Terminal256Formatter

    lexer = Python3Lexer() if filename.endswith(".py") else YamlLexer()
    code = pygments.highlight(code, lexer, Terminal256Formatter(style="monokai"))
    return code


def default_setup(cfg, args):
    """
    Perform some basic common setups at the beginning of a job, including:

    1. Set up the nora logger
    2. Log basic information about environment, cmdline arguments, and config
    3. Backup the config to the output directory

    Args:
        cfg (omegaconf.DictConfig): the full config to be used
        args (argparse.NameSpace): the command line arguments to be logged
    """
    output_dir = _try_get_key(cfg, "output_dir", "train.output_dir")
    if comm.is_main_process() and output_dir:
        PathManager.mkdirs(output_dir)

    rank = comm.get_rank()
    logger = setup_logger(output_dir, distributed_rank=rank)

    logger.info(f"Rank of current process: {rank}. World size: {comm.get_world_size()}")
    logger.info("Environment info:\n" + collect_env_info())

    logger.info("Command line arguments: " + str(args))
    if hasattr(args, "config_file") and args.config_file != "":
        logger.info(f"Contents of args.config_file={args.config_file}:\n{_highlight(PathManager.open(args.config_file, 'r').read(), args.config_file)}")

    if comm.is_main_process() and output_dir:
        # Note: some of our scripts may expect the existence of
        # config.yaml in output directory
        path = os.path.join(output_dir, "config.yaml")
        LazyConfig.save(cfg, path)
        logger.info(f"Full config saved to {path}")

    # make sure each worker has a different, yet deterministic seed if specified
    seed = _try_get_key(cfg, "seed", "train.seed", default=-1)
    seed_all_rng(None if seed < 0 else seed + rank)

    # cudnn benchmark has large overhead. It shouldn't be used considering the small size of
    # typical validation set.
    if not (hasattr(args, "eval_only") and args.eval_only):
        torch.backends.cudnn.benchmark = _try_get_key(cfg, "cudnn_benchmark", "train.cudnn_benchmark", default=False)


def default_writers(output_dir: str, max_iter: Optional[int] = None):
    """
    Build a list of :class:`EventWriter` to be used.
    It now consists of a :class:`CommonMetricPrinter`,
    :class:`TensorboardXWriter` and :class:`JSONWriter`.

    Args:
        output_dir: directory to store JSON metrics and tensorboard events
        max_iter: the total number of iterations

    Returns:
        list[EventWriter]: a list of :class:`EventWriter` objects.
    """
    PathManager.mkdirs(output_dir)
    return [
        # It may not always print what you want to see, since it prints "common" metrics only.
        CommonMetricPrinter(max_iter),
        JSONWriter(os.path.join(output_dir, "metrics.json")),
        TensorboardXWriter(output_dir),
    ]


class DefaultPredictor:
    """
    Create a simple end-to-end predictor with the given config that runs on
    single device for a single input image.

    Compared to using the model directly, this class does the following additions:

    1. Load checkpoint from `cfg.train.init_checkpoint`.
    2. Always take BGR image as the input and apply conversion defined by `cfg.dataloader.test.mapper.image_format`.
    3. Apply augmentations defined by `cfg.dataloader.test.mapper.augmentations`.
    4. Take one input image and produce a single output, instead of a batch.

    This is meant for simple demo purposes, so it does the above steps automatically.
    This is not meant for benchmarks or running complicated inference logic.
    If you'd like to do anything more complicated, please refer to its source code as
    examples to build and use the model manually.

    Attributes:
        metadata (Metadata): the metadata of the underlying dataset, obtained from
            cfg.DATASETS.TEST.

    Examples:

        >>> pred = DefaultPredictor(cfg)
        >>> inputs = cv2.imread("input.jpg")
        >>> outputs = pred(inputs)
    """

    def __init__(self, cfg):
        self.cfg = copy.deepcopy(cfg)
        self.model = instantiate(self.cfg.model)
        self.model.eval()
        self.model.to(cfg.train.device)

        checkpointer = Checkpointer(self.model)
        checkpointer.load(cfg.train.init_checkpoint)

        self.augmentations = T.AugmentationList(instantiate(cfg.dataloader.test.mapper.augmentations))
        self.input_format = cfg.dataloader.test.mapper.image_format
        assert self.input_format in ["RGB", "BGR"], self.input_format

    def __call__(self, original_image):
        """
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).

        Returns:
            predictions (dict):
                the output of the model for one image only.
        """
        with torch.no_grad():
            if self.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                original_image = original_image[:, :, ::-1]

            height, width = original_image.shape[:2]
            aug_input = T.AugInput(image=original_image)
            transforms = self.augmentations(aug_input)
            image = aug_input.image
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            image = image.to(self.cfg.train.device)

            inputs = {"image": image, "height": height, "width": width}

            predictions = self.model([inputs])[0]
            return predictions


class DefaultTrainer(Trainer):
    """
    A trainer with default training logic. It does the following:

    1. Create a :class:`SimpleTrainer` using model, optimizer, dataloader
       defined by the given config. Create a LR scheduler defined by the config.
    2. Load the last checkpoint or `cfg.train.init_checkpoint`, if exists, when
       `resume_or_load` is called.
    3. Register a few common hooks defined by the config.

    It is created to simplify the **standard model training workflow** and reduce code boilerplate
    for users who only need the standard training workflow, with standard features.
    It means this class makes *many assumptions* about your training logic that
    may easily become invalid in a new research. In fact, any assumptions beyond those made in the
    :class:`SimpleTrainer` are too much for research.

    The code of this class has been annotated about restrictive assumptions it makes.
    When they do not work for you, you're encouraged to:

    1. Overwrite methods of this class, OR:
    2. Use :class:`SimpleTrainer`, which only does minimal SGD training and
       nothing else. You can then add your own hooks if needed.

    Note that the behavior of this class, like other functions/classes in
    this file, is not stable, since it is meant to represent the "common default behavior".
    It is only guaranteed to work well with the standard models and training workflow in nora.
    To obtain more stable behavior, write your own training logic with other public APIs.

    Examples:

        >>> trainer = DefaultTrainer(cfg)
        >>> trainer.resume_or_load()  # load last checkpoint or MODEL.WEIGHTS
        >>> trainer.train()

    Attributes:
        scheduler:
        checkpointer (DetectionCheckpointer):
        cfg:
    """

    def __init__(self, cfg):
        super().__init__()

        logger = logging.getLogger("nora")
        if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called for nora
            setup_logger()

        # Assume these objects must be constructed in this order.
        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)
        data_loader = self.build_train_dataloader(cfg)

        model = create_ddp_model(model, **cfg.train.ddp)
        self._trainer = (AMPTrainer if cfg.train.amp.enable else SimpleTrainer)(model, data_loader, optimizer)

        self.scheduler = self.build_lr_scheduler(cfg)
        # Assume you want to save checkpoints together with logs/statistics
        self.checkpointer = Checkpointer(model, cfg.train.output_dir, tranier=weakref.proxy(self))

        self.start_iter = 0
        self.max_iter = cfg.train.max_iter
        self.cfg = cfg

        self.register_hooks(self.build_default_hooks())
        self.register_hooks(self.build_custom_hooks())

    def resume_or_load(self, resume: bool = True):
        """
        If `resume==True` and `cfg.train.output_dir` contains the last checkpoint (defined by
        a `last_checkpoint` file), resume from the file. Resuming means loading all
        available states (eg. optimizer and scheduler) and update iteration counter
        from the checkpoint. ``cfg.train.init_checkpoint`` will not be used.

        Otherwise, this is considered as an independent training. The method will load model
        weights from the file `cfg.train.init_checkpoint` (but will not load other states) and start
        from iteration 0.

        Args:
            resume (bool): whether to do resume or not
        """
        self.checkpointer.resume_or_load(self.cfg.train.init_checkpoint, resume=resume)
        if resume and self.checkpointer.has_checkpoint():
            # The checkpoint stores the training iteration that just finished, thus we start
            # at the next iteration
            self.start_iter = self.iter + 1

    def build_default_hooks(self):
        cfg = copy.deepcopy(self.cfg)

        cfg.dataloader.train.num_workers = 0

        ret = [
            hooks.IterationTimer(),
            hooks.LRScheduler(),
        ]

        precise_bn = _try_get_key(cfg, "precise_bn", "train.precise_bn", default={})
        if precise_bn.get("enable", False) and get_bn_modules(self.model):
            ret.append(hooks.PreciseBN(cfg.train.eval_period, self.model, self.build_train_loader(cfg), precise_bn.num_iter))

        if comm.is_main_process():
            ret.append(hooks.PeriodicCheckpointer(self.checkpointer, **cfg.train.checkpointer))

        def test_and_save_results():
            self._last_eval_results = self.test(self.cfg, self.model)
            return self._last_eval_results

        # Do evaluation after checkpointer, because then if it fails,
        # we can use the saved checkpoint to debug.
        ret.append(hooks.EvalHook(cfg.train.eval_period, test_and_save_results))

        if comm.is_main_process():
            # Here the default print/log frequency of each writer is used.
            # run writers in the end, so that evaluation metrics are written
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=cfg.train.log_period))

        return ret

    def build_custom_hooks(self):
        cfg = copy.deepcopy(self.cfg)

        hooks = []
        for hook in cfg.train.get("hooks", []):
            hook = instantiate(hook)
            hooks.append(hook)

        return hooks

    def build_writers(self):
        """
        Build a list of writers to be used using :func:`default_writers()`.
        If you'd like a different list of writers, you can overwrite it in
        your trainer.

        Returns:
            list[EventWriter]: a list of :class:`EventWriter` objects.
        """
        return default_writers(self.cfg.train.output_dir, self.max_iter)

    def train(self):
        """
        Run training.

        Returns:
            OrderedDict of results, if evaluation is enabled. Otherwise None.
        """
        super().train(self.start_iter, self.max_iter)

        expected_results = _try_get_key(self.cfg, "expected_results", "train.expected_results", default=[])
        if len(expected_results) and comm.is_main_process():
            assert hasattr(self, "_last_eval_results"), "No evaluation results obtained during training!"

            verify_results(self.cfg, self._last_eval_results)
            return self._last_eval_results

    def run_step(self):
        self._trainer.iter = self.iter
        self._trainer.run_step()

    def state_dict(self):
        ret = super().state_dict()

        ret["_trainer"] = self._trainer.state_dict()
        return ret

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)

        self._trainer.load_state_dict(state_dict["_trainer"])

    @classmethod
    def build_model(cls, cfg):
        """
        Returns:
            torch.nn.Module:

        Overwrite it if you'd like a different model.
        """
        model = instantiate(cfg.model)
        model.to(torch.device(cfg.train.device))

        logger = logging.getLogger(__name__)
        logger.info(f"Model:\n{model}")

        return model

    @classmethod
    def build_optimizer(cls, cfg, model):
        """
        Returns:
            torch.optim.Optimizer:

        Overwrite it if you'd like a different optimizer.
        """
        cfg = copy.deepcopy(cfg)
        cfg.optimizer.params.model = model

        return instantiate(cfg.optimizer)

    @classmethod
    def build_lr_scheduler(cls, cfg):
        """
        Overwrite it if you'd like a different scheduler.
        """
        return instantiate(cfg.lr_multiplier)

    @classmethod
    def build_train_dataloader(cls, cfg):
        """
        Returns:
            iterable

        Overwrite it if you'd like a different data loader.
        """
        return instantiate(cfg.dataloader.train)

    @classmethod
    def build_test_dataloader(cls, cfg, dataset_name):
        """
        Returns:
            iterable

        Overwrite it if you'd like a different data loader.
        """
        origin_dataset_names = cfg.dataloader.test.dataset.names
        cfg.dataloader.test.dataset.names = dataset_name
        dataloader = instantiate(cfg.dataloader.test)
        cfg.dataloader.test.dataset.names = origin_dataset_names
        return dataloader

    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        """
        Returns:
            DatasetEvaluator or None

        It is not implemented by default.
        """
        raise NotImplementedError("If you want DefaultTrainer to automatically run evaluation, please implement `build_evaluator()` in subclasses (see train_net.py for example). Alternatively, you can call evaluation functions yourself.")

    @classmethod
    def test(cls, cfg, model, evaluators=None):
        """
        Evaluate the given model. The given model is expected to already contain
        weights to evaluate.

        Args:
            cfg (OmegaConf):
            model (nn.Module):
            evaluators (list[DatasetEvaluator] or None): if None, will call
                :meth:`build_evaluator`. Otherwise, must have the same length as
                ``cfg.dataloader.test.dataset.names``.

        Returns:
            dict: a dict of result metrics
        """
        logger = logging.getLogger(__name__)

        if isinstance(evaluators, DatasetEvaluator):
            evaluators = [evaluators]

        dataset_names = cfg.dataloader.test.dataset.names
        if isinstance(dataset_names, str):
            dataset_names = [dataset_names]

        if evaluators is not None:
            assert len(dataset_names) == len(evaluators), f"{len(dataset_names)} != {len(evaluators)}"

        results = OrderedDict()
        for idx, dataset_name in enumerate(dataset_names):
            data_loader = cls.build_test_dataloader(cfg, dataset_name)
            # When evaluators are passed in as arguments,
            # implicitly assume that evaluators can be created before data_loader.
            if evaluators is not None:
                evaluator = evaluators[idx]
            else:
                try:
                    evaluator = cls.build_evaluator(cfg, dataset_name)
                except NotImplementedError:
                    logger.warning("No evaluator found. Use `DefaultTrainer.test(evaluators=)`, or implement its `build_evaluator` method.")
                    results[dataset_name] = {}
                    continue

            results_i = inference_on_dataset(model, data_loader, evaluator)
            results[dataset_name] = results_i

            if comm.is_main_process():
                assert isinstance(results_i, dict), f"Evaluator must return a dict on the main process. Got {results_i} instead."

                logger.info(f"Evaluation results for {dataset_name} in csv format:")
                print_csv_format(results_i)

        if len(results) == 1:
            results = list(results.values())[0]
        return results


# Access basic attributes from the underlying trainer
for _attr in ["model", "data_loader", "optimizer"]:
    setattr(
        DefaultTrainer,
        _attr,
        property(
            # getter
            lambda self, x=_attr: getattr(self._trainer, x),
            # setter
            lambda self, value, x=_attr: setattr(self._trainer, x, value),
        ),
    )
