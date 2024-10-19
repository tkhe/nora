# Copyright (c) Facebook, Inc. and its affiliates.

import argparse
import os
from itertools import chain

import cv2
from tqdm import tqdm

from nora.config import LazyConfig
from nora.config import instantiate
from nora.data import DatasetCatalog
from nora.data import MetadataCatalog
from nora.data import detection_utils as utils
from nora.data.build import filter_images_with_few_keypoints
from nora.utils.logger import setup_logger
from nora.utils.visualizer import Visualizer


def setup(args):
    cfg = LazyConfig.load(args.config_file)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)
    cfg.dataloader.train.num_workers = 0
    cfg.dataloader.test.num_workers = 0
    return cfg


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize ground-truth data")
    parser.add_argument(
        "--source",
        choices=["annotation", "dataloader"],
        required=True,
        help="visualize the annotations or the data loader (with pre-processing)",
    )
    parser.add_argument("--config-file", metavar="FILE", help="path to config file")
    parser.add_argument("--output-dir", default="./", help="path to output directory")
    parser.add_argument("--show", action="store_true", help="show output in a window")
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    logger = setup_logger()
    logger.info(f"Arguments: {str(args)}")
    cfg = setup(args)

    dirname = args.output_dir
    os.makedirs(dirname, exist_ok=True)
    if isinstance(cfg.dataloader.train.dataset.names, str):
        names = [cfg.dataloader.train.dataset.names]
    else:
        names = cfg.dataloader.train.dataset.names

    metadata = MetadataCatalog.get(names[0])

    def output(vis, fname):
        if args.show:
            print(fname)
            cv2.imshow("window", vis.get_image()[:, :, ::-1])
            cv2.waitKey()
        else:
            filepath = os.path.join(dirname, fname)
            print(f"Saving to {filepath} ...")
            vis.save(filepath)

    scale = 1.0
    if args.source == "dataloader":
        train_dataloader = instantiate(cfg.dataloader.train)
        for batch in train_dataloader:
            for per_image in batch:
                # Pytorch tensor is in (C, H, W) format
                image = per_image["image"].permute(1, 2, 0).cpu().detach().numpy()
                image = utils.convert_image_to_rgb(image, cfg.dataloader.train.mapper.image_format)

                visualizer = Visualizer(image, metadata=metadata, scale=scale)
                target_fields = per_image["instances"].get_fields()
                labels = [metadata.thing_classes[i] for i in target_fields["gt_classes"]]
                vis = visualizer.overlay_instances(
                    labels=labels,
                    boxes=target_fields.get("gt_boxes", None),
                    masks=target_fields.get("gt_masks", None),
                    keypoints=target_fields.get("gt_keypoints", None),
                )
                output(vis, str(per_image["image_id"]) + ".jpg")
    else:
        dicts = list(chain.from_iterable([DatasetCatalog.get(k) for k in names]))
        if cfg.dataloader.train.mapper.use_keypoint:
            dicts = filter_images_with_few_keypoints(dicts, 1)
        for dic in tqdm(dicts):
            image = utils.read_image(dic["file_name"], "RGB")
            visualizer = Visualizer(image, metadata=metadata, scale=scale)
            vis = visualizer.draw_dataset_dict(dic)
            output(vis, os.path.basename(dic["file_name"]))
