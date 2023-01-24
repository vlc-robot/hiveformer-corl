import argparse
import tqdm
import torch
import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], 'Mask2Former'))
sys.path.insert(1, os.path.join(sys.path[0], '../Mask2Former'))

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.modeling import build_model

from mask2former import add_maskformer2_config


def get_parser():
    parser = argparse.ArgumentParser(description="maskformer2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="baseline/mask2former_configs/mask2former_original.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        default=["baseline/desk.jpg"],
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def load_mask2former():
    string_args = f"""
        --config-file baseline/mask2former_configs/mask2former_small.yaml 
        """
    # --opts MODEL.WEIGHTS baseline/mask2former_checkpoints/R50_COCO_instanceseg.pkl
    string_args = string_args.split()
    args = get_parser().parse_args(string_args)
    cfg = setup_cfg(args)
    model = build_model(cfg)

    # Freeze backbone
    for p in model.backbone.parameters():
        p.requires_grad = False

    return model


if __name__ == "__main__":
    model = load_mask2former()
