import argparse
import glob
import multiprocessing as mp
import os

# fmt: off
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
# fmt: on

import tempfile
import time
import warnings

import cv2
import numpy as np
import torch

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger

from mask2former import add_maskformer2_config
from predictor import VisualizationDemo
from tqdm import tqdm

def make_colors():
    from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES
    colors = []
    for cate in COCO_CATEGORIES:
        colors.append(cate["color"])
    return colors


# constants
WINDOW_NAME = "cropformer demo"

def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="maskformer2 demo for builtin configs")
    parser.add_argument(
        "--config_file",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--input_path",
        help="path to colors",
    )
    parser.add_argument(
        "--img_output_path",
        help="A file or directory to save output visualizations. "
        "If not given, will not save.",
    )
    parser.add_argument(
        "--masks_output_path",
        help="save cropformer result [h,w]",
    )
    parser.add_argument(
        "--confidence_threshold",
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

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    inputs = []
    for file in os.listdir(args.input_path):
        inputs.append(os.path.join(args.input_path, file))
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))
    cfg = setup_cfg(args)
    
    colors = make_colors()
    demo = VisualizationDemo(cfg)
    tqdm_bar = tqdm(inputs)
    for path in tqdm_bar:
        idx = int(path.split('/')[-1][:-4])
        
        # use PIL, to be consistent with evaluation
        img = read_image(path, format="BGR")
        start_time = time.time()
        predictions = demo.run_on_image(img)
        tqdm_bar.set_postfix({
            "color path": path,
            "detected instance": len(predictions["instances"]) if "instances" in predictions else "finished",
            "cost time": "{:.2f}s".format(time.time() - start_time),
        })

        ##### color_mask
        pred_masks = predictions["instances"].pred_masks
        pred_scores = predictions["instances"].scores
        
        # select by confidence threshold
        selected_indexes = (pred_scores >= args.confidence_threshold)
        selected_scores = pred_scores[selected_indexes]
        selected_masks  = pred_masks[selected_indexes]
        selected_scores_save = selected_scores.cpu().numpy()
        selected_masks_save = selected_masks.cpu().numpy()
        
        _, m_H, m_W = selected_masks.shape
        mask_id = np.zeros((m_H, m_W), dtype=np.uint8)
        
        # mask=0 是没有背景
        # rank
        selected_scores, ranks = torch.sort(selected_scores)
        ranks = ranks + 1
        for index in ranks:
            mask_id[(selected_masks[index-1]==1).cpu().numpy()] = int(index)
        
        os.makedirs(args.masks_output_path, exist_ok=True)
        np.save(os.path.join(args.masks_output_path, str(idx)+'.npy'), mask_id)
        if args.img_output_path is not None:
            os.makedirs(args.img_output_path, exist_ok=True)
            unique_mask_id = np.unique(mask_id)
            color_mask = np.zeros(img.shape, dtype=np.uint8)
            for count in unique_mask_id:
                if count == 0:
                    continue
                color_mask[mask_id==count] = colors[count]
            
            vis_mask = np.concatenate((img, color_mask), axis=0)
            out_filename = os.path.join(args.img_output_path, os.path.basename(path))
            cv2.imwrite(out_filename, vis_mask)
