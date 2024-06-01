import os
from loguru import logger
from settings import parse_args
from pipeline.segmentation_2d import seg_with_cropformer
from pipeline.merge import merge_masks
from pipeline.clip import add_clip
from pipeline.generate_txt import generate_txt, generate_instance_txt
from pipeline.merge_clip import merge_clip
from pipeline.semantic_segmentation import semantic_segmentation
from pipeline.instance_segmentation import (
    set_pandarallel_workers,
    mask_result_mapping,
    mask_based_instance_segmantation,
)
from pipeline.visualization import visiualization


def main():
    args = parse_args()
    args.project_dir = os.path.dirname(os.path.abspath(__file__))
    print(args)

    logger.info("Step1: Segmantation with Cropformer")
    seg_with_cropformer(args)

    logger.info("Step2: Mapping Result to Point Cloud")
    all_category_result = mask_result_mapping(args)

    logger.info("Step3: Mask Based Instance Segmantation")
    set_pandarallel_workers(args.nb_workers)
    point_mask_result = mask_based_instance_segmantation(
        all_category_result, args)

    logger.info("Step4: Merge the masks")
    segmentation_result = merge_masks(point_mask_result, args)

    logger.info("Step5: Add dict_1")
    clip_dict = add_clip(segmentation_result, args, stage=1)

    logger.info("Step6: Clip merge")
    clip_merge_result = merge_clip(segmentation_result, clip_dict, args)

    logger.info("Step7: Add clip_2")
    merged_clip_dict = add_clip(clip_merge_result, args, stage=2)

    logger.info("Step8: Semantic segmentation")
    semantic_segmentation_result = semantic_segmentation(
        clip_merge_result, merged_clip_dict, args)

    # logger.info("Step9: Generate instance txt")
    # generate_instance_txt(clip_merge_result, semantic_segmentation_result,
    #                       args)

    logger.info("Step10: Visualization")
    visiualization(args, clip_merge_result)


if __name__ == "__main__":
    main()
