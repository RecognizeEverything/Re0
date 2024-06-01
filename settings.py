import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment_name",
        type=str,
        help="experiment_name",
    )
    parser.add_argument(
        "--input_data_path",
        type=str,
        help="Root directory path of input data",
    )
    parser.add_argument(
        "--output_data_path",
        type=str,
        help="Root directory path of input data",
    )
    parser.add_argument(
        "--scene_name",
        default="scene0000_00",
        type=str,
        help="scene name",
    )
    parser.add_argument(
        "--frame_skip",
        default=1,
        type=int,
        help=
        "Frame skip steps for extracting frames when processing a scene in instance segmentation",
    )
    parser.add_argument(
        "--nb_workers",
        default=30,
        type=int,
        help="nb_workers for pandarallel",
    )
    parser.add_argument(
        "--device",
        default="cuda:1",
        type=str,
        help="The device for running",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="debug switch(default False)",
    )
    parser.add_argument(
        "--checkpoint_path",
        default="checkpoint/",
        type=str,
        help="Root directory path of checkpoint",
    )
    parser.add_argument(
        "--cropformer_model_path",
        default="/media/yxh666/CAD490/yxh666/CropFormer_hornet_3x_03823a.pth",
        type=str,
        help="Path of pretrained 2d segmentation model",
    )
    parser.add_argument(
        "--cropformer_config_path",
        default=
        "submodule/Entity/Entityv2/CropFormer/configs/entityv2/entity_segmentation/cropformer_hornet_3x.yaml",
        type=str,
        help="Path of cropformer config",
    )
    parser.add_argument(
        "--cropformer_confidence_threshold",
        default=0.5,
        type=float,
        help="Minimum score for instance predictions to be shown in cropformer",
    )
    parser.add_argument(
        "--iou_threshold_x",
        default=0.4,
        type=float,
        help=
        "The number of points of object i falling on frame i should not be less than iou_threshold_x of the total number of points of object ir",
    )
    parser.add_argument(
        "--iou_threshold_y",
        default=0.6,
        type=float,
        help=
        "Object i must have more than iou_threshold_y of its points projected onto the same mask ",
    )
    parser.add_argument(
        "--points_nums_threshold",
        default=1,
        type=int,
        help="Point count must be greater than the threshold",
    )
    parser.add_argument(
        "--txt_save_path",
        default='experiment/pred_txt/0_05',
        type=str,
        help="Path to store the generated txt text for testing mAP",
    )
    parser.add_argument(
        "--instance_txt_save_path",
        #default='experiment/instance_segmentation',
        type=str,
        help="Path to store the generated txt text for testing mAP",
    )
    parser.add_argument(
        "--label_txt_save_path",
        default='experiment/sementic_segmentation',
        type=str,
        help="Path to store the generated txt text for testing mAP",
    )
    parser.add_argument(
        "--threshold_miou",
        default=0.5,
        type=float,
    )
    parser.add_argument(
        "--save_merge",
        default=True,
        type=bool,
    )
    parser.add_argument("--clip_version",
                        default="ViT-L/14@336px",
                        type=str,
                        help='clip model version')

    parser.add_argument("--topk", default=9, type=int, help='top-k frame')

    args = parser.parse_args()

    return args
