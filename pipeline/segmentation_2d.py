import subprocess
import traceback
import os


def run_script(script_path: str, script_args: list, script_type="python"):
    try:
        process = subprocess.Popen([script_type, script_path] + script_args,
                                   stdout=subprocess.PIPE)
        for line in iter(process.stdout.readline, b""):
            print(line.decode().strip())
        process.wait()
    except subprocess.CalledProcessError as e:
        print(f"error code: {e.returncode}")
        print(f"error output: {e.stderr}")
    except Exception as e:
        print(f"other exception: {e}")
        traceback.print_exc()


def seg_with_cropformer(args):
    """
    --config_file : path to config file which saved in submodule/detectron2/configs
    --intput_path : path to colors
    --img_output_path : A file or directory to save output visualizations. If not given, will not save.
    --masks_output_path : file to save pred_masks
    --confidence-threshold : Minimum score for instance predictions to be shown
    --MODEL.WEIGHTS : path to model weights
    --opts : Modify config options using the command-line 'KEY VALUE' pairs
    """

    cropformer_path = os.path.join(
        args.project_dir,
        "submodule/detectron2/projects/CropFormer/demo_cropformer/cropformer.py",
    )
    config_file = args.cropformer_config_path
    input_path = os.path.join(args.project_dir, args.input_data_path, "scenes",
                              args.scene_name, "color")
    img_output_path = os.path.join(args.project_dir, args.output_data_path,
                                   args.experiment_name, args.scene_name,
                                   "cropformer_pictures")
    masks_output_path = os.path.join(args.project_dir, args.output_data_path,
                                     args.experiment_name, args.scene_name,
                                     "masks")
    model_path = args.cropformer_model_path
    cropformer_args = [
        "--config_file",
        str(config_file),
        "--input_path",
        str(input_path),
        "--img_output_path",
        str(img_output_path),
        "--masks_output_path",
        str(masks_output_path),
        "--confidence_threshold",
        str(args.cropformer_confidence_threshold),
        "--device",
        str(args.device),
        "--opts",
        "MODEL.WEIGHTS",
        str(args.cropformer_model_path),
    ]
    run_script(cropformer_path, cropformer_args)
