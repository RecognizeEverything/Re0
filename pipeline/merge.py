import os
import torch
import numpy as np
from tqdm import tqdm
from utils import compute_mapping, load_ply


def merge_masks(point_mask_result, args):
    device = torch.device(args.device)
    # load pointclouds
    data_path = os.path.join(args.project_dir, args.input_data_path, "scenes")
    points, _ = load_ply(
        os.path.join(data_path, args.scene_name,
                     args.scene_name + '_vh_clean_2.ply'))
    points_cpu = points
    points = torch.tensor(points, device=device)

    # Cluster point clouds by objects
    objects = np.unique(point_mask_result)

    data_path = os.path.join(args.project_dir, args.input_data_path, "scenes")

    # get mask_npy_file
    scene_output_path = os.path.join(args.project_dir, args.output_data_path,
                                     args.experiment_name, args.scene_name)
    masks_npy_path = os.path.join(scene_output_path, "masks")
    mask_npy_files = os.listdir(masks_npy_path)
    range_list = range(0, len(mask_npy_files), args.frame_skip)
    point_mask_result = torch.tensor(point_mask_result,
                                     dtype=torch.long).to(device)
    objects = torch.tensor(objects, dtype=torch.long)
    # Traverse all frames
    for _, mask_index in tqdm(enumerate(range_list), total=len(range_list)):
        # Calculate the mask for the i-th frame
        mask_npy_file = mask_npy_files[mask_index]
        mask_data = torch.from_numpy(
            np.load(os.path.join(masks_npy_path, mask_npy_file))).to(device)

        # Calculate the pixel coordinates of all points projected onto the i-th frame
        frame_id = mask_npy_file[:-4]
        mapping = compute_mapping(points_cpu, data_path, args.scene_name,
                                  frame_id)
        if mapping[:, 2].sum() == 0:
            continue
        mapping = torch.from_numpy(mapping).to(device)

        # Traverse all objects
        for i in range(0, objects.shape[0]):

            # Skip objects with the ID of 0, as it means they haven't been segmented
            if objects[i] == 0:
                continue
            '''
            points_i : points of object_i
            mapping_i : The points of object i projected onto frame i
            masks_i : The mask of these projected points on the 2D plane
            '''
            points_i_idx = point_mask_result == objects[i].item()
            mapping_i_all = mapping[points_i_idx, :]
            mapping_i = mapping_i_all[mapping_i_all[:, 2] == 1, :]
            masks_i = mask_data[mapping_i[:, 0], mapping_i[:, 1]]

            # The number of points of object i falling on frame i should not be less
            # than iou_threshold_x of the total number of points of object i
            if mapping_i_all.shape[0] == 0:
                continue
            if mapping_i.shape[0] / mapping_i_all.shape[
                    0] < args.iou_threshold_x:
                continue

            mode_value_i, mode_count_i = torch.mode(masks_i)
            # Object i must have more than iou_threshold_y of its points projected onto the same mask
            # Point count must be greater than the threshold
            if mode_count_i / masks_i.shape[
                    0] < args.iou_threshold_y or mode_count_i < args.points_nums_threshold:
                continue

            # Traverse (j+1, objects.shape[0]) objects
            for j in range(i + 1, objects.shape[0]):
                points_j_idx = point_mask_result == objects[j].item()
                mapping_j_all = mapping[points_j_idx, :]
                mapping_j = mapping_j_all[mapping_j_all[:, 2] == 1, :]
                masks_j = mask_data[mapping_j[:, 0], mapping_j[:, 1]]
                if mapping_j_all.shape[0] == 0:
                    continue
                if mapping_j.shape[0] / mapping_j_all.shape[
                        0] < args.iou_threshold_x:
                    continue

                mode_value_j, mode_count_j = torch.mode(masks_j)

                if mode_count_j / masks_j.shape[
                        0] < args.iou_threshold_y or mode_count_j < args.points_nums_threshold:
                    continue

                # need same mask
                if mode_value_i != mode_value_j:
                    continue

                # merge
                point_mask_result[points_j_idx] = objects[i].item()

    merge_save_path = os.path.join(args.output_data_path, args.experiment_name,
                                   args.scene_name)
    if not os.path.exists(merge_save_path):
        os.makedirs(merge_save_path)
        print("Path has been created:  ", merge_save_path)
    save_prediction_path = os.path.join(merge_save_path,
                                        args.scene_name + '_merge_result.npy')
    if args.save_merge:
        np.save(save_prediction_path, point_mask_result.cpu().numpy())
        print("Successfully finish merge! ", save_prediction_path)
    return point_mask_result.cpu().numpy()
