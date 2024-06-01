import os
import torch
import numpy as np
from tqdm import tqdm
from utils import compute_mapping, load_ply, chamfer_distance


def get_clip_features(clip_dict, args):
    clip_list_key = []
    clip_list_feature = []
    for key in clip_dict.keys():
        if len(clip_dict[key].data) < 1:
            continue
        clip_list_key.append(key)
        # if len(clip_dict[key].data) < args.topk:
        #     clip_list_feature.append(clip_dict[key].get_all_feature().unsqueeze(0))
        # else:
        #     clip_list_feature.append(clip_dict[key].get_all_feature().unsqueeze(0))
        clip_list_feature.append(clip_dict[key].get_all_feature().unsqueeze(0))
        # clip_list_feature.append(clip_dict[key].get_topk_feature(min(len(clip_dict[key].data),args.topk)))
    clip_features = torch.cat(clip_list_feature)
    clip_features /= clip_features.norm(dim=-1, keepdim=True)
    clip_features = torch.tensor(clip_features, dtype=torch.float16)
    return clip_list_key, clip_features


def get_boundingbox(points_xyz):
    return np.min(points_xyz[:, 0]), np.min(points_xyz[:, 1]), np.min(points_xyz[:, 2]) , \
        np.max(points_xyz[:, 0]) , np.max(points_xyz[:, 1]), np.max(points_xyz[:, 2])


def checkIntersection(points_1, points_2):
    box1 = get_boundingbox(points_1)
    box2 = get_boundingbox(points_2)
    x1_min, y1_min, z1_min, x1_max, y1_max, z1_max = box1
    x2_min, y2_min, z2_min, x2_max, y2_max, z2_max = box2

    if (x1_min <= x2_max and x1_max >= x2_min and y1_min <= y2_max
            and y1_max >= y2_min and z1_min <= z2_max and z1_max >= z2_min):
        # v1 = (x1_max-x1_min)*(y1_max-y1_min)*(z1_max-z1_min)
        # v2 = (x2_max-x2_min)*(y2_max-y2_min)*(z2_max-z2_min)
        # iou_v = (min(x1_max, x2_max)-max(x1_min, x2_min))*(min(y1_max, y2_max)-max(y1_min, y2_min))*(min(z1_max, z2_max)-max(z1_min, z2_min))
        # if (iou_v/v1>0.2) or (iou_v/v2>0.2):
        #     return True
        # else:
        #     return False
        return True
    else:
        return False


def save(clip_merge_result, args):
    merge_save_path = os.path.join(args.output_data_path, args.experiment_name,
                                   args.scene_name)
    save_prediction_path = os.path.join(
        merge_save_path, args.scene_name + '_clip_merge_result.npy')

    np.save(save_prediction_path, clip_merge_result)
    print("Successfully finish clip merge! ", save_prediction_path)


def merge_clip(segmentation_result, clip_dict, args):
    xyz, _ = load_ply(os.path.join(args.project_dir, args.input_data_path, "scenes",\
        args.scene_name, args.scene_name+'_vh_clean_2.ply'))
    clip_merge_result = segmentation_result.copy()
    image_list_key, image_features = get_clip_features(clip_dict, args)
    print(len(image_list_key))
    for i in range(image_features.shape[0]):
        image_feature = image_features[i]
        slices = [j for j in range(image_features.shape[0]) if j != i]
        similarity = (100.0 *
                      image_feature @ image_features[slices].T).softmax(dim=-1)

        values, indices = similarity.topk(len(slices))
        key_i = image_list_key[i]
        for j in range(len(slices)):
            key_j = image_list_key[
                indices[j]] if indices[j] < i else image_list_key[indices[j] +
                                                                  1]
            # Merge large numbers to small ones
            if key_j < key_i:
                continue
            if checkIntersection(xyz[segmentation_result == key_i],
                                 xyz[segmentation_result == key_j]):
                clip_merge_result[clip_merge_result == key_j] = key_i
            # TODO:
            if values[j] < 0.01:
                break
    if args.save_merge:
        save(clip_merge_result, args)
    return clip_merge_result
