from utils import *
import numpy as np
import os
import torch
import pickle
import clip


def semantic_segmentation(merge_result, points_dict, args, clip_model,
                          clip_preprocess):
    valid_class_ids = [item for item in VALID_CLASS_IDS_200]

    # get text feature
    text_list_features = []
    for word in CLASS_LABELS_200:
        text_inputs = clip.tokenize(word).to(device=args.device)
        with torch.no_grad():
            text_feature = clip_model.encode_text(text_inputs)
        text_list_features.append(text_feature)

    text_features = torch.cat(text_list_features)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    # get point cloud feature
    points_list_key = []
    points_list_feature = []
    for key in points_dict.keys():
        if len(points_dict[key].data) < 1:
            continue
        points_list_key.append(key)
        points_list_feature.append(points_dict[key].get_topk_feature(
            min(1, len(points_dict[key].data))))
        #points_list_feature.append(points_dict[key].get_topk_feature(min(args.topk, len(points_dict[key].data))))
        # points_list_feature.append(points_dict[key].get_topk_feature(args.topk).unsqueeze(0))
    points_features = torch.cat(points_list_feature)
    points_features = torch.tensor(points_features, dtype=torch.float16)
    points_features /= points_features.norm(dim=-1, keepdim=True)

    similarity = (100.0 * points_features @ text_features.T).softmax(dim=-1)
    values, indices = similarity[:].topk(1)
    result = np.zeros_like(merge_result)
    for i in range(values.shape[0]):
        result[merge_result == points_list_key[i]] = valid_class_ids[
            indices[i]]
    np.save(
        os.path.join(args.project_dir, args.output_data_path,
                     args.experiment_name, args.scene_name,
                     args.scene_name + '_semantic_segmentation.npy'), result)
    return result
