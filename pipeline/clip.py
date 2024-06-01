import torch
import pickle
from utils import compute_mapping, load_ply
from utils import Features
import clip
import os
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from tqdm import tqdm
import torchvision.transforms as transforms
import numpy as np
import shutil


def bounding_box_cropformer(mask_data, numbers):
    '''
    
    '''
    indices = torch.nonzero(torch.isin(mask_data, numbers), as_tuple=False)
    assert indices.size(0) > 0, "no index"
    min_row = indices[:, 0].min().item()
    max_row = indices[:, 0].max().item()
    min_col = indices[:, 1].min().item()
    max_col = indices[:, 1].max().item()
    return min_row, max_row, min_col, max_col


def bounding_box(points_xy):
    min_row = points_xy[:, 0].min().item()
    max_row = points_xy[:, 0].max().item()
    min_col = points_xy[:, 1].min().item()
    max_col = points_xy[:, 1].max().item()
    return min_row, max_row, min_col, max_col


def coloring(image, mask_data, unique_mask_nonzero):
    '''
        Blacken the area outside the mask
        image: Original image size
    '''
    mask_image = torch.where(torch.isin(mask_data, unique_mask_nonzero), image,
                             torch.tensor(0).to(image.device))
    return mask_image


def crop(min_y, max_y, min_x, max_x, image):
    h = max_y - min_y
    w = max_x - min_x
    max_x2 = max(0, int(max_x - w * 0.1))
    max_y2 = max(0, int(max_y - w * 0.1))
    min_x2 = min(image.shape[1], int(max_x + h * 0.1))
    min_y2 = min(image.shape[2], int(max_y + h * 0.1))
    max_x3 = max(0, int(max_x - w * 0.2))
    max_y3 = max(0, int(max_y - w * 0.2))
    min_x3 = min(image.shape[1], int(max_x + h * 0.2))
    min_y3 = min(image.shape[2], int(max_y + h * 0.2))
    return [
        image[:, min_y:max_y, min_x:max_x],
        image[:, min_y2:max_y2, min_x2:max_x2], image[:, min_y3:max_y3,
                                                      min_x3:max_x3]
    ]


def get_image(image, points_xy, mask_data, stage):
    '''
        mask: (240,340)(h*w)
        image: (3,240,340)(h*w)
        points_xy: points 2D coordinates
        return 
            cropped_image: cropped_image*3   
    '''
    masks = mask_data[points_xy[:, 0], points_xy[:, 1]]
    unique_mask, counts = torch.unique(masks, return_counts=True)
    # x:w, y:h
    if stage == 1:
        # Threshold helps filter out some of the outliers
        threshorld = max(50, 0.2 * points_xy.shape[0])
        unique_mask = unique_mask[counts >= threshorld]
        nonzero_indices = (unique_mask != 0)
        # All the masks covered by this point cloud
        unique_mask_nonzero = unique_mask[nonzero_indices]
        if torch.all(unique_mask == 0).item():
            return [image[:, 0:0, 0:0], image[:, 0:0, 0:0], image[:, 0:0, 0:0]]
        min_y, max_y, min_x, max_x = bounding_box_cropformer(
            mask_data, unique_mask_nonzero)
        image_masked = coloring(image, mask_data, unique_mask_nonzero)
        # print((max_x - min_x)*(max_y- min_y))
        return crop(min_y, max_y, min_x, max_x, image_masked)
    else:
        min_y, max_y, min_x, max_x = bounding_box(points_xy)
        return crop(min_y, max_y, min_x, max_x, image)


def get_clip_feature(image, clip_model, clip_preprocess, args, frame_id, obj,
                     stage):
    '''
        return clip feature
    '''
    to_pil = transforms.ToPILImage()
    image = (to_pil(image)).convert('RGB')

    # image.save(os.path.join(args.project_dir, args.output_data_path, args.experiment_name, 'mask_image', args.scene_name,  str(stage), frame_id+'_'+ obj+'.jpg' ))
    mask_input = clip_preprocess(image).unsqueeze(0).to(args.device)
    with torch.no_grad():
        image_features = clip_model.encode_image(mask_input)
    return image_features


def save_feature(feature_dict, args, stage):
    '''
        First, the segmentation results in the first stage will generate a dictionary, 
        and after merging, a dictionary will also be generated.
    '''
    assert stage == 1 or stage == 2
    if stage == 1:
        dict_save_path = os.path.join(args.project_dir, args.output_data_path,
                                      args.experiment_name, args.scene_name,
                                      args.scene_name + '_clip_feature.pkl')
    if stage == 2:
        dict_save_path = os.path.join(
            args.project_dir, args.output_data_path, args.experiment_name,
            args.scene_name, args.scene_name + '_merged_clip_feature.pkl')
    with open(dict_save_path, 'wb') as f:
        pickle.dump(feature_dict, f)


def get_empty_features_list(array, device):
    '''
        Create a feature class to facilitate access to features
    '''
    feature_dict = {}
    for num in np.unique(array):
        feature_dict[num] = Features(num, device)
    return feature_dict


def add_clip(merge_result, args, clip_model, clip_preprocess, stage):
    '''
        merge_result: result of m1
        weights: The ratio of visible points to the total number of points in the object
        stage: 0 or 1, 0: m2, 1: get open-vocabulary
    '''
    os.makedirs(os.path.join(args.project_dir, args.output_data_path,
                             args.experiment_name, 'mask_image',
                             args.scene_name, str(stage)),
                exist_ok=True)

    device = torch.device(args.device)

    # Create a feature class to facilitate access to features
    feature_dict = get_empty_features_list(merge_result, args.device)

    # Get the list of frames that need to be processed
    scene_output_path = os.path.join(args.project_dir, args.output_data_path,
                                     args.experiment_name, args.scene_name)
    masks_npy_path = os.path.join(scene_output_path, "masks")
    mask_npy_files = os.listdir(masks_npy_path)
    range_list = range(0, len(mask_npy_files), args.frame_skip)

    # get points and segmentation
    data_path = os.path.join(args.project_dir, args.input_data_path, "scenes")
    points, _ = load_ply(
        os.path.join(data_path, args.scene_name,
                     f"{args.scene_name}_vh_clean_2.ply"))
    objs_seg = np.unique(merge_result)
    merge_result = torch.tensor(merge_result, dtype=torch.int64, device=device)
    for i, mask_index in tqdm(enumerate(range_list), total=len(range_list)):
        mask_npy_file = mask_npy_files[mask_index]
        frame_id = mask_npy_file[:-4]
        mask_data = torch.from_numpy(
            np.load(os.path.join(masks_npy_path, mask_npy_file))).to(device)
        mapping = compute_mapping(points, data_path, args.scene_name, frame_id)
        mapping = torch.tensor(mapping, dtype=torch.int64, device=device)

        # read image
        image_orign = Image.open(
            os.path.join(data_path, args.scene_name, 'color',
                         str(frame_id) + '.jpg'))
        transform = transforms.ToTensor()
        image = transform(image_orign)
        image = image.to(device)
        for obj in objs_seg:
            object_pos = (merge_result == obj)
            # print(object_pos)
            visible_points = mapping[object_pos, :]
            visible_points = visible_points[visible_points[:, 2] == 1]
            weights = visible_points.shape[0] / torch.sum(object_pos).item()
            if stage == 1 and weights < args.iou_threshold_x:
                continue

            if stage == 2 and visible_points.shape[0] < 50:
                continue
            cropped_images = get_image(image, visible_points[:, :2], mask_data,
                                       stage)
            for cropped_image in cropped_images:
                if cropped_image.shape[1] * cropped_image.shape[2] == 0:
                    continue
                clip_feature = get_clip_feature(cropped_image, clip_model,
                                                clip_preprocess, args,
                                                str(frame_id), str(obj), stage)
                feature_dict[obj].add_values(weights, clip_feature, frame_id)

    save_feature(feature_dict, args, stage)
    return feature_dict
