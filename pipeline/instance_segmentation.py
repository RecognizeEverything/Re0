import os
import torch

import pandas as pd
import numpy as np

from tqdm import tqdm
from pandarallel import pandarallel
from utils import compute_mapping, load_ply


def set_pandarallel_workers(nb_workers):
    pandarallel.initialize(progress_bar=False, nb_workers=nb_workers)


def mask_result_mapping(args):
    """
    return
        all_category_result : cropformer segmentation mask , [frames, h, w],
    """
    device = torch.device(args.device)
    data_path = os.path.join(args.project_dir, args.input_data_path, "scenes")
    points, _ = load_ply(
        os.path.join(data_path, args.scene_name,
                     f"{args.scene_name}_vh_clean_2.ply"))
    scene_output_path = os.path.join(args.project_dir, args.output_data_path,
                                     args.experiment_name, args.scene_name)
    masks_npy_path = os.path.join(scene_output_path, "masks")

    mask_npy_files = os.listdir(masks_npy_path)
    range_list = range(0, len(mask_npy_files), args.frame_skip)
    all_category_result = torch.zeros(
        [points.shape[0], len(range_list)], device=device)
    for i, mask_index in tqdm(enumerate(range_list), total=len(range_list)):
        mask_npy_file = mask_npy_files[mask_index]
        mask_data = torch.from_numpy(
            np.load(os.path.join(masks_npy_path, mask_npy_file))).to(device)
        frame_id = mask_npy_file[:-4]
        mapping = compute_mapping(points, data_path, args.scene_name, frame_id)
        if mapping[:, 2].sum() == 0:
            continue

        mapping = torch.from_numpy(mapping).to(device)
        indices = torch.nonzero(mapping[:, 2] == 1).squeeze()

        indices = indices.tolist()
        for idx in indices:
            mask = mask_data[mapping[idx][0]][mapping[idx][1]]
            all_category_result[idx][i] = mask.item()

    all_category_result = all_category_result.cpu().numpy()
    # if args.debug:
    #     print(
    #         f"saving all_category_result in {os.path.join(scene_output_path, f'{args.scene_name}_acr.npy')}."
    #     )
    #     np.save(
    #         os.path.join(scene_output_path, f"{args.scene_name}_acr.npy"),
    #         all_category_result,
    #     )
    # return all_category_result


def reset_label(category_result: pd.Series,
                label_book: pd.Series,
                threshold=0.5):
    max_category_label = label_book.max()

    def reset_label_for_one_category(one_category_series: pd.Series):
        nonlocal max_category_label
        if one_category_series.sum() == 0:
            return one_category_series
        value_count = label_book[one_category_series.index].value_counts()
        max_count_num, max_count_label = value_count.max(), value_count.idxmax(
        )
        occupy_rate = max_count_num / len(one_category_series)

        if max_count_label == 0 or occupy_rate <= threshold:
            max_category_label += 1
            max_count_label = max_category_label
        one_category_series[:] = max_count_label
        return one_category_series

    res = category_result.groupby(category_result).apply(
        lambda x: reset_label_for_one_category(x))
    res = res.reset_index(level=0, drop=True)
    return res


def merge_label(label_book: pd.Series, category_result: pd.Series):

    def process_row(row):
        a, b = row.iloc[0], row.iloc[1]
        return max(a, b)

    category_result = reset_label(category_result, label_book)

    merge_df = pd.concat([label_book, category_result], axis=1)
    new_label_book = merge_df.parallel_apply(process_row, axis=1)

    return new_label_book


def mask_based_instance_segmantation(all_category_result, args):
    all_category_result = pd.DataFrame(all_category_result)
    label_book = all_category_result[0]
    for i in tqdm(range(1, len(all_category_result.columns))):
        category_result = all_category_result.iloc[:, i]
        label_book = merge_label(label_book, category_result)

    scene_output_path = os.path.join(args.project_dir, args.output_data_path,
                                     args.experiment_name, args.scene_name)
    ins_seg_result_path = os.path.join(scene_output_path,
                                       f"{args.scene_name}_ins_seg.npy")
    print(f"saving instance segmantation result in {ins_seg_result_path}.")
    np.save(ins_seg_result_path, label_book)
    return label_book.values
