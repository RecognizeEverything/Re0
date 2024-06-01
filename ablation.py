# %%
import os
import numpy as np
from tqdm import tqdm

# %%
def generate_txt(pred, experiment_name, scene_name):
    
    gt_txt_path = os.path.join('/home/yxh666/re0/data/input/scannet200/gt/validation', scene_name+'.txt')
    gt = np.loadtxt(gt_txt_path)
    objects = np.unique(pred)
    txt_path = os.path.join('/home/yxh666/re0/experiment/ablation', experiment_name, scene_name+'.txt')
    os.makedirs(os.path.join('/home/yxh666/re0/experiment/ablation', experiment_name, scene_name), exist_ok=True)
    if not os.path.exists(txt_path):
            open(txt_path, 'w').close()
            
    with open(txt_path, 'w') as file:
        for i in tqdm(objects):
            assert pred.shape[0] == gt.shape[0]
            # gt objects
            gt_objects = gt[pred==i]
            
            unique_values, counts = np.unique(gt_objects, return_counts=True)
            mode_index = np.argmax(counts)
            gt_mode = unique_values[mode_index]
            
            masks = np.zeros(pred.shape[0])
            masks[pred==i] = 1
            txt2_path = os.path.join('/home/yxh666/re0/experiment/ablation', experiment_name, scene_name)
            os.makedirs(txt2_path, exist_ok=True)
            np.savetxt(os.path.join(txt2_path, str(i) + '.txt'), masks, fmt='%d')
            
            file.write(os.path.join(scene_name, str(i) + '.txt') + ' ' + str(int(gt_mode//1000)) + ' ' + '1' + '\n')
if __name__ == "__main__":
# %%
    file_list = os.listdir('/home/yxh666/re0/data320/input/scenes')
    file_list = sorted(file_list)
    # with open("/home/yxh666/re0/data_preprocess/scannet200/scannetv2_val.txt", "r") as f:
    #     file_list = [line.rstrip() for line in f]
    vis = 0
    experiment_name = 'without_m1'
    for scene_name in file_list:
        if scene_name=='intrinsics.txt' or scene_name=='scene0000_00':
            continue
        pred = np.load(f'/home/yxh666/re0/data320/output/output/scannet200/{scene_name}/{scene_name}_ablation_clip_merge_result.npy')
        generate_txt(pred, experiment_name, scene_name)


