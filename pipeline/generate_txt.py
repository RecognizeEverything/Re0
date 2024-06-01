import numpy as np
import os
from tqdm import tqdm
'''
    unzip_root/
    |-- scene0707_00.txt
    |-- scene0708_00.txt
    |-- scene0709_00.txt
        ⋮
    |-- scene0806_00.txt
    |-- predicted_masks/
        |-- scene0707_00_000.txt
        |-- scene0707_00_001.txt
            ⋮
            
    E.g., scene0707_00.txt should be of the format:
    predicted_masks/scene0707_00_000.txt 10 0.7234
    predicted_masks/scene0707_00_001.txt 36 0.9038
        ⋮
        
    and predicted_masks/scene0707_00_000.txt could look like:
    0
    1
    ⋮
    0
'''


def generate_txt(pred, args):
    '''
    our mAP
    '''
    gt_txt_path = os.path.join(args.project_dir, args.input_data_path, 'gt',
                               'instance_gt', 'validation',
                               args.scene_name + '.txt')
    gt = np.loadtxt(gt_txt_path)

    objects = np.unique(pred)

    txt_path = os.path.join(args.project_dir, args.output_data_path, 'map_txt',
                            args.scene_name + '.txt')
    os.makedirs(os.path.join(args.project_dir, args.output_data_path,
                             'map_txt'),
                exist_ok=True)
    if not os.path.exists(txt_path):
        open(txt_path, 'w').close()

    with open(txt_path, 'w') as file:
        for i in tqdm(objects):
            assert pred.shape[0] == gt.shape[0]
            # gt objects
            gt_objects = gt[pred == i]

            unique_values, counts = np.unique(gt_objects, return_counts=True)
            mode_index = np.argmax(counts)
            gt_mode = unique_values[mode_index]

            masks = np.zeros(pred.shape[0])
            masks[pred == i] = 1
            txt2_path = os.path.join(args.project_dir, args.output_data_path,
                                     'map_txt', args.scene_name)
            os.makedirs(txt2_path, exist_ok=True)
            np.savetxt(os.path.join(txt2_path,
                                    str(i) + '.txt'),
                       masks,
                       fmt='%d')

            file.write(
                os.path.join(args.scene_name,
                             str(i) + '.txt') + ' ' +
                str(int(gt_mode // 1000)) + ' ' + '1' + '\n')


def generate_instance_txt(pred, label, args):
    '''
        label.shape = pred.shape
        pred : a segmentation result without semantics
        label : the semantics represented by each point in scannet200
    '''

    txt_path = os.path.join(args.project_dir, args.output_data_path,
                            args.experiment_name, 'instance_segmentation',
                            args.scene_name + '.txt')
    txt2_path = os.path.join(args.project_dir, args.output_data_path,
                             args.experiment_name, 'instance_segmentation',
                             args.scene_name)
    os.makedirs(txt2_path, exist_ok=True)
    #os.makedirs(txt_path, exist_ok=True)
    if not os.path.exists(txt_path):
        open(txt_path, 'w').close()

    objects = np.unique(pred)
    with open(txt_path, 'w') as file:
        for i in tqdm(objects):
            masks = np.zeros(pred.shape[0])
            if i == 0 or np.count_nonzero(pred == i) < 50:
                # if i==0:
                continue
            masks[pred == i] = 1

            np.savetxt(os.path.join(txt2_path,
                                    str(i) + '.txt'),
                       masks,
                       fmt='%d')
            classification = label[pred == i]
            assert np.all(
                classification == classification[0]
            ), 'The classification of a single instance is not unique'

            file.write(
                os.path.join(args.scene_name,
                             str(i) + '.txt') + ' ' +
                str(int(classification[0])) + ' ' + '1' + '\n')
