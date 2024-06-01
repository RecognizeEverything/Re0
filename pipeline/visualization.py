import os
import numpy as np
from pipeline.vis_utils import *
from plyfile import PlyData, PlyElement
import random


def rand_color(seed=None):
    if seed is not None:
        random.seed(seed)

    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)

    return (r, g, b)


def visiualization(args, segmentation):
    data_path = os.path.join(args.project_dir, args.input_data_path, "scenes")
    plydata = PlyData.read(
        os.path.join(data_path, args.scene_name,
                     f"{args.scene_name}_vh_clean_2.ply"))
    vertex_data = plydata['vertex']
    red = vertex_data['red']
    green = vertex_data['green']
    blue = vertex_data['blue']

    colors = np.unique(segmentation)
    for color in colors:
        r, g, b = rand_color(color)
        red[segmentation == color] = r
        green[segmentation == color] = g
        blue[segmentation == color] = b
    new_vertex_data = np.array(list(
        zip(vertex_data['x'], vertex_data['y'], vertex_data['z'], red, green,
            blue)),
                               dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                                      ('red', 'u1'), ('green', 'u1'),
                                      ('blue', 'u1')])
    new_vertex_element = PlyElement.describe(new_vertex_data, 'vertex')

    new_plydata = PlyData([new_vertex_element], text=plydata.text)

    scene_output_path = os.path.join(args.project_dir, args.output_data_path,
                                     args.experiment_name, args.scene_name)
    new_plydata.write(
        os.path.join(scene_output_path,
                     args.scene_name + '_visiualization.ply'))
