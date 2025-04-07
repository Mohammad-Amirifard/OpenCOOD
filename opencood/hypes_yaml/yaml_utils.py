# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>, Hao Xiang <haxiang@g.ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib


import re
import yaml
import os
import math
import numpy as np


def load_yaml(file, opt=None):
    """
    Load a YAML configuration file and return its contents as a dictionary.

    Parameters
    ----------
    file : str
        Path to the YAML configuration file.

    opt : argparse.Namespace, optional. It is like --hypes_yaml
        Parsed command-line arguments. If provided and contains a 'model_dir',
        the configuration will be loaded from 'config.yaml' within that directory.

    Returns
    -------
    param : dict
        Dictionary containing parameters defined in the YAML file.
    
    done
    """
    # Override the file path if a model directory is specified in the options
    if opt and opt.model_dir:
        file = os.path.join(opt.model_dir, 'config.yaml')

    # Open the YAML file for reading
    stream = open(file, 'r')

    # Configure the YAML loader with custom float resolver to handle scientific notation and special float values
    loader = yaml.Loader
    loader.add_implicit_resolver(
        u'tag:yaml.org,2002:float',
        re.compile(u'''^(?:
         [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$''', re.X),
        list(u'-+0123456789.')
    )

    # Load the YAML contents
    param = yaml.load(stream, Loader=loader)
    

    # Dynamically apply a custom parser if specified in the YAML under the "yaml_parser" key
    if "yaml_parser" in param:
        param = eval(param["yaml_parser"])(param)
        
    print(f'Configuration parameters from {opt.hypes_yaml} have been successfully loaded.')
    return param



def load_voxel_params(param):
    """
    Based on the lidar range and resolution of voxel, calcuate the anchor box
    and target resolution. It means we create voxels/grids on the cav_lidar_range.

    Parameters
    ----------
    param : dict
        Original loaded parameter dictionary.

    Returns
    -------
    param : dict
        Modified parameter dictionary with new attribute `anchor_args[W][H][L]`
    """
    anchor_args = param['postprocess']['anchor_args']
    cav_lidar_range = anchor_args['cav_lidar_range'] # Cav Lidar range : [-140.8, -40, -3, 140.8, 40, 1]
    voxel_size = param['preprocess']['args']['voxel_size'] # is somthing like [0.4, 0.4, 4]

    vw = voxel_size[0]
    vh = voxel_size[1]
    vd = voxel_size[2]

    # Add a 3 new key-values to the anchor_args writen in yaml file
    anchor_args['vw'] = vw
    anchor_args['vh'] = vh
    anchor_args['vd'] = vd
    
    # Here you can see the cav_lidar_range includes  140-(-140)/.4 = 704 voxels in Width,
    # 40-(-40)/.4 = 200 voxels in H, and 1-(-3)/4=1 voxel in z direction
    # Add another 3 new key-values to the anchor_args writen in yaml file
    anchor_args['W'] = int((cav_lidar_range[3] - cav_lidar_range[0]) / vw)
    anchor_args['H'] = int((cav_lidar_range[4] - cav_lidar_range[1]) / vh)
    anchor_args['D'] = int((cav_lidar_range[5] - cav_lidar_range[2]) / vd)

    # Now, Update the anchor_args written in the yamle file to include new values.
    param['postprocess'].update({'anchor_args': anchor_args})

    # sometimes we just want to visualize the data without implementing model
    if 'model' in param:
        param['model']['args']['W'] = anchor_args['W']
        param['model']['args']['H'] = anchor_args['H']
        param['model']['args']['D'] = anchor_args['D']

    return param

# The only difference between above and bellow function is the additional added grid_size [704,200,1] key-value to the dictionary.
def load_point_pillar_params(param):
    """
    Based on the lidar range and resolution of voxel, calcuate the anchor box
    and target resolution.

    Parameters
    ----------
    param : dict
        Original loaded parameter dictionary.

    Returns
    -------
    param : dict
        Modified parameter dictionary with new attribute.
    """
    cav_lidar_range = param['preprocess']['cav_lidar_range'] # Cav Lidar range : [-140.8, -40, -3, 140.8, 40, 1]
    voxel_size = param['preprocess']['args']['voxel_size'] # is somthing like [0.4, 0.4, 4]

    grid_size = (np.array(cav_lidar_range[3:6]) - np.array(
        cav_lidar_range[0:3])) / \
                np.array(voxel_size)
    grid_size = np.round(grid_size).astype(np.int64)
    # Grid-size is somthing like [704,200,1] which is added to the parm dict.
    param['model']['args']['point_pillar_scatter']['grid_size'] = grid_size

    anchor_args = param['postprocess']['anchor_args']

    vw = voxel_size[0]
    vh = voxel_size[1]
    vd = voxel_size[2]

    anchor_args['vw'] = vw
    anchor_args['vh'] = vh
    anchor_args['vd'] = vd

    anchor_args['W'] = math.ceil((cav_lidar_range[3] - cav_lidar_range[0]) / vw)
    anchor_args['H'] = math.ceil((cav_lidar_range[4] - cav_lidar_range[1]) / vh)
    anchor_args['D'] = math.ceil((cav_lidar_range[5] - cav_lidar_range[2]) / vd)

    param['postprocess'].update({'anchor_args': anchor_args})

    return param

#// Commented this since I didn't use it anywhere.
# def load_second_params(param):
    """
    Based on the lidar range and resolution of voxel, calcuate the anchor box
    and target resolution.

    Parameters
    ----------
    param : dict
        Original loaded parameter dictionary.

    Returns
    -------
    param : dict
        Modified parameter dictionary with new attribute.
    """
    cav_lidar_range = param['preprocess']['cav_lidar_range']
    voxel_size = param['preprocess']['args']['voxel_size']

    grid_size = (np.array(cav_lidar_range[3:6]) - np.array(
        cav_lidar_range[0:3])) / \
                np.array(voxel_size)
    grid_size = np.round(grid_size).astype(np.int64)
    param['model']['args']['grid_size'] = grid_size

    anchor_args = param['postprocess']['anchor_args']

    vw = voxel_size[0]
    vh = voxel_size[1]
    vd = voxel_size[2]

    anchor_args['vw'] = vw
    anchor_args['vh'] = vh
    anchor_args['vd'] = vd

    anchor_args['W'] = int((cav_lidar_range[3] - cav_lidar_range[0]) / vw)
    anchor_args['H'] = int((cav_lidar_range[4] - cav_lidar_range[1]) / vh)
    anchor_args['D'] = int((cav_lidar_range[5] - cav_lidar_range[2]) / vd)

    param['postprocess'].update({'anchor_args': anchor_args})

    return param


# The following def is used just only for BevPreprocessor.
#// Commented this since I didn't use it anywhere.
#def load_bev_params(param):
    """
    Load bev related geometry parameters s.t. boundary, resolutions, input
    shape, target shape etc.

    Parameters
    ----------
    param : dict
        Original loaded parameter dictionary.

    Returns
    -------
    param : dict
        Modified parameter dictionary with new attribute `geometry_param`.

    """
    res = param["preprocess"]["args"]["res"]
    L1, W1, H1, L2, W2, H2 = param["preprocess"]["cav_lidar_range"] # [-160, -40, -3, 160, 40, 1]
    downsample_rate = param["preprocess"]["args"]["downsample_rate"]

    def f(low, high, r):
        return int((high - low) / r)

    input_shape = (
        int((f(L1, L2, res))),
        int((f(W1, W2, res))),
        int((f(H1, H2, res)) + 1)
    )
    label_shape = (
        int(input_shape[0] / downsample_rate),
        int(input_shape[1] / downsample_rate),
        7
    )
    geometry_param = {
        'L1': L1,
        'L2': L2,
        'W1': W1,
        'W2': W2,
        'H1': H1,
        'H2': H2,
        "downsample_rate": downsample_rate,
        "input_shape": input_shape,
        "label_shape": label_shape,
        "res": res
    }
    param["preprocess"]["geometry_param"] = geometry_param
    param["postprocess"]["geometry_param"] = geometry_param
    param["model"]["args"]["geometry_param"] = geometry_param
    return param


def save_yaml(data, save_name):
    """
    Save the dictionary into a yaml file.

    Parameters
    ----------
    data : dict
        The dictionary contains all data.

    save_name : string
        Full path of the output yaml file.
    """

    with open(save_name, 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)


def save_yaml_wo_overwriting(data, save_name):
    """
    Save the yaml file without overwriting the existing one.

    Parameters
    ----------
    data : dict
        The dictionary contains all data.

    save_name : string
        Full path of the output yaml file.
    """
    if os.path.exists(save_name):
        prev_data = load_yaml(save_name)
        data = {**data, **prev_data}

    save_yaml(data, save_name)