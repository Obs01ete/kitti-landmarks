"""Load Kitti samples"""

import numpy as np
from PIL import Image
from collections import namedtuple

import utils

def get_image(filename):
    """Function to read image files into arrays."""
    return np.asarray(Image.open(filename), np.uint8)


def get_image_pil(filename):
    """Function to read image files into arrays."""
    return Image.open(filename)


def get_velo_scan(filename):
    """Function to parse velodyne binary files into arrays."""
    scan = np.fromfile(filename, dtype=np.float32)
    return scan.reshape((-1, 4))


def get_calib(filename):
    """Function to parse calibration text files into a dictionary."""
    data = {}
    with open(filename, 'r') as f:
        lines = f.readlines()
        for i in range(7):
            line = lines[i]
            split_lines = line.split()
            key = split_lines[0]
            if key[-1] == ':':
                key = key[:-1]
            mat = np.array([float(x) for x in split_lines[1:]])
            if len(mat) == 12:
                mat = mat.reshape((3, 4))
            elif len(mat) == 9:
                mat = mat.reshape((3, 3))
            else:
                assert False
            data[key] = mat
    return data


def get_label(filename):
    """Function to parse label text files into a dictionary."""
    data = []
    with open(filename, 'r') as f:
        for line in f.readlines():
            values = line.split()
            assert len(values) == 15
            obj = {
                'type': str(values[0]),
                'truncated': float(values[1]),
                'occluded': int(values[2]),
                'alpha': float(values[3]),
                'bbox': np.array(values[4:8], dtype=float),
                'dimensions': np.array(values[8:11], dtype=float),
                'location': np.array(values[11:14], dtype=float),
                'rotation_y': float(values[14]),
            }
            data.append(obj)
    return data


def load_oxts_packets_and_poses(oxts_file):
    """Generator to read OXTS ground truth data.

       Poses are given in an East-North-Up coordinate system
       whose origin is the first GPS position.
    """
    # Scale for Mercator projection (from first lat value)
    scale = None
    # Origin of the global coordinate system (first GPS position)
    origin = None

    oxts = []

    with open(oxts_file, 'r') as f:
        for line in f.readlines():
            line = line.split()
            # Last five entries are flags and counts
            line[:-5] = [float(x) for x in line[:-5]]
            line[-5:] = [int(float(x)) for x in line[-5:]]

            packet = utils.OxtsPacket(*line)

            if scale is None:
                scale = np.cos(packet.lat * np.pi / 180.)

            R, t = utils.pose_from_oxts_packet(packet, scale)

            if origin is None:
                origin = t

            T_w_imu = utils.transform_from_rot_trans(R, t - origin)

            oxts.append(utils.OxtsData(packet, T_w_imu))

    return oxts
