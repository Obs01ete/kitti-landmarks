import os
import random
from collections import OrderedDict

import kitti_sample


class ElementList(object):
    def __init__(self, parent, method, lst):
        super().__init__()
        self.parent = parent
        self.method = method
        self.lst = lst
    def __getitem__(self, item):
        return getattr(self.parent, self.method)(self.lst, item)
    def __len__(self):
        return len(self.lst[1])


def list_dir(path, ext):
    return list(sorted([
        os.path.splitext(f)[0]
        for f in os.listdir(path)
        if os.path.isfile(os.path.join(path, f)) and os.path.splitext(f)[1] == ext
    ]))


class NameListDataset(object):
    """Class to load Kitti tracking dataset"""

    def __init__(
            self,
            data_path = 'data/tracking/training/',
        ):
        """
        """

        self._is_pil_image = True
        self.data_path = data_path
        self.image_path = os.path.join(self.data_path, 'image_2')
        self.velo_path = os.path.join(self.data_path, 'velodyne')
        self.calib_path = os.path.join(self.data_path, 'calib')
        self.label_path = os.path.join(self.data_path, 'label_2')
        self.imu_path = os.path.join(self.data_path, 'oxts')

        self.dataset_list = list_dir(self.imu_path, '.txt')

        oxts = [
            kitti_sample.load_oxts_packets_and_poses(os.path.join(self.imu_path, name+'.txt'))
            for name in self.dataset_list
        ]

        frame_name_lists = []
        for seq_name in self.dataset_list:
            dir = os.path.join(self.velo_path, seq_name)
            frame_name_list = list_dir(dir, '.bin')
            frame_name_lists.append(frame_name_list)

        self.sequences = []
        for name, frame_name_list, oxts in zip(self.dataset_list, frame_name_lists, oxts):
            self.sequences.append((name, frame_name_list, oxts))

        pass

    @staticmethod
    def getLabelmap():
        return ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc']

    def list_all_sequences(self):
        """Scan over all samples in the dataset"""

        print('Start generation of a file list')

        names = []
        for root, _, fnames in sorted(os.walk(self.velo_path)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if os.path.isdir(path):
                    nameonly = os.path.splitext(fname)[0]
                    names.append(nameonly)

        print('End generation of a file list')

        return names

    def __getitem__(self, sequence_index):
        """
        Args:
            sequence_index (int): Index of a sequence

        Returns:
        """
        return ElementList(self, '_get_frame', self.sequences[sequence_index])

    def _get_frame(self, frame_list, frame_index):
        """
        Args:
            frame_list: ?
            frame_index (int): Index of a frame

        Returns:
        """

        sequence_name, frame_name_list, oxts = frame_list

        frame_name = frame_name_list[frame_index]

        _, velo, calib, _ = self._getitem(
            sequence_name,
            frame_name,
            load_image=False, load_velodyne=True, load_calib=True, load_label=False)

        imu = oxts[int(frame_name)]

        return sequence_name, velo, calib, imu

    def _getitem(
            self, sequence_name, frame_name,
            load_image=True, load_velodyne=False, load_calib=True, load_label=True):

        image = None
        if load_image:
            path = os.path.join(self.image_path, sequence_name, frame_name+'.png')
            if self._is_pil_image:
                image = kitti_sample.get_image_pil(path)
            else:
                image = kitti_sample.get_image(path)

        velo = None
        if load_velodyne:
            path = os.path.join(self.velo_path, sequence_name, frame_name+'.bin')
            velo = kitti_sample.get_velo_scan(path)

        calib = None
        if load_calib:
            path = os.path.join(self.calib_path, sequence_name+'.txt')
            calib = kitti_sample.get_calib(path)

        label = None
        if load_label:
            path = os.path.join(self.label_path, sequence_name+'.txt')
            # TODO filter only boxes for frame_name
            label = kitti_sample.get_label(path)

        return image, velo, calib, label

    def __len__(self):
        """
        Args:
            none

        Returns:
            int: number of images in the dataset
        """

        return len(self.dataset_list)

