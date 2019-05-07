import math
import numpy as np
import numpy.ma as ma

from sys import platform as sys_pf
import matplotlib
if sys_pf == 'darwin':
    matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib import collections as mc


# World coordinate system
#
# Y ^
#   |
#   |
#   |
#   |
#   |
# Z .----------> X
#
# Right triplet of axes

# Vehicle coordinates
#
# Same as world, car heads towards X
# Based on Kitti lidar setup


class Canvas:

    def __init__(self, world_size, pixels_per_meter):
        self.world_size = world_size
        self.pixels_per_meter = pixels_per_meter

        dpi = 100
        figsize_pixels = world_size * pixels_per_meter
        figsize_inches = figsize_pixels / dpi

        #plt.ioff()

        self._fig = plt.figure(figsize=figsize_inches.tolist(), dpi=dpi)
        self._ax = plt.axes()
        self._ax.axis([0, world_size[0], 0, world_size[1]])

        pass

    def draw_segment_list(self, segment_list, color='b'):
        assert len(segment_list.shape) == 3
        assert segment_list.shape[1] == 2
        assert segment_list.shape[2] == 2
        lc = mc.LineCollection(segment_list, color=color, linewidths=1)
        self._ax.add_collection(lc)
        pass

    def draw_points(self, points, color='r', marker='o'):
        x, y = points[:, 0], points[:, 1]
        self._ax.scatter(x, y, s=10.0, c=color, alpha=0.5)
        self._ax.grid()

    def show(self):
        plt.show()
        pass



class Simulator:
    def __init__(self):
        np.random.seed(17)
        self.world_size = np.array((120, 100), dtype=np.float32)
        num_segments = 10
        segments = np.random.rand(num_segments, 2, 2).astype(np.float32)
        self._segments = segments * self.world_size[np.newaxis, np.newaxis, :]
        pass

    def render_pose(self, pose2d_xycs: np.ndarray) -> np.ndarray:
        """

        :param pose2d_xycs: pose of a car in world coordinate system (x, y, cos(a), sin(a))
        :return:
        """

        num_rays = 360 // 10
        cos_a, sin_a = pose2d_xycs[2:4]
        angle = math.atan2(sin_a, cos_a)
        angles = np.arange(angle, angle+2*math.pi, 2*math.pi/num_rays, dtype=np.float32)
        coss = np.expand_dims(np.cos(angles), 1)
        sins = np.expand_dims(np.sin(angles), 1)
        centers = np.tile(pose2d_xycs[0:2], (num_rays, 1))
        rays = np.concatenate((centers, coss, sins), axis=1)
        hit_segments = self.closest_intersection(self._segments, rays)
        self._visualize(hit_segments)
        hit_points = hit_segments[:, 1, :]
        return hit_points

    @staticmethod
    def closest_intersection(segments: np.ndarray, rays: np.ndarray) -> np.ndarray:
        """

        :param segments: [num_segments, 2=points in a segment, 2=x,y]
        :param rays: [num_rays, 4=x,y,cos,sin]
        :return: hit_segments [some_rays, 2=points in a segment, 2=x,y]
        """

        p1 = segments[:, 0, :]
        pd = segments[:, 1, :] - segments[:, 0, :]
        # alpha[0] scans from p1 to p2 as 0 to 1

        v1 = rays[:, 0:2]
        vd = rays[:, 2:4]
        # alpha[1] scans from v1 to v2 as 0 to 1

        def det(a, b):
            if len(a.shape) == 1:
                a = np.expand_dims(a, 0)
            if len(b.shape) == 1:
                b = np.expand_dims(b, 0)
            result = a[:, 0]*b[:, 1] - a[:, 1]*b[:, 0]
            return result

        alphas = np.zeros((len(segments), len(rays)), dtype=np.float32)
        betas = np.zeros((len(segments), len(rays)), dtype=np.float32)
        for i_s in range(len(segments)):
            alpha = (det(p1[i_s], vd) - det(v1, vd)) / det(vd, pd[i_s])
            beta = (det(v1, pd[i_s]) - det(p1[i_s], pd[i_s])) / det(pd[i_s], vd)
            alphas[i_s, :] = alpha
            betas[i_s, :] = beta
            pass

        has_intersection = np.logical_and(
            np.logical_and(
                alphas >= 0, # segment is intersected
                alphas <= 1), # segment is intersected
            betas >= 0) # positive ray
        #ray_has_intersection = np.any(has_intersection, axis=0)
        masked_betas = ma.array(betas, mask=np.logical_not(has_intersection))
        min_betas = masked_betas.min(axis=0, fill_value=float('inf'))

        hit_segment_list = [
            np.array((
                (ray[0], ray[1]),
                (ray[0]+ray[2]*min_beta, ray[1]+ray[3]*min_beta)
            ), dtype=np.float32)
            for ray, min_beta, invalid in zip(rays, min_betas, ma.getmask(min_betas)) if not invalid]
        hit_segments = np.array(hit_segment_list)

        return hit_segments

    def _visualize(self, hit_segments):
        pixels_per_meter = 5
        canvas = Canvas(self.world_size, pixels_per_meter)
        canvas.draw_segment_list(self._segments)
        canvas.draw_segment_list(hit_segments, color='g')
        canvas.draw_points(hit_segments[:, 1, :])
        canvas.show()

