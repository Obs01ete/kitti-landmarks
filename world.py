import math
import numpy as np


class World:
    def __init__(self):
        self._segments = None
        self._world_size = None

    def get_segments(self):
        return self._segments

    def get_world_size(self):
        return self._world_size


class LineWorld(World):
    def __init__(self):
        super().__init__()

        np.random.seed(17)
        self._world_size = np.array((120, 100), dtype=np.float32)
        num_segments = 10
        segments = np.random.rand(num_segments, 2, 2).astype(np.float32)
        self._segments = segments * self._world_size[np.newaxis, np.newaxis, :]


class CircleWorld(World):
    def __init__(self, num_circles:int=10):
        super().__init__()

        np.random.seed(17)
        self._world_size = np.array((120, 100), dtype=np.float32)
        circle_fineness = 16
        radii = 0.8 * np.random.rand(num_circles) + 0.2
        centers = np.random.rand(num_circles, 2).astype(np.float32) * \
                    self._world_size[np.newaxis, :]
        phase_offsets = np.random.rand(num_circles)
        segments = np.zeros((num_circles, circle_fineness, 2, 2), dtype=np.float32)
        for c in range(num_circles):
            angles = (np.arange(circle_fineness) + phase_offsets[c]) / circle_fineness * 2*math.pi
            radius_vectors = np.stack((np.cos(angles), np.sin(angles)), axis=1)
            locs_start = centers[c] + radii[c] * radius_vectors
            locs_end = np.concatenate((locs_start[-1:], locs_start[:-1]), axis=0)
            circle_segments = np.stack((locs_start, locs_end), axis=1)
            segments[c, ...] = circle_segments
        self._segments = segments.reshape((-1, *segments.shape[2:4]))
        self._radii = radii
        self._centers = centers
        pass

    def get_ground_truth(self):
        return self._centers, self._radii
