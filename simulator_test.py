import math
import unittest
import numpy as np

import world
import simulator


class SimulatorTest(unittest.TestCase):

    def test_intersection(self):
        segments = np.array((((3, 0), (0, 3)),), dtype=np.float32)
        angle = 45 * math.pi / 180
        rays = np.array((((1, 0, math.cos(angle), math.sin(angle))),), dtype=np.float32)
        hit_segments = simulator.Simulator.closest_intersection(segments, rays)
        self.assertEqual(len(hit_segments), 1)
        hit_segment = hit_segments[0]
        self.assertAlmostEqual(hit_segment[1][0], 2.0)
        self.assertAlmostEqual(hit_segment[1][1], 1.0)
        pass

    def test_simulator(self):
        #w = world.LineWorld()
        w = world.CircleWorld(30)
        # num_rays = 360 // 10
        num_rays = 2 * 360
        lidar_inst = simulator.Lidar(num_rays)
        s = simulator.Simulator(w, lidar_inst)
        yaw = 10 * math.pi / 180
        #print(yaw)
        pose2d_xycs = np.array((40, 30, math.cos(yaw), math.sin(yaw)))
        render = s.render_pose(pose2d_xycs)
        self.assertEqual(len(render.shape), 2)
        self.assertEqual(render.shape[1], 2)
        #print(render)
        pass


if __name__ == "__main__":
    unittest.main()
