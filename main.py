import numpy as np
from sys import platform as sys_pf
import matplotlib
if sys_pf == 'darwin':
    matplotlib.use("TkAgg")
import matplotlib.colors
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tabulate import tabulate

from name_list_dataset import NameListDataset

def extend4(m):
    assert len(m.shape) == 2
    assert m.shape[1] == 3
    return np.concatenate((
            m,
            np.zeros((m.shape[0], 1), dtype=m.dtype) + 1.0
        ), axis=1)

def extend34(m):
    assert len(m.shape) == 2
    assert m.shape[0] == 3
    assert m.shape[1] == 4
    row_4 = np.expand_dims(np.array([0, 0, 0, 1], dtype=m.dtype), axis=0)
    return np.concatenate((m, row_4), axis=0)

def filter_by_magnitude(pc, thr):
    mag = pc[:, 3]
    mask = mag >= thr
    pcf = pc[mask, :]
    return pcf

def filter_by_cells(pc):
    assert len(pc.shape) == 2
    assert pc.shape[1] == 3

    int16_info = np.iinfo(np.int16)
    offset = int16_info.max - int16_info.min

    cell_size_ = 0.5
    cell_size = np.expand_dims(np.array([cell_size_]*3, dtype=np.float), 0)
    num_points_in_cell = 1

    cell_locs = np.floor(pc / cell_size)
    cell_locs = cell_locs.astype(np.int16).astype(np.uint16).astype(np.uint64)
    cell_loc_hashes = cell_locs[:, 0] + cell_locs[:, 1]*offset + cell_locs[:, 2]*offset*offset
    unique_cells = np.unique(cell_loc_hashes)
    counts = np.zeros(unique_cells.shape, dtype=np.int32)
    select_mask = np.zeros(cell_loc_hashes.shape[0], dtype=np.bool)
    for i, u in enumerate(unique_cells):
        (idxs_same_cell, ) = np.where(cell_loc_hashes == u)
        counts[i] = len(idxs_same_cell)
        idxs_same_cell = np.random.choice(idxs_same_cell, num_points_in_cell)
        select_mask[idxs_same_cell] = True
    mc = np.mean(counts)
    pcf = pc[select_mask]
    # print(len(pc), len(pcf), len(pcf)/len(pc))
    return pcf

def print_hist(np_vector):
    hist = np.histogram(np_vector, bins=np.arange(-5, 5, 0.1))
    hist = np.vstack((np.pad(hist[0], (0, 1), 'constant'), hist[1]))
    # print(hist)
    table = tabulate(hist.transpose((1, 0)), headers=['N', 'bin'], tablefmt="fancy_grid")
    print(table)


def main():
    np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

    dataset = NameListDataset()

    interactive = False
    preview = False

    lidar_radius = 60 # 80

    if not interactive:
        plt.ioff()

    for i_seq, sequence in enumerate(dataset):
        if preview and i_seq >= 4:
            break

        print('Processing sequence {}'.format(i_seq))
        fig = plt.figure(figsize=(10, 10), dpi=150)
        fig.tight_layout()

        seq_len = len(sequence)
        if preview:
            seq_len = min(seq_len, 10)
        else:
            seq_len = min(seq_len, 50)

        def calc_centers_ltrb():
            locs_list = []
            for i_frame in range(seq_len):
                frame = sequence[i_frame]
                T_w_imu = frame[3].T_w_imu
                # print(T_w_imu)
                loc = T_w_imu[0:2, 3]
                # print(loc)
                locs_list.append(loc)
            locs = np.stack(locs_list, axis=0)
            area_ltrb = np.array((
                np.min(locs[:, 0]),
                np.min(locs[:, 1]),
                np.max(locs[:, 0]),
                np.max(locs[:, 1]),
            ))
            return area_ltrb

        centers_bb_ltrb = calc_centers_ltrb()
        bb_ltrb = centers_bb_ltrb.copy()
        bb_ltrb[0:2] -= lidar_radius
        bb_ltrb[2:4] += lidar_radius

        def get_next_frame_contents():
            total_cloud_in_world = np.zeros((0, 3), dtype=np.float64)

            for i_frame in range(seq_len):
                frame = sequence[i_frame]
                print('Processing frame {}'.format(i_frame))
                frame_point_cloud = frame[1]
                frame_point_cloud = filter_by_magnitude(frame_point_cloud, 0.01)
                # print_hist(frame_point_cloud[:, 2])
                point_cloud_pos_3 = frame_point_cloud[:, :3]
                point_cloud_pos_4 = extend4(point_cloud_pos_3)
                point_cloud_mag = frame_point_cloud[:, 3:4]
                imu2velo_34 = frame[2]['Tr_imu_velo']
                # print(imu2velo_34)
                imu2velo_44 = extend34(imu2velo_34)
                velo2imu_44 = np.linalg.inv(imu2velo_44)
                # print(velo2imu_44)
                # imug is the point on the ground right under IMU
                velo2imug_44 = velo2imu_44.copy(); velo2imug_44[2, 3] += 0.93
                # print(velo2imug_44)
                velo2imug_34 = velo2imug_44[:3, :]
                point_cloud_in_imug = np.matmul(point_cloud_pos_4, velo2imug_34.transpose(1, 0))
                # print_hist(point_cloud_in_imug[:, 2])
                T_w_imu = frame[3].T_w_imu
                point_cloud_in_imug_4 = extend4(point_cloud_in_imug)
                point_cloud_in_world_4 = np.matmul(point_cloud_in_imug_4, T_w_imu.transpose(1, 0))
                point_cloud_in_world = point_cloud_in_world_4[:, :3]
                total_cloud_in_world = np.concatenate((total_cloud_in_world, point_cloud_in_world), axis=0)
                total_cloud_in_world = filter_by_cells(total_cloud_in_world)
                yield total_cloud_in_world

        # gen_next_frame_contents = get_next_frame_contents()
        def updatefig(pc):
            # pc = next(gen_next_frame_contents)
            fig.clear()
            # print_hist(pc[:, 2])
            norm = matplotlib.colors.Normalize(-0.2, 2.0, clip=True)
            plt.scatter(
                pc[:, 0],
                pc[:, 1],
                c=pc[:, 2],
                s=1.0,
                norm=norm,
                marker=',',
                lw=0)
            plt.xlim(bb_ltrb[0], bb_ltrb[2])
            plt.ylim(bb_ltrb[1], bb_ltrb[3])
            # plt.axis('equal', datalim=[v for v in bb_ltrb])
            plt.axes().set_aspect('equal', 'datalim')
            plt.subplots_adjust(left=0.05, top=0.95, right=0.95, bottom=0.05)
            if interactive:
                plt.show()
            else:
                plt.draw()

        anim = animation.FuncAnimation(fig, updatefig, frames=get_next_frame_contents)
        anim.save("{}_{:02d}.mp4".format("prev" if preview else "vis", i_seq), fps=10)

        del fig

    pass

if __name__ == "__main__":
    main()
