import os
import glob
import numpy as np
import networkx as nx

from utils.dataset_utils import load_state, load_pose, get_rel_pose, pose_identity_like


class Sequence(object):
    def __init__(self, seq_path, num_points=4096, cano_idx=0):
        self.seq_path = seq_path
        self.cat = seq_path.split("/")[-1]
        self.num_points = num_points
        self.cano_idx = cano_idx

        self.pose_template = "pose_{}.pkl"
        self.state_template = "state_{}.pkl"
        self.novel_pose_template = "novel_pose_{}.pkl"
        self.pc_path_list, self.pose_list = [], []
        self.novel_pose_list = []

        pose_files = glob.glob(os.path.join(self.seq_path, "pose_*.pkl"))
        pose_files = sorted(pose_files, key=lambda file_name: int(file_name.split("/")[-1].split(".")[0].split("_")[-1]))

        novel_pose_files = glob.glob(os.path.join(self.seq_path, "novel_pose_*.pkl"))
        novel_pose_files = sorted(novel_pose_files, key=lambda file_name: int(file_name.split("/")[-1].split(".")[0].split("_")[-1]))

        self.pc_path_list.append(os.path.join(self.seq_path, self.state_template.format(0)))
        for pose_file in pose_files:
            state_idx = pose_file.split(".")[0].split("_")[-1]
            pc_path = os.path.join(self.seq_path, self.state_template.format(state_idx))
            pose_path = os.path.join(self.seq_path, self.pose_template.format(state_idx))
            self.pc_path_list.append(pc_path)
            pose = load_pose(pose_path)
            self.pose_list.append(pose)

        for novel_pose_file in novel_pose_files:
            novel_state_idx = novel_pose_file.split(".")[0].split("_")[-1]
            novel_pose_path = os.path.join(self.seq_path, self.novel_pose_template.format(novel_state_idx))
            novel_pose = load_pose(novel_pose_path)
            self.novel_pose_list.append(novel_pose)

        self.pose_list.insert(0, pose_identity_like(self.pose_list[0]))

        assert len(self.pc_path_list) == len(self.pose_list)

    def __len__(self):
        return 1

    def __getitem__(self, item):
        complete_pc_list, complete_pc_transform_list, complete_gt_part_list = [], [], []
        for i in range(len(self.pc_path_list)):
            pc_path = self.pc_path_list[i]
            pc, part = load_state(pc_path)
            num_pc = len(pc)
            if self.num_points < num_pc:
                choose_idx = np.arange(self.num_points)
                pc = pc[choose_idx, :]
                part = part[choose_idx]
            complete_pc_list.append(pc)
            complete_gt_part_list.append(part)

        complete_pc_list = np.stack(complete_pc_list).astype('float32')
        complete_gt_part_list = np.stack(complete_gt_part_list)

        cano_pc, gt_cano_part = complete_pc_list[self.cano_idx], complete_gt_part_list[self.cano_idx]
        src_pose = self.pose_list[self.cano_idx]
        unique_part_ids = list(set(complete_gt_part_list[0].tolist()))
        gt_pose_list = []
        for i in range(len(self.pose_list)):
            tgt_pose = self.pose_list[i]
            pc_transform = np.empty_like(cano_pc)
            pose_src2tgt = get_rel_pose(src_pose, tgt_pose)
            pose_list = []
            for part_id in unique_part_ids:
                part_gt_pose = pose_src2tgt[part_id]
                pose_list.append(part_gt_pose)
                pc_idx = gt_cano_part == part_id
                points = cano_pc[pc_idx, :]
                points_homo = np.concatenate([points, np.ones((points.shape[0], 1), dtype=float)], axis=1)
                new_points = (points_homo @ part_gt_pose.T)[:, :3]
                pc_transform[pc_idx, :] = new_points
            pose_list = np.stack(pose_list).astype('float32')
            gt_pose_list.append(pose_list)
            complete_pc_transform_list.append(pc_transform)
        complete_pc_transform_list = np.stack(complete_pc_transform_list).astype('float32')
        gt_flow_list = complete_pc_transform_list[1:] - complete_pc_transform_list[:-1]
        gt_pose_list = np.stack(gt_pose_list).astype('float32')

        pc_list = np.concatenate((complete_pc_list[:self.cano_idx, :], complete_pc_list[self.cano_idx+1:, :]), axis=0)
        pc_transform_list = np.concatenate(
            (complete_pc_transform_list[:self.cano_idx, :], complete_pc_transform_list[self.cano_idx+1:, :]), axis=0)
        sample = {'cano_pc': cano_pc, 'gt_cano_part': gt_cano_part,
                  'gt_flow_list': gt_flow_list, 'gt_pc_list': pc_transform_list,
                  'pc_list': pc_list,
                  'gt_pose_list': gt_pose_list, # complete pose list
                  'complete_pc_list': complete_pc_list,
                  'complete_gt_pc_list': complete_pc_transform_list,
                  'complete_gt_part_list': complete_gt_part_list}
        return sample
