import json
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from msync.utils.motion_util import Isometry


def compute_unalign(base_pc, base_segms, base_cam, base_motions, dest_cam, dest_motions):
    final_pc = np.empty_like(base_pc)
    n_parts = len(base_motions)
    pose_list = []
    for part_id in range(n_parts):
        part_mask = np.where(base_segms == (part_id))[0]
        part_pose = (dest_cam.inv().dot(dest_motions[part_id]).dot(
            base_motions[part_id].inv()).dot(base_cam))
        part_pc = part_pose @ base_pc[part_mask]
        pose_list.append(part_pose.matrix)
        final_pc[part_mask] = part_pc
    pose_list = np.stack(pose_list).astype('float32')
    return final_pc, pose_list


class Sapien(Dataset):
    def __init__(self, base_folder, cano_idx=0):
        self.base_folder = Path(base_folder)
        with (self.base_folder / "meta.json").open() as f:
            self.meta = json.load(f)
        self.data_ids = self.meta['test']
        self.cano_idx = cano_idx

    def __len__(self):
        return len(self.data_ids)

    def _get_item(self, idx):
        data_path = self.base_folder / "data" / ("%06d.npz" % self.data_ids[idx])
        datum = np.load(data_path, allow_pickle=True)

        raw_pc = datum['pc'].astype(np.float32)
        raw_segm = datum['segm']
        raw_trans = datum['trans'].item()

        return raw_pc, raw_segm, raw_trans

    def __getitem__(self, data_id):

        def get_view_motions(view_id):
            return [Isometry.from_matrix(trans_dict[t][view_id]) for t in range(1, n_parts + 1)]

        idx = data_id
        pcs, segms, trans_dict = self._get_item(idx)
        segms = segms - 1  # segmentation part idx start from 0
        n_parts = len(trans_dict) - 1
        n_views = pcs.shape[0]
        assert segms.shape[0] == n_views

        cano_pc, gt_cano_part = pcs[self.cano_idx], segms[self.cano_idx]
        complete_gt_part_list = segms
        complete_pc_transform_list, gt_pose_list = [], []
        base_cam = Isometry.from_matrix(trans_dict['cam'][self.cano_idx])
        base_motions = get_view_motions(self.cano_idx)
        full_flow = []
        
        complete_pc_list = pcs
        for i in range(n_views):
            gt_pc, pose_list = compute_unalign(cano_pc, gt_cano_part, base_cam, base_motions,
                                                Isometry.from_matrix(trans_dict['cam'][i]), get_view_motions(i))
            complete_pc_transform_list.append(gt_pc)
            gt_pose_list.append(pose_list)
        for view_i in range(n_views):
            for view_j in range(n_views):
                pc, pose_list = compute_unalign(pcs[view_i], segms[view_i],
                                                Isometry.from_matrix(trans_dict['cam'][view_i]),
                                                get_view_motions(view_i),
                                                Isometry.from_matrix(trans_dict['cam'][view_j]),
                                                get_view_motions(view_j))
                flow_ij = pc - pcs[view_i]
                full_flow.append(flow_ij)

        complete_pc_list = np.stack(complete_pc_list).astype('float32')
        complete_pc_transform_list = np.stack(complete_pc_transform_list).astype('float32')

        gt_pose_list = np.stack(gt_pose_list).astype('float32')
        gt_flow_list = complete_pc_transform_list[1:] - complete_pc_transform_list[:-1]

        pc_list = np.concatenate((complete_pc_list[:self.cano_idx, :], complete_pc_list[self.cano_idx + 1:, :]), axis=0)
        pc_transform_list = np.concatenate((complete_pc_transform_list[:self.cano_idx, :], complete_pc_transform_list[self.cano_idx + 1:, :]), axis=0)

        full_flow = np.stack(full_flow, axis=0)
        sample = {'cano_pc': cano_pc, 'gt_cano_part': gt_cano_part,
                  'gt_flow_list': gt_flow_list, 'gt_pc_list': pc_transform_list, 'gt_pose_list': gt_pose_list,
                  'pc_list': pc_list,
                  'complete_pc_list': complete_pc_list,
                  'complete_gt_pc_list': complete_pc_transform_list,
                  'complete_gt_part_list': complete_gt_part_list, 'gt_full_flow': full_flow}
        return sample