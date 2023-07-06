import argparse
import yaml
import os
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn as nn

msync_dir = os.path.join(os.path.dirname("__file__"), 'msync')
if msync_dir not in sys.path:
    sys.path.insert(0, msync_dir)

from msync.models.flow_net import FlowNet
from msync.models.conf_net import ConfNet, get_network_input
from msync.models.mot_net import MotNet

from utils.model_utils import knn_query, compute_pc_transform
from screw_se3.geo_utils import inverse_transformation


class TestTimeFullNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.n_iter = 4
        self.rigid_n_iter = 1
        self.t = 0.01
        self.nsample_motion = 256
        self.mcut_thres = args.alpha
        assert self.rigid_n_iter <= self.n_iter

        self.flow_net = FlowNet()
        self.conf_net = ConfNet()
        self.mot_net = MotNet()


def compute_rel_trans(complete_trans_list, src_idx):
    # complete_trans_list: trans_cano2tgt
    # target: src2tgt = (cano2tgt) @ (src2cano) = (cano2tgt) @ (cano2src)^-1
    T, P = complete_trans_list.shape[:2]
    trans_cano2src = complete_trans_list[src_idx:src_idx+1, :].expand(T, P, 4, 4).reshape(-1, 4, 4)
    complete_trans_list = complete_trans_list.reshape(-1, 4, 4)
    trans_src2tgt = torch.bmm(complete_trans_list, inverse_transformation(trans_cano2src))
    trans_src2tgt = trans_src2tgt.reshape(T, P, 4, 4)
    return trans_src2tgt # (T, P, 4, 4)


def compute_full_flow(complete_pc_list, complete_seg_list, complete_trans_list):
    n_views = complete_pc_list.shape[0]
    full_flow = []
    for view_i in range(n_views):
        cano_pc_i = complete_pc_list[view_i]
        cano_part_i = complete_seg_list[view_i]
        trans_i = compute_rel_trans(complete_trans_list, view_i)
        complete_pc_i = compute_pc_transform(cano_pc_i, trans_i, cano_part_i)
        for view_j in range(n_views):
            flow = complete_pc_i[view_j] - complete_pc_i[view_i]
            full_flow.append(flow)
    full_flow = torch.stack(full_flow, dim=0)
    return full_flow


def eval_flow(full_flow, gt_full_flow):
    all_epe3d = []
    n_views = int(np.sqrt(full_flow.shape[0]))
    for view_i in range(n_views):
        for view_j in range(n_views):
            if view_i == view_j:
                continue
            pd_f = full_flow[view_j + view_i * n_views, :]
            gt_f = gt_full_flow[view_j + view_i * n_views, :]
            epe3d = torch.norm(pd_f - gt_f, dim=-1).mean().item()
            all_epe3d.append(epe3d)
    all_epe3d = np.array(all_epe3d)
    return all_epe3d


def load_model(config_path="msync/config/articulated-full.yaml",
               model_path="msync/ckpt/articulated-full/best.pth.tar"):
    with Path(config_path).open() as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)
    model_args = SimpleNamespace()
    for ckey, cvalue in configs.items():
        model_args.__dict__[ckey] = cvalue
    assert model_args.type == "full", "Your config file must be of type 'full'."
    model = TestTimeFullNet(model_args)
    model.load_state_dict(torch.load(model_path)['model_state'])
    model.eval()
    return model


def compute_flow_list(model, pc_list):
    pred_flow_list, pred_conf_list = [], []
    with torch.no_grad():
        for idx in range(len(pc_list) - 1):
            src_pc, tgt_pc = pc_list[idx:idx + 1], pc_list[idx + 1:idx + 2]
            flow, _, _, _, _ = model.flow_net.forward(src_pc, tgt_pc, src_pc, tgt_pc)
            flow = flow[0].transpose(-1, -2)
            pred_flow_list.append(flow.squeeze(dim=0))
            flow = pred_flow_list[idx]
            _, weight = model.conf_net(get_network_input(src_pc, tgt_pc, flow))
            weight.sigmoid_()
            pred_conf_list.append(weight.squeeze(dim=0))
    pred_flow_list = torch.stack(pred_flow_list)
    pred_conf_list = torch.stack(pred_conf_list)
    return pred_flow_list, pred_conf_list


def seg_propagation_list(query_pc_list, ref_pc_list, ref_seg, knn):
    prop_seg_list = []
    for (pc, pc_trans) in zip(query_pc_list, ref_pc_list):
        prop_seg = knn_query(pc, pc_trans, ref_seg, knn)
        prop_seg_list.append(prop_seg)
    prop_seg_list = torch.stack(prop_seg_list)
    return prop_seg_list


def compute_pc_transform_list(pc_list, part_list, pose_list):
    assert len(pc_list) == len(pose_list) == len(part_list)
    pc_transform_list = []
    for idx, (pc_src, pc_tgt, part_src, part_tgt) in enumerate(zip(pc_list[:-1], pc_list[1:], part_list[:-1], part_list[1:])):
        unique_part_ids = np.unique(part_src)
        unique_part_ids = np.sort(unique_part_ids)
        assert np.allclose(unique_part_ids, np.arange(len(unique_part_ids)))
        pc_transform = np.empty_like(pc_src)
        for part_id in unique_part_ids:
            rel_pose = np.linalg.inv(pose_list[idx, part_id]) @ pose_list[idx+1, part_id]
            pc_idx = part_src == part_id
            points = pc_src[pc_idx, :]
            points_homo = np.concatenate([points, np.ones((points.shape[0], 1), dtype=float)], axis=1)
            new_points = (points_homo @ rel_pose.T)[:, :3]
            pc_transform[pc_idx, :] = new_points
        pc_transform_list.append(pc_transform)
    pc_transform_list = np.stack(pc_transform_list, axis=0)
    return pc_transform_list