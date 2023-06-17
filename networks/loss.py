
import torch
import torch.nn.functional as F

from utils.graph_utils import frobenius_cost, compute_mean_screw_param
from screw_se3 import screw_param_to_exponential_coordinates, transform_from_exponential_coordinates



def flow_loss(gt_flow_list, pred_flow_list, flow_mask_list=None, robust=False, smooth_weight=1e-2):
    if flow_mask_list is None:
        flow_mask_list = torch.ones(pred_flow_list.shape[:2], device=pred_flow_list.device)
    if not robust:
        f_loss = F.mse_loss(pred_flow_list, gt_flow_list, reduction="none").sum(dim=2)
    else:
        f_loss = F.huber_loss(pred_flow_list, gt_flow_list, reduction="none").sum(dim=2)
    mask_flow_loss = flow_mask_list * f_loss
    smooth_loss = (pred_flow_list ** 2).sum(dim=2)
    unmask_smooth_loss = torch.logical_not(flow_mask_list) * smooth_loss
    dist_loss = (mask_flow_loss + smooth_weight * unmask_smooth_loss).sum()
    return dist_loss


def recon_loss(pc_trans_list, pc_list, chamfer_dist):
    # pc_trans_list, pc_list: [T-1, N, 3]
    # trans_list: [T-1, P, 4, 4]
    cd = chamfer_dist(pc_trans_list, pc_list, bidirectional=True)  # [T-1, N]
    dist_loss = torch.sum(cd)
    return dist_loss


def structure_loss(rel_trans_list, axis, moment, theta, distance, edge_list):
    # rel_trans_list, s_axis, moment, theta, distance: (T, P, P, *)
    # edge_list: (*, 2)

    T, E = axis.shape[0], edge_list.shape[0]
    sel_rel_trans = rel_trans_list[:, edge_list[:, 0], edge_list[:, 1]]
    sel_s_axis = axis[:, edge_list[:, 0], edge_list[:, 1]]
    sel_moment = moment[:, edge_list[:, 0], edge_list[:, 1]]
    sel_theta = theta[:, edge_list[:, 0], edge_list[:, 1]]
    sel_distance = distance[:, edge_list[:, 0], edge_list[:, 1]]

    with torch.no_grad():
        mean_axis, mean_moment = compute_mean_screw_param(sel_s_axis, sel_moment, sel_theta, sel_distance)
        mean_axis, mean_moment = mean_axis[None].expand(T, E, 3), mean_moment[None].expand(T, E, 3)
        sel_theta_, sel_distance_ = sel_theta.clone(), sel_distance.clone()
        mean_theta, mean_dist = sel_theta_.abs().mean(dim=0), sel_distance_.abs().mean(dim=0)
        pris_ind = (mean_dist > mean_theta).unsqueeze(dim=0).expand(T, E)
        rev_ind = (mean_theta >= mean_dist).unsqueeze(dim=0).expand(T, E)
        sel_theta_[pris_ind], sel_distance_[rev_ind] = 1e-6, 1e-6
        log_transform = screw_param_to_exponential_coordinates(mean_axis.reshape(-1, 3), mean_moment.reshape(-1, 3),
                                                               sel_theta_.reshape(-1), sel_distance_.reshape(-1))
        target_trans = transform_from_exponential_coordinates(log_transform).reshape(T, E, 4, 4)

    geo_cost = frobenius_cost(sel_rel_trans.reshape(-1, 4, 4), target_trans.reshape(-1, 4, 4))
    loss = geo_cost.reshape(T, E).sum()
    return loss


def compute_connection_loss(cano_pc, seg_part, joint_connection, pc_trans_list, chamfer_dist, k=10):
    loss = 0
    for edge in joint_connection:
        src_cano_mask, tgt_cano_mask = seg_part == edge[0], seg_part == edge[1]
        src_cano_pc, tgt_cano_pc = cano_pc[src_cano_mask], cano_pc[tgt_cano_mask]
        src_cano_idx, tgt_cano_idx = torch.where(src_cano_mask)[0], torch.where(tgt_cano_mask)[0]
        dist_src2tgt, nn_tgt_indices = chamfer_dist(src_cano_pc[None, :, :], tgt_cano_pc[None, :, :], return_index=True)
        dist_src2tgt, nn_tgt_indices = dist_src2tgt.squeeze(dim=0), nn_tgt_indices.squeeze(dim=0)
        min_dist_cano, src_idx = torch.topk(dist_src2tgt, k=k, dim=0, largest=False)  # indices in src part pc

        tgt_idx = nn_tgt_indices[src_idx]  # indices in tgt part pc
        # assert torch.allclose(min_dist_cano, ((src_cano_pc[src_idx] - tgt_cano_pc[tgt_idx]) ** 2).sum(dim=1))
        raw_src_idx = src_cano_idx[src_idx]  # indices in cano_pc
        raw_tgt_idx = tgt_cano_idx[tgt_idx]  # indices in cano_pc

        raw_src_pc_list = pc_trans_list[:, raw_src_idx, :]  # (T-1, k, 3)
        raw_tgt_pc_list = pc_trans_list[:, raw_tgt_idx, :]  # (T-1, k, 3)
        dist = ((raw_src_pc_list - raw_tgt_pc_list) ** 2).sum(dim=2).mean(dim=1)
        loss = loss + dist
    return loss.sum(dim=0)  # sum over time