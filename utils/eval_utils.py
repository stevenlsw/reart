import numpy as np
from scipy.spatial import cKDTree as KDTree
import torch


def eval_flow(pred_flow_list, gt_flow_list, acc1_thre=0.05, acc2_thre=0.1):
    error = np.sqrt(np.sum((pred_flow_list - gt_flow_list)**2, 2) + 1e-20)
    gtflow_len = np.sqrt(np.sum(gt_flow_list * gt_flow_list, 2) + 1e-20)  # T-1,N
    acc1 = np.mean(np.logical_or(error <= acc1_thre, error / gtflow_len <= acc1_thre), axis=1)
    acc2 = np.mean(np.logical_or(error <= acc2_thre, error / gtflow_len <= acc2_thre), axis=1)
    acc1 = np.mean(acc1)  # Acc_5
    acc2 = np.mean(acc2)  # Acc_10
    epe = np.mean(error)

    unit_label = gt_flow_list / np.linalg.norm(gt_flow_list, axis=-1, keepdims=True)
    unit_pred = pred_flow_list / np.linalg.norm(pred_flow_list, axis=-1, keepdims=True)
    eps = 1e-7
    dot_product = (unit_label * unit_pred).sum(2).clip(-1+eps, 1-eps)
    dot_product[np.isnan(dot_product)] = 1.0
    angle_error = np.arccos(dot_product).mean(axis=1)
    angle_error = np.mean(angle_error)
    return epe, acc1, acc2, angle_error


def eval_seg(gt_segm, pd_segm):
    n_data = gt_segm.shape[0]
    n_gt_segm = torch.max(gt_segm) + 1
    n_pred_segm = torch.max(pd_segm) + 1
    s = max(n_gt_segm, n_pred_segm)
    pd_segm = torch.eye(s, dtype=torch.float32, device=pd_segm.device)[pd_segm]  # (N, s)
    gt_segm = torch.eye(s, dtype=pd_segm.dtype, device=pd_segm.device)[gt_segm]  # (N, s)
    ri_matrix_gt = torch.mm(gt_segm, gt_segm.transpose(-1, -2))   # (N, N)
    ri_matrix_pd = torch.mm(pd_segm, pd_segm.transpose(-1, -2))   # (N, N)
    ri = torch.sum(ri_matrix_gt == ri_matrix_pd, dim=-1).sum(dim=-1).float() / (n_data * n_data)
    ri = ri.cpu().numpy()
    return ri


def compute_chamfer(points_1, points_2, reduction="sum"):
    # one direction
    points2_kd_tree = KDTree(points_2)
    one_distances, one_vertex_ids = points2_kd_tree.query(points_1)
    # other direction
    points1_kd_tree = KDTree(points_1)
    two_distances, two_vertex_ids = points1_kd_tree.query(points_2)
    if reduction == "mean":
        p1_to_p2_chamfer = np.mean(np.square(one_distances))
        p2_to_p1_chamfer = np.mean(np.square(two_distances))
    else:
        p1_to_p2_chamfer = np.sum(np.square(one_distances))
        p2_to_p1_chamfer = np.sum(np.square(two_distances))
    return p1_to_p2_chamfer + p2_to_p1_chamfer


def compute_chamfer_list(points_set1, points_set2, reduction="sum"):
    cd_list = []
    for (points1, points2) in zip(points_set1, points_set2):
        cd = compute_chamfer(points1, points2, reduction=reduction)
        cd_list.append(cd)
    cd_list = np.stack(cd_list)
    if reduction == "mean":
        return cd_list.mean()
    elif reduction == "sum":
        return cd_list.sum()
    else:
        return cd_list







