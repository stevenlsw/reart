import os
import pickle
import math
import copy
import networkx as nx
import torch
import torch.nn.functional as F


from utils.model_utils import knn_query, compute_pc_transform
from networks.pointnet2_utils import farthest_point_sample, index_points

from screw_se3 import inverse_transformation
from screw_se3 import dq_to_screw, transform_to_dq
from screw_se3 import screw_param_to_exponential_coordinates, transform_from_exponential_coordinates


class Node():
    def __init__(self, link_names):
        self.link_names = link_names


def load_part_mapping(load_path):
    with open(load_path, 'rb') as f:
        part_dict = pickle.load(f)
    face_part_mapping = part_dict["face_part_mapping"]
    node_part_mapping = part_dict["node_part_mapping"]
    return face_part_mapping, node_part_mapping


def search_part_id(link_names, node_part_mapping):
    for part_id in node_part_mapping.keys():
        node_link_names = node_part_mapping[part_id]
        if sorted(link_names) == sorted(node_link_names):
            return part_id
    raise ValueError("{} not found it part id in {}!".format(link_names, node_part_mapping))


def fps_sample_cano(cano_pc, cano_part, uni_label, num_fps=20):
    cano_part_fps_list = []  # in cano frame, for each part, fps sampling
    cano_part_idx_list = []
    for part_id in uni_label:
        pc_idx = cano_part == part_id
        part_cano_pc = cano_pc[pc_idx]
        if pc_idx.sum() < num_fps:
            raise ValueError("part id {} too small, only {} points".format(part_id, pc_idx.sum()))
        idx = farthest_point_sample(part_cano_pc.unsqueeze(dim=0), num_fps)
        cano_part_fps = index_points(part_cano_pc.unsqueeze(dim=0), idx).squeeze(dim=0)
        cano_part_fps_list.append(cano_part_fps)
        part_idx = torch.where(pc_idx == True)[0][idx.squeeze(dim=0)]
        cano_part_idx_list.append(part_idx)
    cano_part_fps_list = torch.stack(cano_part_fps_list)  # [P, num_fps, 3]
    cano_part_idx_list = torch.stack(cano_part_idx_list)  # [P, num_fps] store idx in cano_pc
    return cano_part_fps_list, cano_part_idx_list


def fps_index_list(pc_trans_list, cano_part_idx_list):
    # pc_trans_list: (T, N, 3), cano_part_idx_list: (P, num_fps)
    T = pc_trans_list.shape[0]
    P, num_fps = cano_part_idx_list.shape
    pc_trans_list = pc_trans_list.unsqueeze(dim=1).expand(T, P, -1, 3)
    cano_part_idx_list = cano_part_idx_list.unsqueeze(dim=0).expand(T, P, num_fps)
    T_indices = torch.arange(T, dtype=torch.long, device=cano_part_idx_list.device).unsqueeze(dim=1).unsqueeze(
        dim=2).expand(T, P, num_fps)
    P_indices = torch.arange(P, dtype=torch.long, device=cano_part_idx_list.device).unsqueeze(dim=0).unsqueeze(
        dim=2).expand(T, P, num_fps)
    part_fps_list = pc_trans_list[T_indices, P_indices, cano_part_idx_list, :]
    return part_fps_list


def compute_spatial_cost(cano_part_fps_list, chamfer_dist, return_index=False):
    num_part, num_fps = cano_part_fps_list.shape[:2]  # (num_part, num_fps, 3)
    src_list = cano_part_fps_list[:, None, :, :].expand(num_part, num_part, num_fps, 3)
    tgt_list = cano_part_fps_list[None, :, :, :].expand(num_part, num_part, num_fps, 3)
    dist_src2tgt, nn_tgt_indices = chamfer_dist(src_list.reshape(-1, num_fps, 3), tgt_list.reshape(-1, num_fps, 3),
                                                return_index=True)  # (batch, num_fps), (batch, num_fps)
    dist_cost, src_indices = dist_src2tgt.reshape(num_part, num_part, num_fps).min(dim=2)
    if return_index:
        nn_tgt_indices = nn_tgt_indices.reshape(num_part, num_part, num_fps)
        tgt_indices = torch.gather(nn_tgt_indices, index=src_indices[:, :, None], dim=2).squeeze(dim=2)
        pair_indices = torch.stack([src_indices, tgt_indices], dim=2)  # (num_part, num_part, 2)
        return dist_cost, pair_indices  # pair_indices store shortest pair of sampled points between two parts
    else:
        return dist_cost


def compute_joint_cost(part_fps_list, joint_connection, edge_pair_indices):
    # part_fps_list: (T, P, num_fps, 3)
    # joint_connection: (E, 2) store part idx
    # edge_pair_indices: (E, 2) store fps idx
    # return dist: (E, ) cost of each edge
    E = joint_connection.shape[0]
    joint_0 = part_fps_list[..., joint_connection[:, 0], :, :]  # (T, E, num_fps, 3)
    joint_0 = joint_0[..., torch.arange(E, device=edge_pair_indices.device).long(), edge_pair_indices[:, 0],
              :]  # (T, E, 3)
    joint_1 = part_fps_list[..., joint_connection[:, 1], :, :]  # (T, E, num_fps, 3)
    joint_1 = joint_1[..., torch.arange(E, device=edge_pair_indices.device).long(), edge_pair_indices[:, 1],
              :]  # (T, E, 3)
    dist = ((joint_0 - joint_1) ** 2).sum(dim=-1)
    return dist  # (*, E) or (E, )


def filter_seg_label(cano_part, min_num=10):
    uni_label = torch.unique(cano_part, sorted=True)
    label = []
    for part_id in uni_label:
        pc_idx = cano_part == part_id
        if pc_idx.sum() < min_num:
            continue
        else:
            label.append(part_id)
    label = torch.tensor(label, device=cano_part.device)
    return label


def denoise_seg_label(cano_part, cano_pc, knn, min_num=10):
    mask = torch.zeros_like(cano_part, dtype=torch.bool, device=cano_part.device)
    uni_label = torch.unique(cano_part, sorted=True)
    for part_id in uni_label:
        pc_idx = cano_part == part_id
        if pc_idx.sum() < min_num:
            mask = torch.logical_or(mask, pc_idx)
    cano_part[mask] = knn_query(cano_pc[mask], cano_pc[~mask], cano_part[~mask], knn)
    return cano_part


def compute_geo_cost(rel_trans, axis, moment, theta, distance):
    T, P = axis.shape[:2]
    mean_axis, mean_moment = compute_mean_screw_param(axis.reshape(T, -1, 3), moment.reshape(T, -1, 3),
                                                      theta.reshape(T, -1), distance.reshape(T, -1))
    mean_axis, mean_moment = mean_axis.reshape(P, P, 3)[None].expand(T, P, P, 3), mean_moment.reshape(P, P, 3)[
        None].expand(T, P, P, 3)

    # revolute joint
    distance_ = distance.clone()
    distance_[...] = 1e-6
    log_transform = screw_param_to_exponential_coordinates(mean_axis.reshape(-1, 3), mean_moment.reshape(-1, 3),
                                                           theta.reshape(-1), distance_.reshape(-1))
    T_recon = transform_from_exponential_coordinates(log_transform).reshape(T, P, P, 4, 4)
    geo_cost_r = frobenius_cost(T_recon.reshape(-1, 4, 4), rel_trans.reshape(-1, 4, 4))
    geo_cost_r = geo_cost_r.reshape(T, P, P)
    geo_cost_r = geo_cost_r.sum(dim=0)

    # prismatic joint
    theta_ = theta.clone()
    rel_trans_ = rel_trans.clone()
    rel_trans_[:, :, :, :3, :3] = torch.eye(3, device=rel_trans.device, dtype=rel_trans.dtype).unsqueeze(dim=0).unsqueeze(dim=0).unsqueeze(dim=0).expand(T, P, P, 3, 3)
    theta_[...] = 1e-6
    log_transform = screw_param_to_exponential_coordinates(mean_axis.reshape(-1, 3), mean_moment.reshape(-1, 3),
                                                           theta_.reshape(-1), distance.reshape(-1))
    T_recon = transform_from_exponential_coordinates(log_transform).reshape(-1, 4, 4)
    geo_cost_1 = frobenius_cost(T_recon.reshape(-1, 4, 4), rel_trans_.reshape(-1, 4, 4))
    geo_cost_1 = geo_cost_1.reshape(T, P, P)
    geo_cost_1 = geo_cost_1.sum(dim=0)

    geo_cost_2 = F.mse_loss(T_recon.reshape(-1, 4, 4)[:, :3, :3], rel_trans.reshape(-1, 4, 4)[:, :3, :3])  # mean across elements and time
    geo_cost_p = geo_cost_1 + geo_cost_2
    geo_cost = torch.min(geo_cost_r, geo_cost_p)
    return geo_cost


def compute_relative_trans(trans_list, return_trans=False):
    # trans_list: [T, P, 4, 4]
    T, P = trans_list.shape[:2]
    inv_trans_list = inverse_transformation(trans_list.reshape(-1, 4, 4)).reshape(T, P, 4, 4)
    rel_trans = torch.matmul(inv_trans_list[:, :, None], trans_list[:, None, :]).reshape(-1, 4, 4)
    dq = transform_to_dq(rel_trans)
    s_axis, moment, theta, distance = dq_to_screw(dq)
    s_axis, moment = s_axis.reshape(T, P, P, 3), moment.reshape(T, P, P, 3)
    theta, distance = theta.reshape(T, P, P), distance.reshape(T, P, P)
    if return_trans:
        rel_trans = rel_trans.reshape(T, P, P, 4, 4)
        return s_axis, moment, theta, distance, rel_trans
    else:
        return s_axis, moment, theta, distance


def frobenius_cost(predict, gt):
    """ |predicted*igt - I| (should be 0) """
    igt = inverse_transformation(gt)
    I = torch.eye(4, dtype=predict.dtype, device=predict.device).view(1, 4, 4).expand(predict.shape[0], 4, 4)
    error = torch.bmm(predict, igt)
    distance = F.mse_loss(error, I, reduction="none").sum(dim=(-2, -1))
    return distance


def compute_root_cost(trans_list):
    # root as static part, trans close to identity
    eye = torch.eye(4, dtype=trans_list.dtype, device=trans_list.device)[None, None, :, :]
    cost = F.mse_loss(trans_list, eye, reduction='none').sum(dim=(2, 3)).mean(dim=0)
    return cost


def compute_mean_screw_param(s_axis, moment, theta, distance, eps_tol=1e-5):
    # s_axis, moment: (T, E, 3), theta, distance: (T, E, 1)
    # check identity case, where s_axis could be random
    # eps_tol: eps tolerance, should larger than eps in ../screw_se3/dq_utils.py: eps
    # return: mean_axis, mean_moment: (E, 3)
    assert s_axis.dim() == 3 and moment.dim() == 3
    T, E = s_axis.shape[:2]
    if E <= 1:
        return s_axis.mean(dim=0), moment.mean(dim=0)
    no_rot = torch.logical_or(theta.abs() <= eps_tol, (theta - math.pi).abs() <= eps_tol)
    no_trans = distance <= eps_tol
    unit_transform = torch.logical_and(no_rot, no_trans)
    mean_axis, mean_moment = [], []
    for idx in range(E):
        s, m, u = s_axis[:, idx], moment[:, idx], unit_transform[:, idx]
        if torch.all(u).item():
            mean_axis.append(s.mean(dim=0))
            mean_moment.append(m.mean(dim=0))
        else:
            mask = ~u
            mean_axis.append(s[mask].mean(dim=0))
            mean_moment.append(m[mask].mean(dim=0))
    mean_axis, mean_moment = torch.stack(mean_axis, dim=0), torch.stack(mean_moment, dim=0)
    return mean_axis, mean_moment


def compute_screw_trans(trans_list, return_cost=False):
    # trans_list: (T, *, 4, 4)
    T, E = trans_list.shape[:2]
    dq = transform_to_dq(trans_list.reshape(-1, 4, 4))
    s_axis, moment, theta, distance = dq_to_screw(dq)
    s_axis, moment, theta, distance = s_axis.reshape(T, E, 3), moment.reshape(T, E, 3), theta.reshape(T, E), distance.reshape(T, E)
    mean_axis, mean_moment = compute_mean_screw_param(s_axis, moment, theta, distance)
    mean_axis, mean_moment = mean_axis[None].expand(T, E, 3), mean_moment[None].expand(T, E, 3)

    # revolute joint
    distance_ = distance.clone()
    distance_[...] = 1e-6
    log_transform = screw_param_to_exponential_coordinates(mean_axis.reshape(-1, 3), mean_moment.reshape(-1, 3),
                                                           theta.reshape(-1), distance_.reshape(-1))
    T_recon_r = transform_from_exponential_coordinates(log_transform).reshape(T, E, 4, 4)
    geo_cost_r = frobenius_cost(T_recon_r.reshape(-1, 4, 4), trans_list.reshape(-1, 4, 4))
    geo_cost_r = geo_cost_r.reshape(T, E)
    geo_cost_r = geo_cost_r.sum(dim=0)

    #  prismatic joint
    theta_ = theta.clone()
    trans_list_ = trans_list.clone()
    trans_list_[:, :, :3, :3] = torch.eye(3, device=trans_list.device, dtype=trans_list.dtype).unsqueeze(dim=0).unsqueeze(dim=0).expand(T, E, 3, 3)
    theta_[...] = 1e-6  # not set to 0 for numerical stability
    log_transform = screw_param_to_exponential_coordinates(mean_axis.reshape(-1, 3), mean_moment.reshape(-1, 3),
                                                           theta_.reshape(-1), distance.reshape(-1))
    T_recon_p = transform_from_exponential_coordinates(log_transform).reshape(-1, 4, 4)
    geo_cost_1 = frobenius_cost(T_recon_p.reshape(-1, 4, 4), trans_list_.reshape(-1, 4, 4))
    geo_cost_1 = geo_cost_1.reshape(T, E)
    geo_cost_1 = geo_cost_1.sum(dim=0)
    geo_cost_2 = F.mse_loss(T_recon_p.reshape(-1, 4, 4)[:, :3, :3], trans_list.reshape(-1, 4, 4)[:, :3, :3])  # mean across elements and time
    geo_cost_p = geo_cost_1 + geo_cost_2

    T_recon_r = T_recon_r.reshape(T, E, 4, 4)
    T_recon_p = T_recon_p.reshape(T, E, 4, 4)
    T_recon = T_recon_r.clone()

    pris_ind = (geo_cost_p <= geo_cost_r).unsqueeze(dim=0).expand(T, E)
    T_recon[pris_ind] = T_recon_p[pris_ind]

    if return_cost:
        geo_cost = torch.min(geo_cost_r, geo_cost_p)
        return T_recon, geo_cost.mean() / T
    else:
        return T_recon


def compute_screw_cost(pred_trans_list, pred_connection):
    T, E = pred_trans_list.shape[0], pred_connection.shape[0]
    src_trans, tgt_trans = pred_trans_list[:, pred_connection[:, 0]], pred_trans_list[:, pred_connection[:, 1]]
    inv_src_trans = inverse_transformation(src_trans.reshape(-1, 4, 4)).reshape(T, E, 4, 4)
    rel_trans = torch.matmul(inv_src_trans, tgt_trans)
    T_recon, screw_cost = compute_screw_trans(rel_trans, return_cost=True)  # (T, E, 4 ,4)
    return screw_cost


def mst(cost, uni_label=None, max_cost=None, keep_index=False, verbose=False):
    # keep_index: keep indices, not use uni_label to re-index
    num_parts = cost.shape[0]
    if uni_label is not None:
        assert num_parts == len(uni_label)
    connectivity = torch.eye(num_parts, device=cost.device, dtype=torch.long)
    joint_connection = torch.zeros(num_parts - 1, 2, device=cost.device, dtype=torch.long)
    for j in range(num_parts - 1):
        invalid_connection = connectivity * 1e10
        cur_cost = cost + invalid_connection
        connected = torch.argmin(cur_cost)
        connected_idx_0 = torch.div(connected, num_parts, rounding_mode='trunc')
        connected_idx_1 = connected % num_parts
        if max_cost is not None and cur_cost[connected_idx_0, connected_idx_1] > max_cost:  # stop criteria
            return joint_connection[:j]
        if verbose:
            if uni_label is not None:
                print(uni_label[connected_idx_0].item(), uni_label[connected_idx_1].item(),
                      cur_cost[connected_idx_0, connected_idx_1].item())
            else:
                print(connected_idx_0.item(), connected_idx_1.item(), cur_cost[connected_idx_0, connected_idx_1].item())
        connectivity[connected_idx_0] = torch.maximum(connectivity[connected_idx_0].clone(),
                                                      connectivity[connected_idx_1].clone())
        connectivity[torch.where(connectivity[connected_idx_0] == 1)] = connectivity[connected_idx_0].clone()

        joint_connection[j, 0] = connected_idx_0 if (uni_label is None or keep_index) else uni_label[connected_idx_0]
        joint_connection[j, 1] = connected_idx_1 if (uni_label is None or keep_index) else uni_label[connected_idx_1]
    return joint_connection


def merge_graph(seg_part, joint_connection, trans_list, merge_thr, verbose=True):
    # seg_part (N, ) , joint_connection (E, 2), trans_list (T-1, P, 4, 4): torch tensor
    G = nx.DiGraph()
    T, E = trans_list.shape[0], joint_connection.shape[0]
    part_ids = torch.unique(joint_connection)
    for part_id in part_ids:
        G.add_node(part_id.item())

    src_trans, tgt_trans = trans_list[:, joint_connection[:, 0]], trans_list[:, joint_connection[:, 1]]
    inv_src_trans = inverse_transformation(src_trans.reshape(-1, 4, 4)).reshape(T, E, 4, 4)
    rel_trans = torch.matmul(inv_src_trans, tgt_trans).reshape(-1, 4, 4)
    eye = torch.eye(4, device=rel_trans.device, dtype=rel_trans.dtype)[None].expand(rel_trans.shape[0], 4, 4)
    vanilla_cost = frobenius_cost(rel_trans, eye).reshape(-1, E).mean(dim=0)  # (E, )

    for idx, edge in enumerate(joint_connection):
        cost = vanilla_cost[idx].item()
        G.add_edge(edge[0].item(), edge[1].item(), cost=cost)
        if verbose:
            print("add edge {}-{}: cost {}".format(edge[0], edge[1], cost))

    M = copy.deepcopy(G)
    merge_cano_part = seg_part.clone()
    topo = list(nx.topological_sort(G))  # from leaf to root
    for node in topo:
        if M.has_node(node):
            edges = list(nx.edges(M, node))
        else:
            continue
        for edge in edges:
            if M.has_node(edge[1]):
                edge_cost = M.get_edge_data(edge[0], edge[1])['cost']
                if edge_cost < merge_thr:
                    M = nx.contracted_edge(M, edge, self_loops=False)
                    merge_cano_part[merge_cano_part == edge[1]] = edge[0]
                    if verbose:
                        print("merge edge {}-{}: cost {}".format(edge[1], edge[0], edge_cost))

    # Check that the link graph is weakly connected
    if not nx.is_weakly_connected(M):
        message = 'New graph are not all connected.'
        raise ValueError(message)

        # Check that link graph is acyclic
    if not nx.is_directed_acyclic_graph(M):
        raise ValueError('There are cycles in the link graph')

    joint_connection_list = []
    for edge in M.edges:
        joint_connection_list.append([edge[0], edge[1]])
        edge_cost = M.get_edge_data(edge[0], edge[1])['cost']
        if verbose:
            print("remain edge {}-{}: cost {}".format(edge[0], edge[1], edge_cost))
    new_connection = torch.tensor(joint_connection_list, device=joint_connection.device, dtype=joint_connection.dtype)
    return merge_cano_part, new_connection


def merging_wrapper(seg_part, trans_list, cano_pc, chamfer_dist, merge_thr, n_it=2):
    pred_pc_list = compute_pc_transform(cano_pc, trans_list, seg_part)
    for it in range(n_it):
        uni_label = torch.unique(seg_part, sorted=True)
        cano_part_fps_list, cano_part_idx_list = fps_sample_cano(cano_pc, seg_part, uni_label, num_fps=20)
        part_fps_list = fps_index_list(pred_pc_list, cano_part_idx_list)
        # pair_indices: (num_part, num_part, 2), record fps indices sampled from cano frame
        cano_dist, pair_indices = compute_spatial_cost(cano_part_fps_list, chamfer_dist, return_index=True)
        # dist_cost = 0 * (cano_dist < 5e-3) + 1e4 * (cano_dist >= 5e-3)

        edge_pair_indices = pair_indices.reshape(-1, 2)
        part_x, part_y = torch.meshgrid(torch.arange(len(uni_label), device=edge_pair_indices.device),
                                        torch.arange(len(uni_label), device=edge_pair_indices.device))
        joint_connection = torch.stack([part_x, part_y], dim=2).reshape(-1, 2)

        dist = compute_joint_cost(part_fps_list, joint_connection, edge_pair_indices)
        dist = dist.reshape(-1, len(uni_label), len(uni_label))
        joint_cost = dist.sum(dim=0)

        merge_cost = cano_dist + joint_cost
        merge_cost = merge_cost + 1e4 * torch.eye(merge_cost.shape[0], device=merge_cost.device, dtype=merge_cost.dtype)
        candidates_connection = mst(merge_cost, uni_label=uni_label, verbose=False)
        seg_part, new_connection = merge_graph(seg_part, candidates_connection, trans_list, merge_thr, verbose=False)
        if not len(torch.unique(seg_part)) > 1:
            break

    return seg_part


def mst_wrapper(seg_part, trans, cano_pc, chamfer_dist, verbose=False, num_fps=20,
                cano_dist_thr=1e-2, joint_cost_weight=100):
    pred_pc_list = compute_pc_transform(cano_pc, trans, seg_part)
    uni_label = torch.unique(seg_part, sorted=True)
    axis, moment, theta, distance, rel_trans = compute_relative_trans(trans, return_trans=True)
    sel_axis = axis[:, uni_label, :][:, :, uni_label]
    sel_moment = moment[:, uni_label, :][:, :, uni_label]
    sel_theta = theta[:, uni_label, :][:, :, uni_label]
    sel_distance = distance[:, uni_label, :][:, :, uni_label]
    sel_rel_trans = rel_trans[:, uni_label, :][:, :, uni_label]
    geo_cost = compute_geo_cost(sel_rel_trans, sel_axis, sel_moment, sel_theta, sel_distance)

    cano_part_fps_list, cano_part_idx_list = fps_sample_cano(cano_pc, seg_part, uni_label, num_fps=num_fps)
    part_fps_list = fps_index_list(pred_pc_list, cano_part_idx_list)
    # pair_indices: (num_part, num_part, 2), record fps indices sampled from cano frame
    cano_dist, pair_indices = compute_spatial_cost(cano_part_fps_list, chamfer_dist, return_index=True)
    dist_cost = 0 * (cano_dist < cano_dist_thr) + 1e4 * (cano_dist >= cano_dist_thr)

    edge_pair_indices = pair_indices.reshape(-1, 2)
    part_x, part_y = torch.meshgrid(torch.arange(len(uni_label), device=edge_pair_indices.device),
                                    torch.arange(len(uni_label), device=edge_pair_indices.device))
    joint_connection = torch.stack([part_x, part_y], dim=2).reshape(-1, 2)

    dist = compute_joint_cost(part_fps_list, joint_connection, edge_pair_indices)
    dist = dist.reshape(-1, len(uni_label), len(uni_label))
    joint_cost = dist.sum(dim=0)

    cost = dist_cost + geo_cost + joint_cost_weight * joint_cost  # could not use dist cost or joint cost alone
    cost = cost + 1e4 * torch.eye(cost.shape[0], device=cost.device, dtype=cost.dtype)

    joint_connection = mst(cost, uni_label=uni_label, verbose=verbose)
    return joint_connection