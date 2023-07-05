import os
import math
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
import networkx as nx


from utils.dataset_utils import sparse_sample_novel_state
from utils.graph_utils import compute_root_cost, compute_mean_screw_param, frobenius_cost
from utils.viz_utils import vis_pc
    
from screw_se3 import inverse_transformation
from screw_se3 import dq_to_screw, transform_to_dq
from screw_se3 import screw_param_to_exponential_coordinates, transform_from_exponential_coordinates


def extract_kinematic(seg_part, trans_list, joint_connection):
    # assign new part id from 0 to seg_part, trans_list and joint_connection
    new_seg = seg_part.clone()
    uni_label = torch.unique(seg_part, sorted=True)
    assert torch.allclose(torch.unique(joint_connection, sorted=True), uni_label)
    trans_list = trans_list[:, uni_label]  # (T, P, 4, 4)
    mapping_dict = {} # key is old part id, value is new part id
    for new_part_id, old_part_id in enumerate(uni_label):
        pc_idx = seg_part == old_part_id
        new_seg[pc_idx] = new_part_id
        mapping_dict[old_part_id.item()] = new_part_id
    for edge_idx in range(joint_connection.shape[0]):
        joint_connection[edge_idx, 0] = mapping_dict[joint_connection[edge_idx, 0].item()]
        joint_connection[edge_idx, 1] = mapping_dict[joint_connection[edge_idx, 1].item()]
    return new_seg, trans_list, joint_connection


def to_DAG(G, root_node):
    # G: undirected graph
    # Add edges from CHILDREN TO PARENTS
    paths_to_base = nx.shortest_path(G, target=root_node)
    new_edges_list = []
    for part_id in G.nodes:
        path = paths_to_base[part_id]
        for i in range(len(path) - 1):
            child_part_id, parent_part_id = path[i], path[i + 1]
            edge = (child_part_id, parent_part_id)
            if edge not in new_edges_list:
                new_edges_list.append(edge)
    assert len(new_edges_list) == G.number_of_nodes() - 1, "invalid tree structure"
    G = nx.from_edgelist(new_edges_list, create_using=nx.DiGraph())
    assert len(nx.descendants(G, root_node)) == 0  # Ensure that the root part has no parent
    return G


def build_graph(edges_list, trans_list, verbose=False, root_part=None,
                revolute_only=True, return_joint_type=False):
    # trans_list: (T, P, 4, 4), P is number of valid parts
    # edges_list: edges store pair of part id range from 0 to P
    init_G = nx.from_edgelist(edges_list.cpu().numpy(), create_using=nx.Graph())  # undirected graph
    uni_label = torch.unique(edges_list, sorted=True)
    assert torch.allclose(uni_label, torch.arange(trans_list.shape[1], dtype=uni_label.dtype, device=uni_label.device))
    if root_part is None:
        root_cost = compute_root_cost(trans_list)
        root_part = uni_label[root_cost.argmin().item()].item()
    if verbose:
        print("root part id", root_part)

    G = to_DAG(init_G, root_node=root_part)
    theta_list = []
    distance_list = []
    axis_list = []
    moment_list = []
    edge_index = {}
    joint_type_list = []
    for idx, edge in enumerate(G.edges()):
        child_part_id, parent_part_id = edge  # e.g. edge = (6, 5)
        parent_trans, child_trans = trans_list[:, parent_part_id], trans_list[:, child_part_id]
        inv_parent_trans = inverse_transformation(parent_trans)
        rel_trans = torch.bmm(inv_parent_trans, child_trans)
        dq = transform_to_dq(rel_trans)
        s_axis, moment, theta, distance = dq_to_screw(dq)
        mean_axis, mean_moment = compute_mean_screw_param(s_axis[:, None], moment[:, None], theta[:, None], distance[:, None])
        axis_list.append(mean_axis.squeeze(dim=0))
        moment_list.append(mean_moment.squeeze(dim=0))
        if revolute_only: # robot
            joint_type_list.append("revolute")
            theta_list.append(theta)
            no_rot = torch.logical_or(theta.abs() < 1e-6, (theta - math.pi).abs() < 1e-6)
            assert torch.sum(no_rot) == 0
        else: # sapien, real
            T = trans_list.shape[0]
            
            # revolute
            distance_ = distance.clone()
            distance_[...] = 1e-6
            log_transform = screw_param_to_exponential_coordinates(mean_axis.reshape(-1, 3).expand(T, 3),
                                                                   mean_moment.reshape(-1, 3).expand(T, 3),
                                                                   theta.reshape(-1), distance_.reshape(-1))
            T_recon = transform_from_exponential_coordinates(log_transform).reshape(T, 4, 4)
            geo_cost_r = frobenius_cost(T_recon.reshape(-1, 4, 4), rel_trans.reshape(-1, 4, 4))
            geo_cost_r = geo_cost_r.sum()

            # prismatic
            rel_trans_ = rel_trans.clone()
            rel_trans_[:, :3, :3] = torch.eye(3).unsqueeze(dim=0).expand(rel_trans_.shape[0], 3, 3)

            theta_ = theta.clone()
            theta_[...] = 1e-6
            log_transform = screw_param_to_exponential_coordinates(mean_axis.reshape(-1, 3).expand(T, 3),
                                                                   mean_moment.reshape(-1, 3).expand(T, 3),
                                                                   theta_.reshape(-1), distance.reshape(-1))
            T_recon = transform_from_exponential_coordinates(log_transform).reshape(-1, 4, 4)
            geo_cost_1 = frobenius_cost(T_recon.reshape(-1, 4, 4), rel_trans_.reshape(-1, 4, 4))
            geo_cost_1 = geo_cost_1.sum()
            geo_cost_2 = F.mse_loss(T_recon.reshape(-1, 4, 4)[:, :3, :3], rel_trans.reshape(-1, 4, 4)[:, :3, :3])  # mean across elements and time
            geo_cost_p = geo_cost_1 + geo_cost_2

            if geo_cost_p <= geo_cost_r:  # prismatic joint
                joint_type_list.append("prismatic")
                theta_list.append(1e-6 * torch.ones_like(theta, dtype=theta.dtype, device=theta.device))
                distance_list.append(distance)
            else:  # revolute joint
                joint_type_list.append("revolute")
                distance_list.append(1e-6 * torch.ones_like(distance, dtype=distance.dtype, device=distance.device))
                theta_list.append(theta)

        edge_name = "_".join([str(child_part_id), str(parent_part_id)])
        edge_index[edge_name] = idx
    axis_list = torch.stack(axis_list, dim=0)    # (E, 3)
    moment_list = torch.stack(moment_list, dim=0)  # (E, 3)
    theta_list = torch.stack(theta_list, dim=1)  # (T, E)
    print("joint types at each edge: {}".format(joint_type_list))
    if revolute_only:
        return G, root_part, axis_list, moment_list, theta_list, edge_index
    else:
        distance_list = torch.stack(distance_list, dim=1)  # (T, E)
        if return_joint_type:
            return G, root_part, axis_list, moment_list, theta_list, distance_list, edge_index, joint_type_list
        else:
            return G, root_part, axis_list, moment_list, theta_list, distance_list, edge_index


def edge_index2edges(edge_index):
    edges_list = []
    for edge_name in edge_index.keys():
        child_part_id, parent_part_id = edge_name.split("_")
        child_part_id, parent_part_id = int(child_part_id), int(parent_part_id)
        edges_list.append([child_part_id, parent_part_id])
    return edges_list


def fk(paths_to_base, reverse_topo, edge_index, axis_list, moment_list, theta_list,
       distance_list=None, joint_type_list=None):
    # paths_to_base: returned from nx.shortest_path(G, target=root_part), dict
    # reverse_topo: returned from list(reversed(list(nx.topological_sort(G)))), traverse node from root to leaf
    # edge_index: dict, key: edge name: "_".join([str(child_part_id), str(parent_part_id)]), value: edge index
    # axis_list, moment_list: (E, 3)
    # theta_list: (T, E)
    # distance_list: (T, E)
    # return: fk_trans_list (T, P, 4, 4), P = E+1

    T, E = theta_list.shape[:2]  # assume part from 0 to E, total (E+1) parts
    assert sorted(reverse_topo) == np.arange(E+1).tolist()

    fk_trans_list = []
    fk_valid_list = torch.zeros(E+1).bool()
    for part_id in reverse_topo:
        path = paths_to_base[part_id]  # e.g. path = [7, 6, 5, 4, 3, 2, 1, 0]
        pose = torch.eye(4, device=theta_list.device).float().unsqueeze(dim=0).expand(T, 4, 4)
        for i in range(len(path) - 1):
            child_part_id, parent_part_id = path[i], path[i + 1]
            edge_name = "_".join([str(child_part_id), str(parent_part_id)])
            s_axis = axis_list[edge_index[edge_name]].unsqueeze(dim=0).expand(T, 3)
            moment = moment_list[edge_index[edge_name]].unsqueeze(dim=0).expand(T, 3)
            if joint_type_list is None:
                theta = theta_list[:, edge_index[edge_name]]
                distance = 1e-6 * torch.ones_like(theta, dtype=theta.dtype, device=theta.device) if distance_list is None else distance_list[:, edge_index[edge_name]]
            else:
                joint_type = joint_type_list[edge_index[edge_name]]
                if joint_type == "prismatic":
                    distance = distance_list[:, edge_index[edge_name]]
                    theta = 1e-6 * torch.ones_like(distance, dtype=distance.dtype, device=distance.device) # numerical stability
                else: # revolute
                    theta = theta_list[:, edge_index[edge_name]]
                    distance = 1e-6 * torch.ones_like(theta, dtype=theta.dtype, device=theta.device)
            log_transform = screw_param_to_exponential_coordinates(s_axis, moment, theta, distance)
            T_rel = transform_from_exponential_coordinates(log_transform)
            pose = torch.bmm(T_rel, pose)
            if fk_valid_list[parent_part_id]:
                idx = reverse_topo.index(parent_part_id)
                pose = torch.bmm(fk_trans_list[idx], pose)
                break
        fk_trans_list.append(pose)
        fk_valid_list[part_id] = True
    assert torch.any(fk_valid_list).item() is True
    fk_trans_list = torch.stack(fk_trans_list, dim=1)
    order = torch.tensor(reverse_topo).argsort()
    fk_trans_list = fk_trans_list[:, order]
    return fk_trans_list


def ik(dataset, model, device, verbose=True, vis=True, save_dir=None, **ikargs):
    # robot only
    from networks.model import BaseModel
    sample = dataset[0]
    cano_pose = dataset.pose_list[dataset.cano_idx]
    cano_pc = torch.from_numpy(sample['cano_pc']).to(device)
    retarget_err = []
    for novel_state in range(len(dataset.novel_pose_list)):
        novel_pose = dataset.novel_pose_list[novel_state]
        novel_sample = sparse_sample_novel_state(sample['cano_pc'], sample['gt_cano_part'], cano_pose, novel_pose,
                                                 sparse_sample_per_part=1)
        sparse_cano_pc = torch.from_numpy(novel_sample['sparse_cano_pc']).float().to(device)
        sparse_novel_pc = torch.from_numpy(novel_sample['sparse_novel_pc']).float().to(device)

        kwargs = {}
        opt_list = []
        if isinstance(model, BaseModel):
            kwargs["tau"] = ikargs["tau"] if "tau" in ikargs else 1.0
            proposal_6d = torch.tensor([[[1, 0, 0, 0, 1, 0]]], device=device).float().repeat((1, model.num_parts, 1))
            proposal_6d.requires_grad = True
            proposal_t = torch.tensor([[[0, 0, 0]]], device=device).float().repeat((1, model.num_parts, 1))
            proposal_t.requires_grad = True

            opt_list.append(proposal_6d)
            opt_list.append(proposal_t)
            kwargs["proposal_6d"] = proposal_6d
            kwargs["proposal_t"] = proposal_t
        else:
            E = model.axis_list.shape[0]
            init_theta_list = 1e-6 * torch.ones((1, E), device=device)
            init_theta_list.requires_grad = True
            opt_list.append(init_theta_list)
            kwargs["theta_list"] = init_theta_list

        optimizer = torch.optim.Adam(opt_list, lr=1e-1, amsgrad=True)
        n_iter = 200

        for i in tqdm(range(n_iter)):
            pc_trans, seg_part, trans_list = model(sparse_cano_pc, **kwargs)
            pc_trans = pc_trans.squeeze(dim=0)
            loss = F.mse_loss(pc_trans, sparse_novel_pc, reduction='sum')
            if verbose:
                print(f"Loss: {loss:.3f}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            pc_trans, seg_part, trans_list = model(cano_pc, **kwargs)
            pc_trans = pc_trans.squeeze(dim=0)

        pc_trans = pc_trans.cpu().numpy()
        seg_part = seg_part.cpu().numpy()
        mse_dist = np.sqrt(((pc_trans - novel_sample['novel_pc']) ** 2).sum(axis=-1)).mean()
        recon_err = 100 * mse_dist
        if verbose:
            print(f"Novel retarget err: {recon_err:.3f}")
        retarget_err.append(recon_err)

        if vis and save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, "novel_{}.html".format(novel_state))
            vis_pc(pc_trans, seg_part, pc_gt=novel_sample['novel_pc'], gt_part=sample['gt_cano_part'], save_path=save_path)
            print("save retarget result {} to {}".format(novel_state, save_path))
            
    retarget_err = np.array(retarget_err).mean()
    return retarget_err