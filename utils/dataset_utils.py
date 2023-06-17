import os
import numpy as np
import networkx as nx
import pickle


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


def load_state(load_path):
    with open(load_path, 'rb') as f:
        state_info = pickle.load(f)
    pc = state_info['pc']
    part_id = state_info['part_id']
    return pc, part_id


def load_pose(load_path):
    with open(load_path, 'rb') as f:
        rel_pose = pickle.load(f)
    return rel_pose


def get_T_from_Rt(R, t):
    T = np.concatenate((R, t[:, None]), axis=1)
    T = np.concatenate((T, np.array([[0, 0, 0, 1]], dtype=T.dtype)), axis=0)
    return T


def get_rel_pose(pose_cano2src, pose_cano2tgt):
    pose_src2tgt = {}
    for part_id in pose_cano2src.keys():
        pose_src2tgt[part_id] = pose_cano2tgt[part_id] @ np.linalg.inv(pose_cano2src[part_id])
    return pose_src2tgt


def load_normalize_dict(normalize_filename):
    with open(normalize_filename, 'rb') as f:
        normalize_dict = pickle.load(f)
    return normalize_dict


def pose_identity_like(pose_dict):
    identity_dict = {}
    for part_id in pose_dict.keys():
        identity_dict[part_id] = np.eye(4)
    return identity_dict


def sparse_sample_novel_state(cano_pc, gt_cano_part, cano_pose, novel_pose, sparse_sample_per_part=1):
    unique_part_ids = sorted(list(set(gt_cano_part.tolist())))
    pc_transform = np.empty_like(cano_pc)
    pose_cano2novel = get_rel_pose(cano_pose, novel_pose)
    pose_list = []
    num_sparse_points = sparse_sample_per_part * len(unique_part_ids)
    sparse_pc_0 = np.empty((num_sparse_points, 3))
    sparse_pc_1 = np.empty_like(sparse_pc_0)
    sparse_part_id = np.empty(num_sparse_points)
    start_idx = 0
    for part_id in unique_part_ids:
        part_gt_pose = pose_cano2novel[part_id]
        pose_list.append(part_gt_pose)
        pc_idx = gt_cano_part == part_id
        points = cano_pc[pc_idx, :]
        points_homo = np.concatenate([points, np.ones((points.shape[0], 1), dtype=float)], axis=1)
        new_points = (points_homo @ part_gt_pose.T)[:, :3]
        pc_transform[pc_idx, :] = new_points

        assert len(points) > 10 + sparse_sample_per_part
        choose_idx = 10 + np.arange(sparse_sample_per_part) # fix retarget point index
        points = points[choose_idx, :]
        
        sparse_pc_0[start_idx:start_idx + sparse_sample_per_part, :] = points
        sparse_part_id[start_idx:start_idx + sparse_sample_per_part] = part_id
        points_homo = np.concatenate([points, np.ones((points.shape[0], 1), dtype=float)], axis=1)
        new_points = (points_homo @ part_gt_pose.T)[:, :3]
        sparse_pc_1[start_idx:start_idx + sparse_sample_per_part, :] = new_points
        start_idx = start_idx + sparse_sample_per_part

    pose_list = np.stack(pose_list).astype('float32')
    sample = {'gt_novel_pose': pose_list, 'gt_sparse_part': sparse_part_id, 'novel_pc': pc_transform,
              'sparse_cano_pc': sparse_pc_0, 'sparse_novel_pc': sparse_pc_1}
    return sample


def load_gt_graph(graph_root_path):
    import sys
    from utils import graph_utils
    sys.modules["dataset.merge"] = graph_utils # import graph definition and load graph data
    graph_path = os.path.join(graph_root_path, "graph.gpickle")
    graph_mapping_path = os.path.join(graph_root_path, "part_mapping.pkl")
    assert os.path.exists(graph_path)
    assert os.path.exists(graph_mapping_path)
    graph = nx.read_gpickle(graph_path)
    face_part_mapping, node_part_mapping = graph_utils.load_part_mapping(graph_mapping_path)
    for idx, node in enumerate(graph.nodes):
        node.part_id = graph_utils.search_part_id(node.link_names, node_part_mapping)
    gt_edges_list = []
    for edge in graph.edges:
        child, parent = edge
        child_part_id, parent_part_id = child.part_id, parent.part_id
        gt_edges_list.append((child_part_id, parent_part_id)) # edge from child to parent (here parent is shallow depth nodes)
    gt_graph = nx.from_edgelist(gt_edges_list, create_using=nx.DiGraph())
    return gt_graph, gt_edges_list
