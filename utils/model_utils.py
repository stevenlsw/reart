import math
import torch
import torch.nn.functional as F
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from multiprocessing import Pool


from screw_se3 import inverse_transformation


def th_with_zeros(tensor):
    batch_size = tensor.shape[0]
    padding = tensor.new([0.0, 0.0, 0.0, 1.0])
    padding.requires_grad = False

    concat_list = [tensor, padding.view(1, 1, 4).repeat(batch_size, 1, 1)]
    cat_res = torch.cat(concat_list, 1)
    return cat_res


def create_transformation(rotation, translation):
    # rotation: 	[Bx3x3]
    # translation:  [Bx3x1]
    B = rotation.shape[0]
    last_row = torch.zeros(B, 1, 4, device=rotation.device)
    last_row[:, :, 3] = 1
    T = torch.cat([rotation, translation], axis=2)
    T = torch.cat([T, last_row], axis=1)
    return T


def tau_cosine(cur_iter, max_iter, end_temp, start_temp):
    """
    """
    assert end_temp <= start_temp
    return end_temp + (start_temp - end_temp) * (math.cos(math.pi * cur_iter / max_iter) + 1.0) * 0.5



def knn_query(query_pc, src_pc, src_input, knn):
    _, idx = knn(ref=src_pc.unsqueeze(dim=0), query=query_pc.unsqueeze(dim=0))  # [bs x nq x k]
    idx = idx.squeeze(dim=0).reshape(-1)
    if len(src_input.shape) == 2:
        target_seg = src_input[idx].reshape(src_input.shape[0], knn.k, src_input.shape[1])
        target_seg = target_seg.mean(dim=1)
        return target_seg
    else:
        part_ids = src_input[idx].reshape(-1, knn.k)
        target_group_id = torch.mode(part_ids, dim=1)[0]
        return target_group_id


def compute_pc_transform(cano_pc, pose_list, cano_part):
    # cano_pc: (N, 3)
    # pred_pose_list: (T-1, P, 4, 4)
    # pred_cano_part: (N, )
    pose_len, num_parts = pose_list.shape[:2]
    N = cano_pc.shape[0]
    rotation, translation = pose_list[:, :, :3, :3].reshape(-1, 3, 3), pose_list[:, :, :3, 3].reshape(-1, 3)
    cano_pc = cano_pc.unsqueeze(dim=0).unsqueeze(dim=1).expand(pose_len, num_parts, N, 3).reshape(-1, N,
                                                                                                  3)  # [T-1 * P, N, 3]
    pc_trans_list = torch.bmm(cano_pc, rotation.transpose(1, 2)) + translation[:, None, :]  # [T-1 * P, N, 3]
    pc_trans_list = pc_trans_list.reshape(pose_len, num_parts, N, 3)  # [T-1, P, N, 3]
    weight = F.one_hot(cano_part, num_classes=num_parts).permute(1, 0)[None, :, :, None]  # [1, P, N, 1]
    pc_trans_list = (weight * pc_trans_list).sum(dim=1)  # [T-1, N, 3]
    return pc_trans_list



def get_src_permutation_idx(indices):
    # permute predictions following indices
    batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
    src_idx = torch.cat([src for (src, _) in indices])
    return batch_idx, src_idx


def get_tgt_permutation_idx(indices):
    # permute targets following indices
    batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
    tgt_idx = torch.cat([tgt for (_, tgt) in indices])
    return batch_idx, tgt_idx


def parallel_lap(cost, nproc):
    # cost_matrx: (T, N, N)
    with Pool(processes=nproc) as pool:
        mapresult = pool.starmap_async(linear_sum_assignment, zip(cost))
        return mapresult.get()


def compute_ass_err(pc_trans_list, pc_list, use_nproc=True): # used in model selection
    cost = torch.cdist(pc_trans_list, pc_list).cpu().numpy()
    if not use_nproc:
        indices = [linear_sum_assignment(c) for c in cost]
    else:
        indices = parallel_lap(cost, nproc=len(cost))
    assign_indices = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
                      for i, j in indices]
    ass_src_idx = get_src_permutation_idx(assign_indices)
    ass_tgt_idx = get_tgt_permutation_idx(assign_indices)
    ass_err = ((pc_trans_list[ass_src_idx] - pc_list[ass_tgt_idx]) ** 2).sum(dim=-1).mean()
    return ass_err


def compute_group_temporal_err(pc_list, seg_part):
    # seg_part: [N, ]
    # pc_list: [T, N, 3]
    uni_label = torch.unique(seg_part, sorted=True)
    cost = []
    for part_id in uni_label:
        pc_idx = seg_part == part_id
        part_pc = pc_list[:, pc_idx, :]
        part_centroid = part_pc.mean(dim=1, keepdim=True)
        dist = ((part_pc - part_centroid)**2).sum(dim=2)
        cost.append(dist.mean().item())
    cost = torch.tensor(cost).float().max()
    return cost


def compute_align_trans(trans_list, root_trans):
    # trans_list: (T, P, 4, 4)
    # root_trans: (T, 4, 4)
    inv_root_trans_list = inverse_transformation(root_trans)[:, None, :, :].expand(trans_list.shape)
    align_trans_list = torch.matmul(inv_root_trans_list, trans_list)
    return align_trans_list
