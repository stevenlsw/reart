import torch
from typing import Optional, Tuple




def match_snn(
    desc1: torch.Tensor, desc2: torch.Tensor, th: float = 0.9, dm: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Function, which finds nearest neighbors in desc2 for each vector in desc1.

    The method satisfies first to second nearest neighbor distance <= th.

    If the distance matrix dm is not provided, :py:func:`torch.cdist` is used.

    Args:
        desc1: Batch of descriptors of a shape :math:`(B1, D)`.
        desc2: Batch of descriptors of a shape :math:`(B2, D)`.
        th: distance ratio threshold.
        dm: Tensor containing the distances from each descriptor in desc1
          to each descriptor in desc2, shape of :math:`(B1, B2)`.

    Return:
        - Descriptor distance of matching descriptors, shape of :math:`(B3, 1)`.
        - Long tensor indexes of matching descriptors in desc1 and desc2. Shape: :math:`(B3, 2)`,
          where 0 <= B3 <= B1.
    """

    if desc2.shape[0] < 2:
        raise AssertionError

    if dm is None:
        dm = torch.cdist(desc1, desc2)
    else:
        if not ((dm.size(0) == desc1.size(0)) and (dm.size(1) == desc2.size(0))):
            raise AssertionError

    vals, idxs_in_2 = torch.topk(dm, 2, dim=1, largest=False)
    ratio = vals[:, 0] / vals[:, 1]
    mask = ratio <= th
    match_dists = ratio[mask]
    idxs_in1 = torch.arange(0, idxs_in_2.size(0), device=dm.device)[mask]
    idxs_in_2 = idxs_in_2[:, 0][mask]
    matches_idxs = torch.cat([idxs_in1.view(-1, 1), idxs_in_2.view(-1, 1)], dim=1)
    return match_dists.view(-1, 1), matches_idxs.view(-1, 2)


def match_smnn(
    desc1: torch.Tensor, desc2: torch.Tensor, th: float = 0.9, dm: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Function, which finds mutual nearest neighbors in desc2 for each vector in desc1.

    the method satisfies first to second nearest neighbor distance <= th.

    If the distance matrix dm is not provided, :py:func:`torch.cdist` is used.

    Args:
        desc1: Batch of descriptors of a shape :math:`(B1, D)`.
        desc2: Batch of descriptors of a shape :math:`(B2, D)`.
        th: distance ratio threshold.
        dm: Tensor containing the distances from each descriptor in desc1
          to each descriptor in desc2, shape of :math:`(B1, B2)`.

    Return:
        - Descriptor distance of matching descriptors, shape of. :math:`(B3, 1)`.
        - Long tensor indexes of matching descriptors in desc1 and desc2,
          shape of :math:`(B3, 2)` where 0 <= B3 <= B1.
    """

    if desc1.shape[0] < 2:
        raise AssertionError
    if desc2.shape[0] < 2:
        raise AssertionError

    if dm is None:
        dm = torch.cdist(desc1, desc2)
    else:
        if not ((dm.size(0) == desc1.size(0)) and (dm.size(1) == desc2.size(0))):
            raise AssertionError

    dists1, idx1 = match_snn(desc1, desc2, th, dm)
    dists2, idx2 = match_snn(desc2, desc1, th, dm.t())

    if len(dists2) > 0 and len(dists1) > 0:
        idx2 = idx2.flip(1)
        idxs_dm = torch.cdist(idx1.float(), idx2.float(), p=1.0)
        mutual_idxs1 = idxs_dm.min(dim=1)[0] < 1e-8
        mutual_idxs2 = idxs_dm.min(dim=0)[0] < 1e-8
        good_idxs1 = idx1[mutual_idxs1.view(-1)]
        good_idxs2 = idx2[mutual_idxs2.view(-1)]
        dists1_good = dists1[mutual_idxs1.view(-1)]
        dists2_good = dists2[mutual_idxs2.view(-1)]
        _, idx_upl1 = torch.sort(good_idxs1[:, 0])
        _, idx_upl2 = torch.sort(good_idxs2[:, 0])
        good_idxs1 = good_idxs1[idx_upl1]
        match_dists = torch.max(dists1_good[idx_upl1], dists2_good[idx_upl2])
        matches_idxs = good_idxs1
    else:
        matches_idxs, match_dists = torch.empty(0, 2, device=dm.device), torch.empty(0, 1, device=dm.device)
    return match_dists.view(-1, 1), matches_idxs.view(-1, 2)


def find_mutual_correspondences(nns01, nns10):
    corres01_idx0 = torch.arange(len(nns01)).to(nns10.device)
    corres01_idx1 = nns01

    corres10_idx0 = nns10
    mutual_filter = (corres10_idx0[corres01_idx1] == corres01_idx0)

    corres_idx0 = corres01_idx0[mutual_filter]
    corres_idx1 = corres01_idx1[mutual_filter]

    return corres_idx0, corres_idx1


def compute_corr_list_filter(norm_pc_list, feature_extractor, knn, matching="smnn"):
    # norm_pc_list: [T, N, 3]
    # return corr_list: [T-1, N]
    # conduct mutual filtering
    norm_pc_list = norm_pc_list.transpose(1, 2)
    corrs_src_list, corrs_tgt_list = [], []
    with torch.no_grad():
        src_feat = feature_extractor(norm_pc_list[:-1])
        tgt_feat = feature_extractor(norm_pc_list[1:])

        if matching == "mnn":
            _, nns_src2tgt = knn(ref=tgt_feat, query=src_feat)  # [bs x k x nq]
            nns_src2tgt = nns_src2tgt.squeeze(dim=1)

            _, nns_tgt2src = knn(ref=src_feat, query=tgt_feat)  # [bs x k x nq]
            nns_tgt2src = nns_tgt2src.squeeze(dim=1)

            batch_size = src_feat.shape[0]
            for batch_idx in range(batch_size):
                nns01 = nns_src2tgt[batch_idx]
                nns10 = nns_tgt2src[batch_idx]
                pred_corrs_src, pred_corrs_tgt = find_mutual_correspondences(nns01, nns10)
                corrs_src_list.append(pred_corrs_src), corrs_tgt_list.append(pred_corrs_tgt)
        else:
            for (src, tgt) in zip(src_feat, tgt_feat):
                _, matches_idxs = match_smnn(src.T, tgt.T)
                corrs_src_list.append(matches_idxs[:, 0]), corrs_tgt_list.append(matches_idxs[:, 1])
    return corrs_src_list, corrs_tgt_list



def blend_anchor_motion(query_loc, reference_loc, reference_flow, knn, return_mask=False):
    '''approximate flow on query points
    this function assume query points are sub- or un-sampled from reference locations
    @param query_loc:[m,3]
    @param reference_loc:[n,3]
    @param reference_flow:[n,3]
    @param knn: KNN CUDA instance
    @return:
        blended_flow:[m,3]
        return_mask: [m], mask of valid blended flow
    '''
    dists, idx = knn(reference_loc.unsqueeze(dim=0), query_loc.unsqueeze(dim=0))
    dists, idx = dists.squeeze(dim=0), idx.squeeze(dim=0)
    dists[dists < 1e-10] = 1e-10
    weight = 1.0 / dists
    weight = weight / weight.sum(dim=-1, keepdim=True)  # [m, 3]
    blended_flow = (reference_flow[idx] * weight.reshape([-1, knn.k, 1])).sum(dim=1, keepdims=False)
    if return_mask:
        min_dists, _ = dists.min(dim=-1)  # [m, ]
        flow_dists, _ = (reference_flow[idx]**2).sum(dim=-1).max(dim=1)  # [m, ]
        flow_mask = torch.logical_or(min_dists <= flow_dists, min_dists <= 0.05)
        return blended_flow, flow_mask
    else:
        return blended_flow


def normalize_pc_list(pc_list, centroid, scale):
    norm_pc_list = (pc_list - centroid) * scale
    return norm_pc_list