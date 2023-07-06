import argparse
import os
import sys
import functools
import random
import networkx as nx
import numpy as np
from scipy.optimize import linear_sum_assignment
import pickle
import torch
from tqdm import tqdm

from utils.chamfer import ChamferDistance # https://github.com/krrish94/chamferdist
from knn_cuda import KNN  # https://github.com/unlimblue/KNN_CUDA

from utils.viz_utils import vis_pc, vis_structure, vis_pc_seq
from utils.model_utils import compute_pc_transform, tau_cosine, compute_ass_err, get_src_permutation_idx, get_tgt_permutation_idx, parallel_lap, compute_align_trans
from utils.kinematic_utils import extract_kinematic, build_graph, edge_index2edges, compute_root_cost
from utils.graph_utils import denoise_seg_label, merging_wrapper, mst_wrapper, compute_screw_cost
from utils.eval_utils import eval_seg, compute_chamfer_list


from dataset.dataset_sapien import Sapien
from utils.sapien_utils import compute_full_flow, eval_flow, load_model, compute_flow_list, seg_propagation_list
from msync.models.full_net import feature_propagation

from networks.model import BaseModel, KinematicModel
from networks.loss import recon_loss, flow_loss
from networks.pointnet2_utils import farthest_point_sample, index_points
from networks.feature_extractor import get_extractor


def main(args):
    # Initialize randoms seeds
    torch.cuda.manual_seed_all(args.manual_seed)
    torch.manual_seed(args.manual_seed)
    np.random.seed(args.manual_seed)
    random.seed(args.manual_seed)

    dataset = Sapien(args.sapien_base_folder, cano_idx=args.cano_idx)
    exp_name = "sapien_{}".format(args.sapien_idx)
    save_dir = os.path.join(args.save_root, exp_name)
    os.makedirs(save_dir, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.backends.cudnn.deterministic = True
    else:
        device = torch.device("cpu")

    chamfer_dist = ChamferDistance()
    sample = dataset[args.sapien_idx]
    cano_pc = torch.from_numpy(sample['cano_pc']).float().to(device)

    complete_pc_list = torch.from_numpy(sample['complete_pc_list']).float().to(device)
    pc_list = torch.from_numpy(sample['pc_list']).float().to(device)  # exclude cano frame
    cano_idx = dataset.cano_idx
    
    # visualize input point cloud sequence
    save_path = os.path.join(save_dir, f"input.gif")
    vis_pc_seq(sample['complete_pc_list'], name="input", save_path=save_path)
    print("save input pc vis to {}".format(save_path))

    complete_gt_part_list = torch.from_numpy(sample['complete_gt_part_list']).long().to(device)
    gt_full_flow = torch.from_numpy(sample['gt_full_flow']).to(device)

    if args.use_flow_loss:
        # use multibody-sync predicted flow
        flow_model = load_model(config_path=args.flow_model_config_path, model_path=args.flow_model_path)
        flow_model.to(device)
        flow_model.eval()
        knn_flow = KNN(k=3, transpose_mode=True)
        complete_pc_list = torch.from_numpy(sample['complete_pc_list']).float().to(device)
        flow_ref_list, _ = compute_flow_list(flow_model, complete_pc_list)
        pc_ref_list = complete_pc_list[:-1]

    if args.evaluate and args.resume is None:
        raise ValueError("need model path to evaluate!")

    tau_func = functools.partial(tau_cosine, max_iter=args.n_iter, end_temp=args.end_tau, start_temp=args.start_tau)
    if args.model == "base":
        model = BaseModel(num_parts=args.num_parts, pose_len=pc_list.shape[0])
        if args.resume is not None:
            checkpoint = torch.load(args.resume[0], map_location=device)
            model.load_state_dict(checkpoint["state_dict"], strict=False)
            print("=> loaded model checkpoint {}".format(args.resume[0]))
            tau = checkpoint["tau"]
            tau_func = lambda x: tau
            if "cano_idx" in checkpoint:
                assert args.cano_idx == checkpoint["cano_idx"]

    elif args.model == "kinematic":
        if args.resume is None:
            assert args.base_result_path is not None
            with open(os.path.join(args.base_result_path), 'rb') as f:
                result = pickle.load(f)
            print(f"load base result from {args.base_result_path}")
            assert args.cano_idx == result['cano_idx']
            seg_part = torch.from_numpy(result['pred_cano_part']).long().to(device)
            trans_list = torch.from_numpy(result['pred_pose_list']).float().to(device)

            if "joint_connection" in result:
                joint_connection = torch.from_numpy(np.array(result['joint_connection'])).long().to(device)
            else:
                root_part = torch.mode(seg_part).values.item()
                root_trans = trans_list[:, root_part]
                align_trans_list = compute_align_trans(trans_list, root_trans)
                seg_part = merging_wrapper(seg_part, align_trans_list, cano_pc, chamfer_dist, args.merge_thr)
                joint_connection = mst_wrapper(seg_part, align_trans_list, cano_pc, chamfer_dist, verbose=False,
                                               num_fps=20, cano_dist_thr=args.cano_dist_thr, joint_cost_weight=args.lambda_joint)
            # new_trans_list: (T, 20, 4, 4), trans list of each part
            # new_connection: (E, 2), edge list
            new_seg, new_trans_list, new_connection = extract_kinematic(seg_part, trans_list, joint_connection)
            root_part = torch.mode(new_seg).values.item()
            root_trans = trans_list[:, root_part]
            align_trans_list = compute_align_trans(new_trans_list, root_trans)
            G, root_part, axis_list, moment_list, theta_list, distance_list, edge_index, joint_type_list = build_graph(
                new_connection, align_trans_list,
                verbose=False, revolute_only=False, root_part=root_part, return_joint_type=True)
            paths_to_base = nx.shortest_path(G, target=root_part)
            reverse_topo = list(reversed(list(nx.topological_sort(G))))  # traverse node from root to leaf
            model = KinematicModel(pose_len=pc_list.shape[0], seg_part=new_seg, cano_pc=cano_pc,
                                   knn=KNN(k=1, transpose_mode=True),
                                   edge_index=edge_index, paths_to_base=paths_to_base, reverse_topo=reverse_topo,
                                   axis_list=axis_list, moment_list=moment_list, theta_list=theta_list,
                                   distance_list=distance_list, root_trans=root_trans, joint_type_list=joint_type_list)
        else:
            checkpoint = torch.load(args.resume[0], map_location=device)
            model = KinematicModel(pose_len=pc_list.shape[0], seg_part=checkpoint["seg_part"],
                                   cano_pc=checkpoint["cano_pc"],
                                   knn=KNN(k=1, transpose_mode=True),
                                   edge_index=checkpoint["edge_index"],
                                   paths_to_base=checkpoint["paths_to_base"],
                                   reverse_topo=checkpoint["reverse_topo"],
                                   load_distance=True, load_root_trans=True,
                                   joint_type_list=checkpoint["joint_type_list"] if "joint_type_list" in checkpoint else None)

            model.load_state_dict(checkpoint["state_dict"], strict=True)
            print("=> loaded model checkpoint {}".format(args.resume[0]))
            if "cano_idx" in checkpoint:
                assert args.cano_idx == checkpoint["cano_idx"]

    else:
        raise ValueError("unknown model type {}".format(args.model))
    model.to(device)
    knn = KNN(k=1, transpose_mode=True)

    if args.evaluate:
        args.n_iter = 1
        model.eval()
    else:
        if args.model == "base":
            seg_params = filter(lambda p: p.requires_grad, model.seg_head.parameters())
            optimizer = torch.optim.Adam([{'params': [model.proposal_6d, model.proposal_t], 'lr': args.trans_lr},
                                          {'params': seg_params, 'lr': args.seg_lr}], lr=1e-3,
                                         weight_decay=args.weight_decay)
        else:
            model_params = filter(lambda p: p.requires_grad, model.parameters())
            optimizer = torch.optim.Adam(model_params, lr=args.trans_lr, weight_decay=args.weight_decay)

    n_iter = args.n_iter
    for i in tqdm(range(n_iter)):
        kwargs = {}
        if not args.evaluate and tau_func is not None:
            tau = tau_func(cur_iter=i + 1)
            kwargs["tau"] = tau
        pc_trans_list, seg_part, trans_list = model(cano_pc, **kwargs)
        if not args.evaluate:
            loss = 0
            losses = {}
            loss_info = ""

            # Default with chamfer loss
            dist_loss = recon_loss(pc_trans_list, pc_list, chamfer_dist=chamfer_dist)
            losses.update({"recon_loss": dist_loss.detach().cpu().numpy()})
            loss_info += f"iteration: {i} | recon Loss: {losses['recon_loss']:.3f} | "
            loss = loss + dist_loss

            if args.use_assign_loss and i >= args.assign_iter:
                if i == args.assign_iter or i % args.assign_gap == 0:
                    num_fps = pc_trans_list.shape[1] // args.downsample
                    src_idx = farthest_point_sample(cano_pc.unsqueeze(dim=0), num_fps).expand(pc_trans_list.shape[0],
                                                                                              num_fps)
                    pc_src = index_points(pc_trans_list, src_idx)
                    tgt_idx = farthest_point_sample(pc_list, num_fps)
                    pc_tgt = index_points(pc_list, tgt_idx)
                    with torch.no_grad():
                        cost = torch.cdist(pc_src, pc_tgt).cpu().numpy()
                    if not args.use_nproc:
                        indices = [linear_sum_assignment(c) for c in cost]
                    else:
                        indices = parallel_lap(cost, nproc=len(cost))
                    assign_indices = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for
                                      i, j in indices]
                else:
                    pc_src = index_points(pc_trans_list, src_idx)
                    pc_tgt = index_points(pc_list, tgt_idx)
                ass_src_idx = get_src_permutation_idx(assign_indices)
                ass_tgt_idx = get_tgt_permutation_idx(assign_indices)
                ass_loss = args.lambda_assign * ((pc_src[ass_src_idx] - pc_tgt[ass_tgt_idx]) ** 2).sum(dim=-1).sum()
                loss_info += f"opt assignment loss: {ass_loss:.3f} | "
                losses.update({"ass_loss": ass_loss.detach().cpu().numpy()})
                loss = loss + ass_loss

            if args.use_flow_loss:
                with torch.no_grad():
                    query_list = torch.cat((pc_trans_list[:dataset.cano_idx], cano_pc[None], pc_trans_list[dataset.cano_idx:]), dim=0)[:-1]
                    pairwise_flow_list = feature_propagation(query_list, pc_ref_list, flow_ref_list, False)

                complete_pred_pc_list = torch.cat((pc_trans_list[:dataset.cano_idx], cano_pc[None], pc_trans_list[dataset.cano_idx:]), dim=0)
                pred_flow_list = complete_pred_pc_list[1:, :, :] - complete_pred_pc_list[:-1, :, :]
                f_loss = args.lambda_flow * flow_loss(pairwise_flow_list,
                                                      pred_flow_list)  # flow_mask_list=pred_flow_mask_list

                loss_info += f"flow Loss: {f_loss:.3f} | "
                losses.update({"flow_loss": f_loss.detach().cpu().numpy()})
                loss = loss + f_loss

            loss_info += f"total Loss: {loss:.3f}"
            losses.update({"total_loss": loss.detach().cpu().numpy()})
            print(loss_info)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.update({"iter": i})

        if i % args.snapshot_gap == 0 or i == n_iter - 1:
            trans_list = trans_list.detach()  # [T-1, P, 4, 4]

            if i == n_iter - 1:
                if not args.evaluate:  # use the latest model weight after optimizer step
                    with torch.no_grad():
                        _, seg_part, trans_list = model(cano_pc)

                seg_part = denoise_seg_label(seg_part, cano_pc, knn, min_num=20)

                if not isinstance(model, KinematicModel) and len(torch.unique(seg_part)) > 1:
                    root_part = torch.mode(seg_part).values.item()
                    root_trans = trans_list[:, root_part]
                    align_trans_list = compute_align_trans(trans_list, root_trans)
                    seg_part = merging_wrapper(seg_part, align_trans_list, cano_pc, chamfer_dist, args.merge_thr,
                                               n_it=args.merge_it)

                if isinstance(model, KinematicModel) and hasattr(model, "edge_index"):
                    joint_connection_list = edge_index2edges(model.edge_index)
                    joint_connection = torch.from_numpy(np.array(joint_connection_list)).long().to(device)
                else:
                    root_part = torch.mode(seg_part).values.item()
                    root_trans = trans_list[:, root_part]
                    align_trans_list = compute_align_trans(trans_list, root_trans)
                    joint_connection = mst_wrapper(seg_part, align_trans_list, cano_pc, chamfer_dist, verbose=False,
                                                   num_fps=20,
                                                   cano_dist_thr=args.cano_dist_thr,
                                                   joint_cost_weight=args.lambda_joint)

                seg_part, trans_list, joint_connection = extract_kinematic(seg_part, trans_list, joint_connection)
                joint_connection_list = joint_connection.cpu().numpy().tolist()

                root_part = torch.mode(seg_part).values.item()
                root_trans = trans_list[:, root_part]

            pred_pc_list = compute_pc_transform(cano_pc, trans_list, seg_part)
            seg_part_np = seg_part.cpu().numpy()
            pred_pc_list_ = pred_pc_list.cpu().numpy()
            complete_pred_pc_list = np.concatenate((pred_pc_list_[:dataset.cano_idx], sample['cano_pc'][None], pred_pc_list_[dataset.cano_idx:]), axis=0)

            seg_part_list = seg_propagation_list(pc_list, pred_pc_list, seg_part, knn)
            T, P = complete_pc_list.shape[0], trans_list.shape[1]
            identity_matrix = torch.eye(4, device=cano_pc.device, dtype=cano_pc.dtype)[None, None].expand(1, P, 4, 4)
            complete_trans_list = torch.cat((trans_list[:cano_idx], identity_matrix, trans_list[cano_idx:]))
            complete_seg_list = torch.cat((seg_part_list[:cano_idx], seg_part[None], seg_part_list[cano_idx:]))
            full_flow = compute_full_flow(complete_pc_list, complete_seg_list, complete_trans_list)
            all_epe3d = eval_flow(full_flow, gt_full_flow)
            epe = np.mean(all_epe3d).item()
            epe = 100 * epe  # error in cm

            ri_list = []
            for (pred_seg, gt_seg) in zip(complete_seg_list, complete_gt_part_list):
                ri = eval_seg(gt_seg, pred_seg)
                ri_list.append(ri)

            ri_list = np.array(ri_list)
            per_ri = np.array(ri_list).mean()

            multi_ri = eval_seg(complete_seg_list.reshape(-1), complete_gt_part_list.reshape(-1))

            cd_dist = compute_chamfer_list(pred_pc_list_, sample['pc_list'], reduction="mean")
            cd_err = cd_dist.mean()
            cd_err = cd_err

            mse_dist = np.sqrt(((complete_pred_pc_list - sample['complete_gt_pc_list']) ** 2).sum(axis=-1)).mean(axis=1)
            recon_err = mse_dist.mean()
            recon_err = recon_err

            print(f'Flow eval: EPE: {epe:.3f}')
            print(f'Per-Scan Seg eval: RI: {per_ri:.3f}')
            print(f'Multi-Scan Seg eval: RI: {multi_ri:.3f}')
            print(f'Recon eval: recon: {recon_err:.3f}')

            if i == n_iter - 1:
                
                # visualize reconstructed point cloud sequence
                save_path = os.path.join(save_dir, f"recon.gif")
                vis_pc_seq(complete_pred_pc_list, pred_part=seg_part_np, name="reconstruct", save_path=save_path)
                print("save reconstruct pc vis to {}".format(save_path))
                
                save_path = os.path.join(save_dir, f"gt.gif")
                vis_pc_seq(sample['complete_gt_pc_list'], pred_part=sample['gt_cano_part'], name="gt", save_path=save_path)
                print("save gt pc vis to {}".format(save_path))

                save_path = os.path.join(save_dir, "seg.html")
                vis_pc(sample['cano_pc'], pred_part=seg_part_np, gt_part=sample['gt_cano_part'], save_path=save_path)
                print("save seg result to {}".format(save_path))

                save_path = os.path.join(save_dir, f"structure.html")
                vis_structure(sample['cano_pc'], seg_part.cpu().numpy(), joint_connection_list, save_path)
                print("save structure result to {}".format(save_path))

                f_result = open(os.path.join(save_dir, f"result.txt"), 'w')
                
                if not args.evaluate:  # save model prediction
                    ass_err = compute_ass_err(pred_pc_list, pc_list, use_nproc=True)
                    ass_err = ass_err
                    screw_err = compute_screw_cost(trans_list, joint_connection)
                    total_err = ass_err + screw_err
                    print(f'Energy eval: total: {total_err:.3f}')
                    f_result.write(f"ass_err: {ass_err:.3f}\n")
                    f_result.write(f"cd_err: {cd_err:.3f}\n")
                    f_result.write(f"screw_err: {screw_err:.3f}\n")
                    f_result.write(f"total_err: {total_err:.3f}\n\n")

                save_dict = {"pred_cano_part": seg_part_np,
                             "pred_pose_list": trans_list.cpu().numpy(),
                             "cano_idx": dataset.cano_idx}

                save_dict.update({"joint_connection": joint_connection_list})
                save_dict.update(sample)
                with open(os.path.join(save_dir, f"result.pkl"), 'wb') as f:
                    pickle.dump(save_dict, f)

                f_result.write(f"recon_err: {recon_err:.3f}\n")
                f_result.write(f"flow_epe: {epe:.3f}\n")
                f_result.write(f"per_scan_seg_ri: {per_ri:.3f}\n")
                f_result.write(f"multi_scan_seg_ri: {multi_ri:.3f}\n")
                f_result.close()

                if not args.evaluate:
                    model_save_path = os.path.join(save_dir, "model.pth.tar")
                    model_dict = {"state_dict": model.state_dict(), "tau": tau, "cano_idx": args.cano_idx}

                    if isinstance(model, KinematicModel):
                        if hasattr(model, "seg_part"):
                            model_dict.update({"seg_part": model.seg_part})
                        if hasattr(model, "cano_pc"):
                            model_dict.update({"cano_pc": model.cano_pc})
                        if hasattr(model, "edge_index"):
                            model_dict.update({"edge_index": model.edge_index})
                        if hasattr(model, "paths_to_base"):
                            model_dict.update({"paths_to_base": model.paths_to_base})
                        if hasattr(model, "reverse_topo"):
                            model_dict.update({"reverse_topo": model.reverse_topo})
                        if hasattr(model, "joint_type_list"):
                            model_dict.update({"joint_type_list": model.joint_type_list})

                    torch.save(model_dict, model_save_path)
    print("all done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sapien")
    # common
    parser.add_argument("--manual_seed", default=2, type=int, help="manual seed")
    parser.add_argument("--resume", type=str, nargs="+", metavar="PATH",
                        help="path to latest checkpoint (default: none)")
    parser.add_argument("--evaluate", dest="evaluate", action="store_true", help="evaluate mode")
    parser.add_argument("--snapshot_gap", default=100, type=int,
                        help="How often to take a snapshot vis of the training")
    parser.add_argument("--use_cuda", default=1, type=int, help="use GPU (default: True)")

    # dataset
    parser.add_argument("--cano_idx", default=0, type=int, help="cano frame idx")
    parser.add_argument("--seq_path", default="/home/shaowei3/datasets/articulation/robot/nao/test_1", type=str)
    
    # optimization
    parser.add_argument("--start_tau", default=1, type=float, help="gumbel softmax start temperature")
    parser.add_argument("--end_tau", default=1, type=float, help="gumbel softmax end temperature")
    parser.add_argument("--seg_lr", default=1e-3, type=float, help="seg MLP learning rate")
    parser.add_argument("--trans_lr", default=1e-2, type=float, help="seg MLP learning rate")
    parser.add_argument("--weight_decay", default=0, type=float)

    parser.add_argument("--n_iter", default=2000, type=int, help="number of optimization iterations")
    parser.add_argument("--assign_iter", default=1000, type=int, help="iteration apply assignment loss")

    # network
    parser.add_argument("--num_parts", default=10, type=int, help="seg MLP number of parts")
    parser.add_argument("--model", default="base", type=str, choices=['base', 'kinematic'], help="model type")
    parser.add_argument("--base_result_path", default=None, type=str, help="kinematic model initialization")

    # flow
    parser.add_argument("--use_flow_loss", action="store_true", help="use flow loss")

    # other constraints
    parser.add_argument("--use_assign_loss", action="store_true", help="use pc assignment loss")
    
    parser.add_argument("--use_nproc", action="store_true", help="use multi process to compute assignment loss")
    parser.add_argument("--downsample", default=1, type=int, help="downsample rate when computing assignment loss")
    parser.add_argument("--assign_gap", default=5, type=int, help="assignment loss gap")

    # loss weight
    parser.add_argument("--lambda_assign", default=3e-1, type=float, help="assignment loss weight")
    parser.add_argument("--lambda_flow", default=1, type=float, help="flow loss weight")
    parser.add_argument("--lambda_joint", default=1e-1, type=float, help="joint cost/loss weight")

    # structure_utils
    parser.add_argument("--cano_dist_thr", default=1e-2, type=float,
                        help="mst cano dist threshold (below consider an edge candidate)")
    parser.add_argument("--merge_thr", default=3e-2, type=float, help="graph geo merging threshold")
    parser.add_argument("--merge_it", default=3, type=int, help="graph geo merging iteration")

    # utils func
    parser.add_argument("--save_root", default="exp", type=str, help="results saving path")
   
    # sapien utils
    parser.add_argument("--sapien_base_folder", default="data/mbs-sapien", type=str, help="sapien dataset base folder")
    parser.add_argument("--sapien_idx", default=212, type=int, help="sapien dataset test index")
    parser.add_argument("--flow_model_config_path", type=str, default="msync/config/articulated-full.yaml")
    parser.add_argument("--flow_model_path", type=str, default="msync/ckpt/articulated-full/best.pth.tar")
    
    args = parser.parse_args()
    os.makedirs(args.save_root, exist_ok=True)
    main(args)