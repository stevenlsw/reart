import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.model_utils import th_with_zeros, knn_query, create_transformation
from utils.kinematic_utils import fk
from networks.blocks import MLPConv1d
from screw_se3 import rotation_6d_to_matrix, matrix_to_rotation_6d


class BaseModel(nn.Module):
    def __init__(self, num_parts, pose_len, joint_trajectory=None, init_6d=None, init_t=None):
        super(BaseModel, self).__init__()
        self.num_parts = num_parts
        self.pose_len = pose_len
        initial_connection = torch.stack([torch.arange(self.num_parts - 1),
                                          torch.arange(self.num_parts - 1) + 1], dim=1)  # (n_parts - 1, 2)
        self.register_buffer("joint_connection", initial_connection.long())
        self.seg_head = MLPConv1d(3, (128, num_parts,), bn=False, gn=False, last_activation='none')

        self.joint_trajectory = joint_trajectory
        if joint_trajectory is None:
            if init_6d is None:
                self.proposal_6d = nn.Parameter(torch.tensor([[[1, 0, 0, 0, 1, 0]]]).float().repeat((pose_len, num_parts, 1)), requires_grad=True)
            else:
                self.proposal_6d = nn.Parameter(init_6d, requires_grad=False)

            if init_t is None:
                self.proposal_t = nn.Parameter(torch.tensor([[[0, 0, 0]]]).float().repeat((pose_len, num_parts, 1)), requires_grad=True)
            else:
                self.proposal_t = nn.Parameter(init_t, requires_grad=False)

    def seg_forward(self, cano_pc, **kwargs):
        # cano_pc: [N, 3]
        input = cano_pc.permute(1, 0).unsqueeze(dim=0)  # [1, 3, N]
        seg = self.seg_head(input).squeeze(dim=0).permute(1, 0)  # [N, P]
        return seg.argmax(dim=-1) if ("argmax" in kwargs and kwargs["argmax"]) else seg

    def forward(self, cano_pc, **kwargs):
        # cano_pc: [N, 3]
        N = cano_pc.shape[0]
        input = cano_pc.permute(1, 0).unsqueeze(dim=0) # [1, 3, N]
        seg = self.seg_head(input).squeeze(dim=0).permute(1, 0)  # [N, P]
        weight = F.gumbel_softmax(seg, tau=kwargs["tau"] if "tau" in kwargs else 1.0, hard=True)  # [N, P]
        if self.joint_trajectory is not None:
            pose_len = kwargs["pose_len"] if "pose_len" in kwargs else self.pose_len
            frame_id = torch.arange(pose_len, dtype=cano_pc.dtype, device=cano_pc.device)
            rotation, translation = self.joint_trajectory(frame_id)
            rotation = rotation.reshape(-1, 3, 3)
            translation = translation.reshape(-1, 3)
        else:
            if "proposal_6d" in kwargs:
                proposal_6d = kwargs["proposal_6d"]
            else:
               proposal_6d = self.proposal_6d
            if "proposal_t" in kwargs:
                proposal_t = kwargs["proposal_t"]
            else:
                proposal_t = self.proposal_t
            rotation = rotation_6d_to_matrix(proposal_6d.reshape(-1, 6))  # [T-1 * P, 3, 3]
            translation = proposal_t.reshape(-1, 3)  # [T-1 * P, 3]
        T = proposal_6d.shape[0]
        cano_pc = cano_pc.unsqueeze(dim=0).unsqueeze(dim=1).expand(T, self.num_parts, N, 3).reshape(-1, N, 3)  # [T-1 * P, N, 3]
        pc_trans_list = torch.bmm(cano_pc, rotation.transpose(1, 2)) + translation[:, None, :]  # [T-1 * P, N, 3]
        pc_trans_list = pc_trans_list.reshape(T, self.num_parts, N, 3)  # [T-1, P, N, 3]
        trans = torch.cat((rotation, translation[:, :, None]), dim=2)
        trans = th_with_zeros(trans)
        trans_list = trans.reshape(T, self.num_parts, 4, 4)  # (T-1, P, 4, 4)
        pc_trans_list = (weight.permute(1, 0)[None, :, :, None] * pc_trans_list).sum(dim=1)  # [T-1, N, 3]
        return pc_trans_list, seg.argmax(dim=-1), trans_list


class KinematicModel(nn.Module):
    def __init__(self, pose_len, seg_part, cano_pc, knn, **kwargs):
        super(KinematicModel, self).__init__()
        # seg_part: (N, ), segmentation of canonical frame
        # cano_pc: (N, 3), cano_pc of seg_part
        # knn: knn=1 instance of KNN(k=1, transpose_mode=True)

        # paths_to_base: returned from nx.shortest_path(G, target=root_part), dict
        # reverse_topo: returned from list(reversed(list(nx.topological_sort(G)))), traverse node from root to leaf
        # edge_index: dict, key: edge name: "_".join([str(child_part_id), str(parent_part_id)]), value: edge index

        self.seg_part = seg_part.long()
        self.cano_pc = cano_pc
        self.pose_len = pose_len
        self.num_parts = len(torch.unique(self.seg_part))
        self.knn = knn
        if knn is not None:
            assert self.knn.k == 1

        self.edge_index = kwargs["edge_index"]
        self.paths_to_base = kwargs["paths_to_base"]
        self.reverse_topo = kwargs["reverse_topo"]
        E = len(self.edge_index)
        assert self.num_parts == E + 1  # P = E + 1

        if "axis_list" in kwargs:
            assert self.num_parts == kwargs["axis_list"].shape[0] + 1  # P = E + 1
            self.axis_list = torch.nn.Parameter(kwargs["axis_list"], requires_grad=True)
        else:
            self.axis_list = torch.nn.Parameter(torch.zeros(E, 3), requires_grad=True)
        if "moment_list" in kwargs:
            assert self.num_parts == kwargs["moment_list"].shape[0] + 1  # P = E + 1
            self.moment_list = torch.nn.Parameter(kwargs["moment_list"], requires_grad=True)
        else:
            self.moment_list = torch.nn.Parameter(torch.zeros(E, 3), requires_grad=True)
        if "theta_list" in kwargs:
            self.theta_list = torch.nn.Parameter(kwargs["theta_list"], requires_grad=True)
        else:  # used for evaluation loading model
            self.theta_list = torch.nn.Parameter(torch.zeros(pose_len, E), requires_grad=True)

        if "distance_list" in kwargs:
            self.distance_list = torch.nn.Parameter(kwargs["distance_list"], requires_grad=True)
        elif "load_distance" in kwargs and kwargs["load_distance"]:
            self.distance_list = torch.nn.Parameter(torch.zeros(pose_len, E), requires_grad=True)

        if "root_trans" in kwargs:
            root_6d = matrix_to_rotation_6d(kwargs["root_trans"][:, :3, :3])
            self.root_6d = torch.nn.Parameter(root_6d, requires_grad=True)
            translation = kwargs["root_trans"][:, :3, 3]
            self.root_t = torch.nn.Parameter(translation, requires_grad=True)
        elif "load_root_trans" in kwargs and kwargs["load_root_trans"]:
            self.root_6d = nn.Parameter(torch.tensor([[1, 0, 0, 0, 1, 0]]).float().repeat((pose_len, 1)), requires_grad=True)
            self.root_t = nn.Parameter(torch.tensor([[0, 0, 0]]).float().repeat((pose_len, 1)), requires_grad=True)

        if "joint_type_list" in kwargs:
            self.joint_type_list = kwargs["joint_type_list"]
        else:
            self.joint_type_list = None

    def seg_forward(self, input_pc, **kwargs):
        # cano_pc: [*, 3]
        seg_part = knn_query(input_pc, self.cano_pc, self.seg_part, self.knn)
        return seg_part

    def forward(self, input_pc, **kwargs):
        seg_part = knn_query(input_pc, self.cano_pc, self.seg_part, self.knn)
        if "theta_list" in kwargs:
            theta_list = kwargs["theta_list"]
        else:
            theta_list = self.theta_list

        if hasattr(self, "distance_list"):
            distance_list = self.distance_list
        else:
            distance_list = None

        T, N = theta_list.shape[0], input_pc.shape[0] # T here is T-1
        weight = F.one_hot(seg_part, num_classes=self.num_parts)  # [N, P]
        trans_list = fk(self.paths_to_base, self.reverse_topo, self.edge_index, self.axis_list, self.moment_list,
                        theta_list, distance_list=distance_list,
                        joint_type_list=self.joint_type_list) # (T-1, P, 4, 4)
        if hasattr(self, "root_6d") and hasattr(self, "root_t"):
            root_rotation = rotation_6d_to_matrix(self.root_6d) # (T, 3, 3)
            root_translation = self.root_t[:, :, None] # (T-1, 3, 1)
            root_trans = create_transformation(root_rotation, root_translation)
            root_trans_list = root_trans[:, None, :, :].expand(trans_list.shape)
            trans_list = torch.matmul(root_trans_list, trans_list)

        rotation, translation = trans_list[:, :, :3, :3].reshape(-1, 3, 3), trans_list[:, :, :3, 3].reshape(-1, 3)
        input_pc = input_pc.unsqueeze(dim=0).unsqueeze(dim=1).expand(T, self.num_parts, N, 3).reshape(-1, N, 3)  # [T-1 * P, N, 3]
        pc_trans_list = torch.bmm(input_pc, rotation.transpose(1, 2)) + translation[:, None, :]  # [T-1 * P, N, 3]
        pc_trans_list = pc_trans_list.reshape(T, self.num_parts, N, 3)  # [T-1, P, N, 3]
        pc_trans_list = (weight.permute(1, 0)[None, :, :, None] * pc_trans_list).sum(dim=1)  # [T-1, N, 3]
        return pc_trans_list, seg_part, trans_list