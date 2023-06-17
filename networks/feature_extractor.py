import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings


from networks.pointnet2_utils import PointNetSetAbstractionMsg, PointNetSetAbstraction, PointNetFeaturePropagation


class PointNet2Msg2(nn.Module):
    def __init__(self, out_dim, normal_channel=False):
        super(PointNet2Msg2, self).__init__()
        if normal_channel:
            additional_channel = 3
        else:
            additional_channel = 0
        self.out_dim = out_dim
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstractionMsg(512, [0.05, 0.1, 0.2], [32, 64, 128], 3 + additional_channel,
                                             [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(128, [0.2, 0.4], [64, 128], 128 + 128 + 64,
                                             [[128, 128, 256], [128, 196, 256]])
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=512 + 3,
                                          mlp=[256, 512, 1024], group_all=True)
        self.fp3 = PointNetFeaturePropagation(in_channel=1536, mlp=[256, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=576, mlp=[256, 128])
        self.fp1 = PointNetFeaturePropagation(in_channel=134 + additional_channel, mlp=[128, 128])
        self.conv1 = nn.Conv1d(128, out_dim, 1)
        self.bn1 = nn.BatchNorm1d(out_dim)

    def forward(self, xyz):
        # Set Abstraction layers
        B, C, N = xyz.shape
        if self.normal_channel:
            l0_points = xyz
            l0_xyz = xyz[:, :3, :]
        else:
            l0_points = xyz
            l0_xyz = xyz
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        # Feature Propagation layers
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, torch.cat([l0_xyz, l0_points], 1), l1_points)
        # FC layers
        feat = F.relu(self.bn1(self.conv1(l0_points)))
        return feat


def rec_freeze(model):
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.momentum = 0
    for name, child in model.named_children():
        for param in child.parameters():
            param.requires_grad = False
        rec_freeze(child)
        

def get_extractor(args):
    feature_extractor = PointNet2Msg2(out_dim=64)  # predefined in correspondence model
    feature_extractor = torch.nn.DataParallel(feature_extractor)

    if args.use_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
        feature_extractor.cuda()
    else:
        device = torch.device("cpu")
    assert args.corr_model_path is not None, "pretrained correspondence model is None!"
    checkpoint = torch.load(args.corr_model_path, map_location=device)
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    state_dict = {k.replace('net.', ''): v for k, v in state_dict.items()}
    print("=> loaded correspondence model checkpoint '{}'".format(args.corr_model_path))
    missing_states = set(feature_extractor.state_dict().keys()) - set(state_dict.keys())
    if len(missing_states) > 0:
        warnings.warn("Missing keys ! : {}".format(missing_states))
    feature_extractor.load_state_dict(state_dict, strict=True)

    rec_freeze(feature_extractor.module)

    return feature_extractor

