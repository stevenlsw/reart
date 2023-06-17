import math
import torch
from .geo_utils import se3_exp_map


def screw_param_to_exponential_coordinates(l, m, theta, d):
    # l: (B, 3), rotation axis
    # m: (B, 3), moment
    # theta: (B, )
    # d: (B, )
    eps = 1e-6
    no_rot = torch.logical_or(theta.abs() < eps, (theta - math.pi).abs() < eps)
    with_rot = ~no_rot
    q = torch.zeros(l.shape[0], 3, device=l.device)
    q[with_rot] = torch.cross(l[with_rot], m[with_rot], dim=-1)
    w = torch.zeros(l.shape[0], 3, device=l.device)
    v = torch.zeros(l.shape[0], 3, device=l.device)
    w[with_rot] = l[with_rot]
    h = d[with_rot] / theta[with_rot]
    v[with_rot] = torch.cross(q[with_rot], l[with_rot], dim=-1) + h[:, None] * l[with_rot]
    v[no_rot] = l[no_rot]
    screw_axis = torch.cat((w, v), dim=1)
    return screw_axis * theta[:, None]



def transform_from_exponential_coordinates(log_transform):
    log_transform_input = torch.cat((log_transform[:, 3:], log_transform[:, :3]), dim=1)
    transform_matrix = se3_exp_map(log_transform_input)
    return transform_matrix.permute(0, 2, 1)