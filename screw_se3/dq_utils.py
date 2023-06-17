import numpy as np
import torch
import math
import warnings
from .geo_utils import matrix_to_quaternion


# ref: https://gist.github.com/Flunzmas/d9485d9fee6244b544e7e75bdc0c352c


def dq_mul(dq1, dq2):
    """
    Multiply dual quaternion dq1 with dq2.
    Expects two equally-sized tensors of shape [*, 8], where * denotes any number of dimensions.
    Returns dq1*dq2 as a tensor of shape [*, 8].
    """
    assert dq1.shape[-1] == 8
    assert dq2.shape[-1] == 8

    dq1_r, dq1_d = torch.split(dq1, [4, 4], dim=-1)
    dq2_r, dq2_d = torch.split(dq2, [4, 4], dim=-1)

    dq_prod_r = q_mul(dq1_r, dq2_r)
    dq_prod_d = q_mul(dq1_r, dq2_d) + q_mul(dq1_d, dq2_r)
    dq_prod = torch.cat([dq_prod_r, dq_prod_d], dim=-1)
    return dq_prod


def dq_translation(dq):
    """
    Returns the translation component of the input dual quaternion tensor of shape [*, 8].
    Translation is returned as tensor of shape [*, 3].
    """
    assert dq.shape[-1] == 8

    dq_r, dq_d = torch.split(dq, [4, 4], dim=-1)
    mult = q_mul((2.0 * dq_d), q_conjugate(dq_r))
    return mult[..., 1:]


def dq_normalize(dq):
    """
    Normalize the coefficients of a given dual quaternion tensor of shape [*, 8].
    """
    assert dq.shape[-1] == 8

    dq_r = dq[..., :4]
    norm = torch.sqrt(torch.sum(torch.square(dq_r), dim=-1))  # ||q|| = sqrt(w²+x²+y²+z²)
    assert not torch.any(torch.isclose(norm, torch.zeros_like(norm, device=dq.device)))  # check for singularities
    return torch.div(dq, norm[:, None])  # dq_norm = dq / ||q|| = dq_r / ||dq_r|| + dq_d / ||dq_r||


def dq_quaternion_conjugate(dq):
    """
    Returns the quaternion conjugate of the input dual quaternion tensor of shape [*, 8].
    The quaternion conjugate is composed of the complex conjugates of the real and the dual quaternion.
    """

    assert dq.shape[-1] == 8

    conj = torch.tensor([1, -1, -1, -1, 1, -1, -1, -1], device=dq.device)  # multiplication coefficients per element
    return dq * conj.expand_as(dq)


def q_mul(q1, q2):
    """
    Multiply quaternion q1 with q2.
    Expects two equally-sized tensors of shape [*, 4], where * denotes any number of dimensions.
    Returns q1*q2 as a tensor of shape [*, 4].
    """
    assert q1.shape[-1] == 4
    assert q2.shape[-1] == 4
    original_shape = q1.shape

    # Compute outer product
    terms = torch.bmm(q2.view(-1, 4, 1), q1.view(-1, 1, 4))
    w = terms[:, 0, 0] - terms[:, 1, 1] - terms[:, 2, 2] - terms[:, 3, 3]
    x = terms[:, 0, 1] + terms[:, 1, 0] - terms[:, 2, 3] + terms[:, 3, 2]
    y = terms[:, 0, 2] + terms[:, 1, 3] + terms[:, 2, 0] - terms[:, 3, 1]
    z = terms[:, 0, 3] - terms[:, 1, 2] + terms[:, 2, 1] + terms[:, 3, 0]

    q1q2 = torch.stack((w, x, y, z), dim=1).view(original_shape)
    return q1q2


def wrap_angle(theta):
    """
    Helper method: Wrap the angles of the input tensor to lie between -pi and pi.
    Odd multiples of pi are wrapped to +pi (as opposed to -pi).
    """
    pi_tensor = torch.ones_like(theta, device=theta.device) * math.pi
    result = ((theta + pi_tensor) % (2 * pi_tensor)) - pi_tensor
    result[result.eq(-pi_tensor)] = math.pi

    return result


def q_angle(q):
    """
    Determine the rotation angle of given quaternion tensors of shape [*, 4].
    Return as tensor of shape [*, 1]
    """
    assert q.shape[-1] == 4

    q = q_normalize(q)
    q_re, q_im = torch.split(q, [1, 3], dim=-1)
    norm = torch.linalg.norm(q_im, dim=-1).unsqueeze(dim=-1)
    angle = 2.0 * torch.atan2(norm, q_re)

    return angle  # wrap_angle(angle) !!! very careful


def q_normalize(q):
    """
    Normalize the coefficients of a given quaternion tensor of shape [*, 4].
    """
    assert q.shape[-1] == 4

    norm = torch.sqrt(torch.sum(torch.square(q), dim=-1))  # ||q|| = sqrt(w²+x²+y²+z²)
    assert not torch.any(torch.isclose(norm, torch.zeros_like(norm, device=q.device)))  # check for singularities
    return torch.div(q, norm[:, None])  # q_norm = q / ||q||


def q_conjugate(q):
    """
    Returns the complex conjugate of the input quaternion tensor of shape [*, 4].
    """
    assert q.shape[-1] == 4

    conj = torch.tensor([1, -1, -1, -1], device=q.device)  # multiplication coefficients per element
    return q * conj.expand_as(q)


def transform_to_dq(T):
    q_r = matrix_to_quaternion(T[:, :3, :3])
    q_d = 0.5 * q_mul(torch.cat((torch.zeros((T.shape[0], 1), dtype=T.dtype, device=T.device), T[:, :3, 3]), dim=1),
                      q_r)
    dq = torch.cat([q_r, q_d], dim=1)
    return dq


def dq_to_screw(dq, eps=1e-6):
    """
    Return the screw parameters that describe the rigid transformation encoded in the input dual quaternion.
    Input shape: [*, 8]
    Output:
     - Plucker coordinates (l, m) for the roto-translation axis (both of shape [*, 3])
     - Amount of rotation and translation around/along the axis (both of shape [*])
    """
    assert dq.shape[-1] == 8

    dq_r, dq_d = torch.split(dq, [4, 4], dim=-1)
    theta = q_angle(dq_r)  # shape: [b, 1]
    theta_sq = theta.squeeze(dim=-1)
    no_rot = torch.logical_or(theta_sq.abs() < eps, (theta_sq - math.pi).abs() < eps)
    with_rot = ~no_rot
    dq_t = dq_translation(dq)

    l = torch.zeros(*dq.shape[:-1], 3, device=dq.device)
    d = torch.zeros(*dq.shape[:-1], device=dq.device)

    l[with_rot] = dq_r[with_rot, 1:] / torch.sin(theta[with_rot] / 2)
    d[no_rot] = torch.linalg.norm(dq_t[no_rot], dim=-1)
    l[no_rot] = dq_t[no_rot] / (d[no_rot, None] + 1e-10)

    # align axis
    up_axis = torch.tensor([1, 1, 1], dtype=l.dtype, device=l.device).unsqueeze(dim=0)
    cos = (l * up_axis).sum(dim=-1, keepdim=True)
    theta = torch.where(cos >= 0, theta, -theta)
    l = torch.where(cos >= 0, l, -l)
    d[no_rot] = torch.where(cos[no_rot, 0] >= 0, d[no_rot], -d[no_rot])
    d[with_rot] = (dq_t[with_rot] * l[with_rot]).sum(dim=-1)  # batched dot product

    no_trans = torch.isclose(d, torch.zeros_like(d, device=dq.device))
    unit_transform = torch.logical_and(no_rot, no_trans)
    l[unit_transform, 0] = 1
    if unit_transform.sum() > 0:
        warnings.warn("Warning: input contains identity matrix, the screw axis is not determined")

    theta[no_rot] = eps
    t_l_cross = torch.cross(dq_t, l, dim=-1)
    m = 0.5 * (t_l_cross + torch.cross(l, t_l_cross / torch.tan(theta / 2), dim=-1))
    return l, m, theta.squeeze(-1), d