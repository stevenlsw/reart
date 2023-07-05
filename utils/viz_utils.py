import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import cm
import plotly.graph_objects as go
import io
import imageio
from PIL import Image


def vis_pc(pc, pred_part, pc_gt=None, gt_part=None, name="pred", save_path=None):
    jet = plt.get_cmap('jet')
    cNorm = colors.Normalize(vmin=0, vmax=len(set(pred_part.tolist())))
    scalarMap = cm.ScalarMappable(norm=cNorm, cmap=jet)
    pc_colors = np.empty((len(pc), 3))
    unique_part_ids = list(set(pred_part.tolist()))
    for color_idx, unique_id in enumerate(unique_part_ids):
        pc_idx = pred_part == unique_id
        color = scalarMap.to_rgba(color_idx)[:3]
        pc_colors[pc_idx, :] = np.array(color)[None, :]
    fig = go.Figure(data=go.Scatter3d(x=pc[:, 0], y=pc[:, 2], z=pc[:, 1],
                                      mode='markers', name=name,
                                      marker=dict(color=pc_colors, size=5)))
    if gt_part is not None:
        if pc_gt is None:
            pc_gt = pc.copy()
        else:
            pc_gt = pc_gt.copy()
        src_scale = pc[:, 0].max() - pc[:, 0].min()
        t = max(0, pc[:, 0].max() - pc[:, 0].min() + 0.4 * src_scale)
        pc_gt[:, 0] = pc_gt[:, 0] + t
        cNorm = colors.Normalize(vmin=0, vmax=len(set(gt_part.tolist())))
        scalarMap = cm.ScalarMappable(norm=cNorm, cmap=jet)
        gt_colors = np.empty((len(gt_part), 3))
        unique_part_ids = list(set(gt_part.tolist()))
        for color_idx, unique_id in enumerate(unique_part_ids):
            pc_idx = gt_part == unique_id
            color = scalarMap.to_rgba(color_idx)[:3]
            gt_colors[pc_idx, :] = np.array(color)[None, :]
        fig.add_trace(go.Scatter3d(x=pc_gt[:, 0], y=pc_gt[:, 2], z=pc_gt[:, 1],
                                mode='markers', name="gt",
                                marker=dict(color=gt_colors, size=5)))

    fig.update_layout(showlegend=True, scene=dict(xaxis_title='x', yaxis_title='z', zaxis_title='y',
                                                  xaxis=dict(),
                                                  yaxis=dict(),
                                                  zaxis=dict(),
                                                  aspectmode='data'))
    if save_path is not None:
        fig.write_html(save_path)
    return fig


def plotly_fig2array(fig):
    fig_bytes = fig.to_image(format="png")
    buf = io.BytesIO(fig_bytes)
    img = Image.open(buf)
    return np.asarray(img)


def vis_pc_seq(pc_list, pred_part=None, gt_part=None, name="pred", save_path=None):
    imgs = []
    for pc in pc_list:
        if pred_part is None:
            color = np.expand_dims([0, 0, 1], axis=0).repeat(len(pc), axis=0)
            fig = go.Figure(data=go.Scatter3d(x=pc[:, 0], y=pc[:, 2], z=pc[:, 1],
                                    mode='markers', name=name,
                                    marker=dict(color=color, size=2)))
        else:
            fig = vis_pc(pc, pred_part=pred_part, gt_part=gt_part, name=name)
        imgs.append(plotly_fig2array(fig))
    if save_path is not None:
        imageio.mimsave(save_path, imgs, format='gif', duration=0.3)
    return imgs
            

def cylinder(r, h, a=0, nt=100, nv=50):
    """
    parametrize the cylinder of radius r, height h, base point a
    """
    theta = np.linspace(0, 2 * np.pi, nt)
    v = np.linspace(a, a + h, nv)
    theta, v = np.meshgrid(theta, v)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = v
    return x, y, z


def vis_structure(pc, pc_part, edges_list, save_path):
    num_parts = len(set(pc_part.tolist()))
    jet = plt.get_cmap('jet')
    cNorm = colors.Normalize(vmin=0, vmax=num_parts)
    scalarMap = cm.ScalarMappable(norm=cNorm, cmap=jet)
    unique_part_ids = list(set(pc_part.tolist()))
    unique_part_ids.sort()
    fig = go.Figure()
    centroid_dict = {}
    pc_colors = np.empty((len(pc_part), 3))
    for color_idx, unique_id in enumerate(unique_part_ids):
        pc_idx = pc_part == unique_id
        color = scalarMap.to_rgba(color_idx)[:3]
        pc_colors[pc_idx, :] = np.array(color)[None, :]
        centroid = pc[pc_idx].mean(axis=0, keepdims=True)
        centroid_dict[unique_id] = centroid
        fig.add_trace(go.Scatter3d(x=centroid[:, 0], y=centroid[:, 2], z=centroid[:, 1],
                                   mode='markers', name="joint_{}".format(unique_id),
                                   marker=dict(color="black", size=20)))
    fig.add_trace(go.Scatter3d(x=pc[:, 0], y=pc[:, 2], z=pc[:, 1],
                               mode='markers', name="pc",
                               marker=dict(color=pc_colors, size=5)))
    for edge in edges_list:
        parent_part_id, child_part_id = edge
        if parent_part_id not in centroid_dict:
            print(f"empty edges {parent_part_id}-{child_part_id}, {parent_part_id} has no points")
            continue
        if child_part_id not in centroid_dict:
            print(f"empty edges {parent_part_id}-{child_part_id}, {child_part_id} has no points")
            continue
        parent_centroid, child_centroid = centroid_dict[parent_part_id].squeeze(axis=0), centroid_dict[
            child_part_id].squeeze(axis=0)
        x, y, z = cylinder(r=0.01, h=np.linalg.norm(parent_centroid - child_centroid) + 1e-6)
        cy_pc = np.stack([x, y, z], axis=2)  # (50, 100, 3)
        line1 = np.array([0.0, 0.0, 1.0])
        line2 = (parent_centroid - child_centroid) / (np.linalg.norm(parent_centroid - child_centroid) + 1e-6)
        v = np.cross(line1, line2)
        c = np.dot(line1, line2) + 1e-8
        k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + k + np.matmul(k, k) * (1 / (1 + c))
        if np.abs(c + 1.0) < 1e-4:  # the above formula doesn't apply when cos(∠(𝑎,𝑏))=−1
            R = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
        t = child_centroid + 5e-3 * line2
        cy_pc = cy_pc @ R.T + t
        fig.add_trace(go.Surface(x=cy_pc[:, :, 0], y=cy_pc[:, :, 2], z=cy_pc[:, :, 1], opacity=0.5, showscale=False))

    fig.update_layout(showlegend=True, scene=dict(xaxis_title='x', yaxis_title='z', zaxis_title='y',
                                                  xaxis=dict(),
                                                  yaxis=dict(),
                                                  zaxis=dict(),
                                                  aspectmode='data'))
    fig.update_coloraxes(showscale=False)
    fig.write_html(save_path)
    return fig