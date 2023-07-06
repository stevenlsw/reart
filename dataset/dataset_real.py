import os
import glob
import trimesh
import numpy as np


def normalize_mesh(cano_mesh):
    norm_cano_mesh = cano_mesh.copy()
    v = norm_cano_mesh.vertices
    vmax, vmin = v.max(axis=0), v.min(axis=0)
    diag = vmax - vmin
    c = v.mean(axis=0)
    norm = 1 / np.linalg.norm(diag)
    vNorm = (v - c) * norm
    norm_cano_mesh.vertices = vNorm
    return norm_cano_mesh, np.array(c), np.array(norm)


class Sequence(object):
    def __init__(self, seq_dir, num_points=4096, cano_idx=0):
        self.num_points = num_points
        self.cano_idx = cano_idx
        self.seq_dir = seq_dir
        files = glob.glob(os.path.join(self.seq_dir, "*.*"))
        files = sorted(files, key=lambda file_name: int(file_name.split("/")[-1].split(".")[0].split("_")[-1]))
        self.mesh_list = []
        for file in files:
            if file.split(".")[-1] == "glb":
                scene = trimesh.load_mesh(file)
                geometries = list(scene.geometry.values())
                mesh = geometries[0]
            else:
                mesh = trimesh.load_mesh(file)
            self.mesh_list.append(mesh)
        cano_mesh = self.mesh_list[cano_idx]
        norm_cano_mesh, centroid, scale = normalize_mesh(cano_mesh)
        self.centroid = centroid
        self.scale = scale

    def __len__(self):
        return 1

    def __getitem__(self, item):
        complete_pc_list = []
        for mesh in self.mesh_list:
            pc, face_idx = trimesh.sample.sample_surface(mesh, count=self.num_points)
            complete_pc_list.append(pc)

        complete_pc_list = np.stack(complete_pc_list).astype('float32')
        cano_pc = complete_pc_list[self.cano_idx]

        pc_list = np.concatenate((complete_pc_list[:self.cano_idx, :], complete_pc_list[self.cano_idx+1:, :]), axis=0)
        sample = {'cano_pc': cano_pc,
                  'pc_list': pc_list,
                  'complete_pc_list': complete_pc_list}
        return sample