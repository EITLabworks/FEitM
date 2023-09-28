import numpy as np
from pyeit.mesh import PyEITMesh, distmesh, shape
from pyeit.mesh.wrapper import PyEITAnomaly_Circle


def create_head_symm_mesh(h0: float = 0.1, n_el: int = 16) -> PyEITMesh:
    """create head phantom mesh (symmetric)"""
    p, t = distmesh.build(
        fd=shape.head_symm, fh=shape.area_uniform, pfix=shape.head_symm_pfix, h0=h0
    )
    mesh = PyEITMesh(p, t)
    mesh.el_pos = np.arange(n_el)
    return mesh


def set_sqr_perm(
    mesh: PyEITMesh,
    x_m: float,
    x_p: float,
    y_m: float,
    y_p: float,
    obj_perm: float = 10.0,
) -> PyEITMesh:
    x_min = x_m
    x_max = x_p
    y_max = y_p
    y_min = y_m
    mesh.perm = mesh.perm_array

    for i, element in enumerate(mesh.element):
        x_ele = np.mean(mesh.node[element][:, 0])
        y_ele = np.mean(mesh.node[element][:, 1])
        if x_ele > x_min and x_ele < x_max and y_ele > y_min and y_ele < y_max:
            mesh.perm[i] = obj_perm
    return mesh


def move_el(
    mesh: PyEITMesh, el_idx: list = [0], target_x_y: tuple = [(0, 0)]
) -> PyEITMesh:
    """!!!"""
    # reset posis:
    mesh.el_pos = np.arange(mesh.n_el)
    x_node = mesh.node[:, 0]
    y_node = mesh.node[:, 1]

    for el_n, tar in zip(el_idx, target_x_y):
        x_tar, y_tar = tar
        tar_idx = np.argsort(np.abs((x_node - x_tar) ** 2 + (y_node - y_tar) ** 2))[0]
        mesh.el_pos[el_n] = tar_idx
    return mesh


def set_perm_circle(mesh_obj: PyEITMesh, anomaly: PyEITAnomaly_Circle) -> PyEITMesh:
    pts = mesh_obj.element
    tri = mesh_obj.node
    perm = mesh_obj.perm
    tri_centers = np.mean(tri[pts], axis=1)
    index = (
        np.sqrt(
            (tri_centers[:, 1] - anomaly.center[1]) ** 2
            + (tri_centers[:, 0] - anomaly.center[0]) ** 2
        )
        < anomaly.r
    )

    mesh_obj.perm[index] = anomaly.perm  # Zuweisen von 10

    return mesh_obj


def create_rect_mesh(h0: float = 0.1, n_el: int = 16) -> PyEITMesh:
    """create head phantom mesh (symmetric)"""

    def _fd(pts):
        return shape.rectangle(pts, p1=[-1, -1], p2=[1, 1])

    p_fix_16 = [
        [-1, -1],
        [-1, -1 / 3],
        [-1, 1 / 3],
        [-1, 1],
        [-1 / 3, 1],
        [1 / 3, 1],
        [1, 1],
        [1, 1 / 3],
        [1, -1 / 3],
        [1, -1],
        [1 / 3, -1],
        [-1 / 3, -1],
        [-0.2, -0.2],
        [-0.2, 0.2],
        [0.2, 0.2],
        [0.2, -0.2],
    ]

    p, t = distmesh.build(
        _fd,
        shape.area_uniform,
        pfix=p_fix_16,
        h0=h0,
    )
    mesh = PyEITMesh(p, t)
    mesh.el_pos = np.arange(n_el)
    return mesh
