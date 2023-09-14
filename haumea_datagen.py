import numpy as np
import pyeit.eit.protocol as protocol
import pyeit.mesh as mesh
from pyeit.eit.fem import EITForward
from pyeit.mesh import PyEITMesh
from tqdm import tqdm
## from pyeit.mesh.shape import thorax
from pyeit.mesh.wrapper import PyEITAnomaly_Circle
from sciopy import plot_mesh

"""
Parameters:
"""
save_path = "data/"
h0 = 0.05
num_samples = 10_000
n_el = 16  # number of electrodes
r_bone = 0.2
bone_base_perm_val = 30
bone_perm_variation = 5  # +/- 5

cart_base_perm_val = 15
cart_perm_variation = 10  # +/- 10

min_r_cart = 0.2
max_r_cart = 0.5
r_cartilage = 0.4  # must be > 0.2

# probably a place to optimize!
dist_exc = 8
step_meas = 1
"""
---
"""


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
        tar_idx = np.argsort(
            np.abs((x_node - x_tar) ** 2 + (y_node - y_tar) ** 2)
        )[0]
        mesh.el_pos[el_n] = tar_idx
    return mesh


def set_perm_circle(
    mesh_obj: PyEITMesh, anomaly: PyEITAnomaly_Circle
) -> PyEITMesh:
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


mesh_empty = mesh.create(n_el, h0=h0)
mesh_obj = mesh.create(n_el, h0=h0)


# change electrode positions
el_posis = [(0.2, 0), (0, -0.2), (-0.2, 0.0), (0, 0.2)]
lst = [0, 4, 8, 12]

mesh_empty = move_el(mesh_empty, el_idx=lst, target_x_y=el_posis)
mesh_obj = move_el(mesh_obj, el_idx=lst, target_x_y=el_posis)
el_pos = mesh_empty.el_pos


# FOR LOOP
s_idx = 0
for bone_perm_var, cart_perm_var, r_car in tqdm(np.round(
    np.random.uniform(
        low=(
            bone_base_perm_val - bone_perm_variation,
            cart_base_perm_val - cart_perm_variation,
            min_r_cart,
        ),
        high=(
            bone_base_perm_val + bone_perm_variation,
            cart_base_perm_val + cart_perm_variation,
            max_r_cart,
        ),
        size=(num_samples, 3),
    ),
    3,
)):
    anomaly_cartilage = PyEITAnomaly_Circle(
        center=[0, 0], r=r_car, perm=cart_base_perm_val + cart_perm_var
    )
    anomaly_bone = PyEITAnomaly_Circle(
        center=[0, 0], r=r_bone, perm=bone_base_perm_val + bone_perm_var
    )

    mesh_empty = mesh.set_perm(
        mesh_empty, anomaly=anomaly_bone, background=1.0
    )
    mesh_obj = mesh.set_perm(
        mesh_obj, anomaly=anomaly_cartilage, background=1.0
    )
    mesh_obj = set_perm_circle(mesh_obj, anomaly=anomaly_bone)

    protocol_obj = protocol.create(
        n_el, dist_exc=dist_exc, step_meas=step_meas, parser_meas="std"
    )

    fwd_v = EITForward(mesh_empty, protocol_obj)
    v_empty = fwd_v.solve_eit(perm=mesh_empty.perm)
    v_obj = fwd_v.solve_eit(perm=mesh_obj.perm)

    np.savez(
        save_path + "sample_{:06d}.npz".format(s_idx),
        params=f"{bone_perm_var=}, {cart_perm_var=}, {r_car=}",
        mesh=mesh_obj,
        v_empty=v_empty,
        v_obj=v_obj,
    )
    s_idx += 1
