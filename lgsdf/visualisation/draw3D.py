# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import trimesh
import numpy as np
import torch
from PIL import Image
import io
import skimage.measure

from lgsdf import geometry

# 删除image的警告
import warnings
warnings.simplefilter('ignore', Image.DecompressionBombWarning) 


def draw_camera(
    camera, transform, color=(0., 1., 0., 0.8), marker_height=0.2
):
    marker = trimesh.creation.camera_marker(
        camera, marker_height=marker_height)
    marker[0].apply_transform(transform)
    marker[1].apply_transform(transform)
    marker[1].colors = (color, ) * len(marker[1].entities)

    return marker


def draw_cameras_from_eyes(eyes, ats, up, scene):
    for eye, at in zip(eyes, ats):
        R, t = geometry.transform.look_at(eye, at, up)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t

        transform = T @ geometry.transform.to_replica()
        camera = trimesh.scene.Camera(
            fov=scene.camera.fov, resolution=scene.camera.resolution)
        marker = draw_camera(camera, transform)
        scene.add_geometry(marker)


def draw_cams(batch_size, T_WC_batch_np, scene, color=None, latest_diff=True):
    no_color = color is None
    if no_color:
        color = (0.0, 1.0, 0.0, 0.8)
    for batch_i in range(batch_size):
        # if batch_i == (batch_size - 1):
        #     color = (1., 0., 0., 0.8)
        T_WC = T_WC_batch_np[batch_i]

        camera = trimesh.scene.Camera(
            fov=scene.camera.fov, resolution=scene.camera.resolution
        )
        marker_height = 0.3
        if batch_i == batch_size - 1 and latest_diff:
            if no_color:
                color = (1.0, 1.0, 1.0, 1.0)
                marker_height = 0.5

        marker = draw_camera(
            camera, T_WC, color=color, marker_height=marker_height
        )
        scene.add_geometry(marker[1])


def draw_segment(t1, t2, color=(1., 1., 0.)):
    line_segment = trimesh.load_path([t1, t2])
    line_segment.colors = (color, ) * len(line_segment.entities)

    return line_segment


def draw_trajectory(trajectory, scene, color=(1., 1., 0.)):
    for i in range(trajectory.shape[0] - 1):
        if (trajectory[i] != trajectory[i + 1]).any():
            segment = draw_segment(trajectory[i], trajectory[i + 1], color)
            scene.add_geometry(segment)


def draw_pc(batch_size,
            pcs_cam,
            T_WC_batch_np,
            im_batch=None,
            scene=None):

    pcs_w = []
    for batch_i in range(batch_size):
        T_WC = T_WC_batch_np[batch_i]
        pc_cam = pcs_cam[batch_i]

        col = None
        if im_batch is not None:
            img = im_batch[batch_i]
            col = img.reshape(-1, 3)

        pc_tri = trimesh.PointCloud(vertices=pc_cam, colors=col)
        pc_tri.apply_transform(T_WC)
        pcs_w.append(pc_tri.vertices)

        if scene is not None:
            scene.add_geometry(pc_tri)

    pcs_w = np.concatenate(pcs_w, axis=0)
    return pcs_w

def draw_add_pc(add_points,
            scene=None):
    pc_tri = trimesh.PointCloud(vertices=add_points, colors=[255, 0, 0])
    if scene is not None:
        scene.add_geometry(pc_tri)

def draw_add_pc_near(add_points_near,
            scene=None):
    pc_tri = trimesh.PointCloud(vertices=add_points_near, colors=[255, 255, 0])
    if scene is not None:
        scene.add_geometry(pc_tri)


def marching_cubes_trimesh(numpy_3d_sdf_tensor, grid_activate=None):
    """
    Convert sdf samples to triangular mesh.
    将sdf采样转换为三角形网格。
    """
    if grid_activate is not None:
        vertices, faces, vertex_normals, _ = skimage.measure.marching_cubes(numpy_3d_sdf_tensor, level=0.0,mask=grid_activate,)
    else:
        # 直接找到sdf的0势能面，得到顶点
        vertices, faces, vertex_normals, _ = skimage.measure.marching_cubes(numpy_3d_sdf_tensor, level=0.0,)
    # 256 256 256
    dim = numpy_3d_sdf_tensor.shape[0]
    # 顶点位置缩小256倍，不知道为啥
    vertices = vertices / (dim - 1)
    # 构建得到mesh，并返回
    mesh = trimesh.Trimesh(vertices=vertices,  vertex_normals=vertex_normals, faces=faces)
    return mesh


def draw_mesh(sdf, scale=None, transform=None, grid_activate=None):
    """
    Run marching cubes on sdf tensor to return mesh.
    在sdf张量上运行行进立方体以返回网格。
    """
    if isinstance(sdf, torch.Tensor):
        sdf = sdf.detach().cpu().numpy()
    if grid_activate is not None:
        if isinstance(grid_activate, torch.Tensor):
            grid_activate = grid_activate.detach().cpu().numpy()
        mesh = marching_cubes_trimesh(sdf, grid_activate)
    else:
        # 利用网格的sdf构建得到mesh
        mesh = marching_cubes_trimesh(sdf)
    # Transform to [-1, 1] range
    # 将mesh变换到[-1,1]的范围
    mesh.apply_translation([-0.5, -0.5, -0.5])
    mesh.apply_scale(2)

    # Transform to scene coordinates
    # 转换到场景坐标系
    if scale is not None:
        mesh.apply_scale(scale)
    if transform is not None:
        mesh.apply_transform(transform)
    # 设置颜色灰色
    mesh.visual.face_colors = [160, 160, 160, 255]
    # 返回设置好的mesh
    return mesh


def capture_scene_im(
    scene, pose, tm_pose=False, resolution=(1080, 720)
):
    if not tm_pose:
        pose = geometry.transform.to_trimesh(pose)
    scene.camera_transform = pose
    data = scene.save_image(resolution=resolution)
    image = np.array(Image.open(io.BytesIO(data)))

    return image