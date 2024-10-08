# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import matplotlib.pylab as plt
import torch

import torch.optim as optim
import torch.nn.functional  as F
from torchvision import transforms
import trimesh
import imgviz
import json
import cv2
import copy
import os
import kdtree
import time
import open3d as o3d
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt
import time

from lgsdf.datasets import (
    dataset, image_transforms, sdf_util, data_util
)
from lgsdf.datasets.data_util import FrameData
from lgsdf.modules import (
    fc_map, embedding, render, sample, loss
)
from lgsdf import geometry, visualisation
from lgsdf.eval import metrics, eval_pts
from lgsdf.visualisation import draw, draw3D
from lgsdf.eval.metrics import start_timing, end_timing


class Trainer():
    def __init__(
        self,
        device,
        config_file,
        chkpt_load_file=None,
        incremental=True,
        grid_dim=256,
    ):
        super(Trainer, self).__init__()

        # 先参数传递进来
        self.device = device
        self.incremental = incremental

        # 初始化一些变量
        self.tot_step_time = 0.
        self.last_is_keyframe = False
        self.steps_since_frame = 0
        self.optim_frames = 0

        self.gt_depth_vis = None
        self.gt_im_vis = None

        # 评估所需使用的参数
        self.gt_sdf_interp = None
        self.stage_sdf_interp = None
        self.sdf_dims = None
        self.sdf_transform = None

        # 
        self.grid_dim = grid_dim
        self.chunk_size = 200000

        with open(config_file) as json_file:
            self.config = json.load(json_file)

        self.frames = FrameData()  # keyframes

        # 根据config文件配置参数，包括数据集、真值、评估、模型、保存、采样
        self.set_params()
        # 确定相机参数，不同数据集用的不一样，并进行了降采样和分块
        self.set_cam()
        # 根据相机内参确定ray的方向
        self.set_directions()
        # 加载数据，初始化了数据集的类
        self.load_data()
        if self.gt_scene:
            # 如果提供了真值场景，设置场景的属性
            self.set_scene_properties()

        self.load_networks()
        if chkpt_load_file is not None:
            # 如果有保存好的模型，可以加载进来
            self.load_checkpoint(chkpt_load_file)
        # 设置模式为训练模式
        self.sdf_map.train()

        # for evaluation
        if self.sdf_transf_file is not None:
            # 如果存在真值数据
            self.load_gt_sdf()
        # 余弦相似度
        self.cosSim = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)

        # 可视化调试用
        self.grid_points = np.zeros((1,3))

    # Init functions ---------------------------------------

    def get_latest_frame_id(self):
        return int(self.tot_step_time * self.fps)

    def set_scene_properties(self):
        # 加载了.obj文件
        # self.scene_file = "/data/dyn/dataset/seqs/conference_room_1/scene/mesh.ply"
        scene_mesh = trimesh.exchange.load.load(self.scene_file, process=False)
        # bounds_transform是将网格边界框的中心移动到原点的变换矩阵。
        # bounds_extents是使用bounds_ transform变换后mesh的范围
        # mesh原点的相对变换与mesh边界范围
        T_extent_to_scene, bounds_extents =  trimesh.bounds.oriented_bounds(scene_mesh)
        self.inv_bounds_transform = torch.from_numpy(T_extent_to_scene).float().to(self.device)
        # bounds_transform是将网格边界框的中心移动到原点的变换矩阵，既求逆矩阵
        self.bounds_transform_np = np.linalg.inv(T_extent_to_scene)
        # 一个 4*4的矩阵，把mesh中心移动到原点用的
        self.bounds_transform = torch.from_numpy(self.bounds_transform_np).float().to(self.device)
        # 需要除以range_dist，因为它将缩放在range=[-1，1]中创建的网格，也需要除以0.9，因此范围比gt mesh稍大
        grid_range = [-1.0, 1.0]
        # 2.0
        range_dist = grid_range[1] - grid_range[0]
        # 场景的范围比真值的稍大，方面后面切片可视化和评估
        self.scene_scale_np = bounds_extents / (range_dist * 0.9)
        self.scene_scale = torch.from_numpy(self.scene_scale_np).float().to(self.device)
        # 一个三个数的数组，之后的场景成以这个数字，可以把坐标缩放到[0.9, 0.9]
        self.inv_scene_scale = 1. / self.scene_scale
        # 场景边界的一半
        self.scene_center = scene_mesh.bounds.mean(axis=0)
        # 绘制一个三维grid，grid_dim为256，有变换和边界可以得到场景每个grid的真值坐标
        self.grid_pc = geometry.transform.make_3D_grid(
            grid_range,
            self.grid_dim,
            self.device,
            transform=self.bounds_transform,
            scale=self.scene_scale,
        )
        print("The scale of scene is: ", self.scene_scale_np*2)
        # 得到每个grid的坐标
        self.grid_pc = self.grid_pc.view(-1, 3).to(self.device)
        # 计算每列的最小值和最大值
        min_values, _ = torch.min(self.grid_pc, dim=0)
        max_values, _ = torch.max(self.grid_pc, dim=0)
        # 打印结果
        print("最小值：", min_values)
        print("最大值：", max_values)
        # print("From grid ", self.grid_pc[0].cpu().numpy(), " to grid ", self.grid_pc[-1].cpu().numpy(),)
        # print("The transform of scene is: \n", self.bounds_transform.cpu().numpy())
        # dyndyn：这个grid_pc就是每个栅格的中心点，下面可以得到每个grid的上下界限
        self.grid_scale =  (self.scene_scale / (self.grid_dim-1)) * 2
        # print(self.bounds_transform)
        print("The scale of each grid is: ", self.grid_scale.cpu().numpy())
        self.idx_mul = torch.Tensor([self.grid_dim**2, self.grid_dim, 1],).to(self.device)
        # 设置该grid是否被激活
        self.grid_activate = torch.zeros(self.grid_pc.shape[0]).to(self.device)
        # self.grid_activate = grid_activate == 1
        # 存储被激活grid的sdf值，以及权重，以及loss，还有grad
        self.grid_sdf = torch.zeros(self.grid_pc.shape[0], 6).to(self.device)
        
        self.all_grid_indices = torch.arange(len(self.grid_pc), device=self.device)

        # self.up是为了俯视图t
        # 0
        self.up_ix = np.argmax(np.abs(np.matmul(self.up, self.bounds_transform_np[:3, :3])))
        # bounds_transform_np第一行前三个
        self.grid_up = self.bounds_transform_np[:3, self.up_ix]
        # 判断是否向上对齐了
        self.up_aligned = np.dot(self.grid_up, self.up) > 0

        self.all_gt_pc = None
        self.rand_id_ok = None

    def set_params(self):
        # 本次深度图像序列的目录，如"/data/dyn/dataset/seqs/apt_3_obj/"
        self.seq_dir = self.config["dataset"]["seq_dir"]
        # 本次序列的名称，如apt_3_obj
        self.seq = [x for x in self.seq_dir.split('/') if x != ''][-1]
        # 图像所在目录，如"/data/dyn/dataset/seqs/apt_3_obj/results"
        self.ims_file = os.path.join(self.seq_dir, "results")
        # 数据集，如replicaCAD
        self.dataset_format = self.config["dataset"]["format"]
        # seq的跟目录，可以看到所有数据的位置，如"/data/dyn/dataset/seqs"
        seq_root = "/".join(self.seq_dir.split('/')[:-2])
        # 找到对应的info文件，里面记录了相机内参，如"/data/dyn/dataset/seqs/replicaCAD_info.json"
        info_file = os.path.join(seq_root, f"{self.dataset_format}_info.json")
        if "gt_mesh_dir" in self.config["dataset"]:
            info_file = os.path.join(seq_root, f"Myself_info.json")
        with open(info_file, 'r') as f:
            # 相机内参文件
            self.seq_info = json.load(f)
        # 深度像素对应于距离的换算关系，应该是除以这个数字吧，1/depth_scale
        self.inv_depth_scale = 1. / self.seq_info["depth_scale"]
        # 相机帧率
        self.fps = self.seq_info["fps"]

        # 真值所用的，gt
        self.obj_bounds_file = None
        if os.path.exists(self.seq_dir + "/obj_bounds.txt"):
            # 如果有确定边界的文件，如"/data/dyn/dataset/seqs/apt_3_obj/obj_bounds.txt"
            self.obj_bounds_file = self.seq_dir + "/obj_bounds.txt"
        self.gt_scene = False
        self.sdf_transf_file = None
        self.myself = False
        if "gt_sdf_dir" in self.config["dataset"]:
            # 确定真值场景
            # 真值场景所在目录，如"/data/dyn/dataset/gt_sdfs/apt_3/"
            gt_sdf_dir = self.config["dataset"]["gt_sdf_dir"]
            # 真值场景的obj文件
            self.scene_file = gt_sdf_dir + "mesh.obj"
            # self.scene_file = gt_sdf_dir + "mesh.ply"
            # 真值场景的以1cm为分辨率的sdf数据，用于评估
            self.gt_sdf_file = gt_sdf_dir + "/1cm/sdf.npy"
            # 真值场景的以1cm为分辨率的分段sdf数据，用于评估
            self.stage_sdf_file = gt_sdf_dir + "/1cm/stage_sdf.npy"
            # 到真值场景需要进行的变换，需要平移
            self.sdf_transf_file = gt_sdf_dir + "/1cm/transform.txt"
            self.gt_scene = True
        if "gt_mesh_dir" in self.config["dataset"]:
            gt_sdf_dir = self.config["dataset"]["gt_mesh_dir"]
            self.scene_file = gt_sdf_dir + "mesh.ply"
            self.gt_scene = True
            self.myself = True

        self.scannet_dir = None
        if "scannet_dir" in self.config["dataset"]:
            # 如果有scannet，确定它的目录，目前没有
            self.scannet_dir = self.config["dataset"]["scannet_dir"]
        if "" in self.config["dataset"]:
            # 图像索引，如[0, 200, 400, 500, 570, 650]
            self.indices = self.config["dataset"]["im_indices"]
        self.noisy_depth = False
        if "noisy_depth" in self.config["dataset"]:
            # 深度图像的噪声，1
            self.noisy_depth = bool(self.config["dataset"]["noisy_depth"])
        # 相机轨迹，用的真值
        self.traj_file = self.seq_dir + "/traj.txt"
        assert os.path.exists(self.traj_file)
        self.gt_traj = None
        # 训练，20000
        self.n_steps = self.config["trainer"]["steps"]

        # 模型的有关参数 Model
        # 是否使用激活，否
        self.do_active = bool(self.config["model"]["do_active"])
        # 对输出进行缩放，0.14
        self.scale_output = self.config["model"]["scale_output"]
        # 噪声标准差 0.25
        self.noise_std = self.config["model"]["noise_std"]
        # 噪声关键帧 0.08
        self.noise_kf = self.config["model"]["noise_kf"]
        # 噪声帧 0.04
        self.noise_frame = self.config["model"]["noise_frame"]
        # 窗口大小数，5
        self.window_size = self.config["model"]["window_size"]
        # 隐含层块个数，2
        self.hidden_layers_block = self.config["model"]["hidden_layers_block"]
        # 隐含层特征尺寸，256
        self.hidden_feature_size = self.config["model"]["hidden_feature_size"]
        # 时间感知，实时的？ 1
        self.frac_time_perception = self.config["model"]["frac_time_perception"]
        # 每个关键帧迭代次数，60
        self.iters_per_kf = self.config["model"]["iters_per_kf"]
        # 每帧迭代次数，10
        self.iters_per_frame = self.config["model"]["iters_per_frame"]
        # 单个像素深度loss阈值 0.1
        self.kf_dist_th = self.config["model"]["kf_dist_th"]
        # 判断关键帧用的，如果损失低于阈值的比例低于这些 0.65
        self.kf_pixel_ratio = self.config["model"]["kf_pixel_ratio"]

        embed_config = self.config["model"]["embedding"]
        # 坐标编码正二十面体输入尺寸，0.05937489
        self.scale_input = embed_config["scale_input"]
        # 坐标编码的频率个数，5
        self.n_embed_funcs = embed_config["n_embed_funcs"]
        # 是否使用高斯嵌入 0
        self.gauss_embed = bool(embed_config["gauss_embed"])
        # 高斯嵌入的标准差 11
        self.gauss_embed_std = embed_config["gauss_embed_std"]
        # 是否优化嵌入，0
        self.optim_embedding = bool(embed_config["optim_embedding"])

        # 评估所用的，Evaluation
        # 是否进行voxblox的比较，0
        self.do_vox_comparison = (
            bool(self.config["eval"]["do_vox_comparison"])
            and "eval_pts_root" in self.config["eval"])
        # 是否进行比较，0
        self.do_eval = self.config["eval"]["do_eval"]
        # 比较的频率，1
        self.eval_freq_s = self.config["eval"]["eval_freq_s"]
        # 是否比较sdf，1
        self.sdf_eval = bool(self.config["eval"]["sdf_eval"])
        # 是否比较mesh，0
        self.mesh_eval = bool(self.config["eval"]["mesh_eval"])
        # 在哪些时间评估评估耗时
        self.eval_times = []
        if self.do_vox_comparison and "eval_pts_root" in self.config["eval"]:
            # 评估所需要的目录，如"/data/dyn/dataset/eval_pts/"
            self.eval_pts_root = self.config["eval"]["eval_pts_root"]
            # 评估目录，如"/data/dyn/dataset/eval_pts/vox/"
            self.eval_pts_dir = self.config["eval"]["eval_pts_root"]
            self.eval_pts_dir += "vox/"
            # 确定和哪种分辨率的vox比较，如果需要实时全部用来处理的话，分辨率就是0.055，如果再快的话，只用一般时间处理就是0.078
            if self.frac_time_perception == 1.:
                self.eval_pts_dir += "0.055/"
            elif self.frac_time_perception == 0.75:
                self.eval_pts_dir += "0.063/"
            elif self.frac_time_perception == 0.5:
                self.eval_pts_dir += "0.078/"
            elif self.frac_time_perception == 0.25:
                self.eval_pts_dir += "0.11/"
            else:
                raise ValueError('Frace perception time not in [0.25, 0.5, 0.75, 1.]')
            # 评估的目录，"/data/dyn/dataset/eval_pts/vox/0.055/apt_3_obj/eval_pts/"
            self.eval_pts_dir += [x for x in self.seq_dir.split('/') if x != ""][-1] + "/eval_pts/"
            # 评估时间选择和voxblox的相同，其中的所有目录，如"/data/dyn/dataset/eval_pts/vox/0.055/apt_3_obj/eval_pts/0.553"
            self.eval_times = [float(x) for x in os.listdir(self.eval_pts_dir)]
            self.eval_times.sort()
            print("eval pts dir", self.eval_pts_dir)
        # if self.do_eval:
            # 使用真值轨迹进行评估，只要评估就需要用到缓存数据
        self.cached_dataset = eval_pts.get_cache_dataset(self.seq_dir, self.dataset_format, self.scannet_dir)

        # 保存所用的参数，save
        # 保存的周期 10
        self.save_period = self.config["save"]["save_period"]
        # 所需要保存的时间段，10:10:2000
        self.save_times = np.arange(
            self.save_period, 2000, self.save_period).tolist()
        # 是否保存checkpoints
        self.save_checkpoints = bool(self.config["save"]["save_checkpoints"])
        # 是否保存切片
        self.save_slices = bool(self.config["save"]["save_slices"])
        # 是否保存mesh
        self.save_meshes = bool(self.config["save"]["save_meshes"])

        # 损失函数有关参数，Loss
        # 所使用的边界法，既论文中提到的三种方法，"ray"是最简单的
        self.bounds_method = self.config["loss"]["bounds_method"]
        assert self.bounds_method in ["ray", "normal", "pc", "all_pc"]
        # 损失函数方式，L1
        self.loss_type = self.config["loss"]["loss_type"]
        assert self.loss_type in ["L1", "L2","Huber"]
        # sdf的权重，5.38344020，试出来的把
        self.trunc_weight = self.config["loss"]["trunc_weight"]
        # 0.29365022
        self.trunc_distance = self.config["loss"]["trunc_distance"]
        # Eikonal regularisation权重 0.268，试出来的吧
        self.eik_weight = self.config["loss"]["eik_weight"]
        # 0.1
        self.eik_apply_dist = self.config["loss"]["eik_apply_dist"]
        # 法向量的权重，0.018，试出来的吧
        self.grad_weight = self.config["loss"]["grad_weight"]
        # 是否应用这个损失
        self.orien_loss = bool(self.config["loss"]["orien_loss"])

        self.do_normal = True
        # if self.bounds_method == "n的ormal" or self.grad_weight != 0:
        #     # 如果确定sdf的方法是法线话，或者grad有权重的话，需要计算边界法线
        #     self.do_normal = True

        # 根据损失优化的参数 optimiser
        # 学习率 0.0013
        self.learning_rate = self.config["optimiser"]["lr"]
        # 权重衰减 防止过拟合 0.012
        self.weight_decay = self.config["optimiser"]["weight_decay"]

        # 采样 Sampling
        # 最大的深度范围 12.0
        self.max_depth = self.config["sample"]["depth_range"][1]
        # 最小的深度范围 0.07， 采样的起始
        self.min_depth = self.config["sample"]["depth_range"][0]
        # 表面背后的探讨距离 0.1
        self.dist_behind_surf = self.config["sample"]["dist_behind_surf"]
        # 每帧图像采样个数 200
        self.n_rays = self.config["sample"]["n_rays"]
        # 关键帧图像采样个数 400
        self.n_rays_is_kf = self.config["sample"]["n_rays_is_kf"]
        # 均匀采样个数 19，论文里面可是20诶
        self.n_strat_samples = self.config["sample"]["n_strat_samples"]
        # 表面高斯采样个数 8
        self.n_surf_samples = self.config["sample"]["n_surf_samples"]
        # replay的grid数目
        self.n_replay_grids = self.config["sample"]["n_replay_grids"]
        

    def set_cam(self):
        if self.dataset_format == "ScanNet":
            # 如果是scannet数据集，打开给定的info文件
            info = {}
            info_file = self.scannet_dir + self.seq + '.txt'
            with open(info_file, 'r') as f:
                for line in f.read().splitlines():
                    split = line.split(' = ')
                    info[split[0]] = split[1]
            self.fx = float(info['fx_depth'])
            self.fy = float(info['fy_depth'])
            self.cx = float(info['mx_depth'])
            self.cy = float(info['my_depth'])
            self.H = int(info['depthHeight'])
            self.W = int(info['depthWidth'])
            self.W = int(info['depthWidth'])
        else:
            # 如果是replicaCAD数据集，用相机参数
            self.fx = self.seq_info["camera"]["fx"]
            self.fy = self.seq_info["camera"]["fy"]
            self.cx = self.seq_info["camera"]["cx"]
            self.cy = self.seq_info["camera"]["cy"]
            self.H = self.seq_info["camera"]["h"]
            self.W = self.seq_info["camera"]["w"]
        
        self.k = torch.tensor([[self.fx, 0.0, self.cx],
                  [0.0, self.fy, self.cy],
                  [0.0, 0.0, 1.0]], device='cuda')
        reduce_factor = 10
        # 把图像缩小，不然计算量太大了，10倍缩小，参数也要缩小
        self.H_vis = self.H // reduce_factor
        self.W_vis = self.W // reduce_factor
        self.fx_vis = self.fx / reduce_factor
        self.fy_vis = self.fy / reduce_factor
        self.cx_vis = self.cx / reduce_factor
        self.cy_vis = self.cy / reduce_factor

        # 每帧图像采样是要分块的，分了8 x 8块
        self.loss_approx_factor = 8
        # 每个block的长和款
        self.w_block = self.W // self.loss_approx_factor
        self.h_block = self.H // self.loss_approx_factor
        # 这些block去和起来的长和宽，可以确定每个block的位置
        # [   0,  150,  300,  450,  600,  750,  900, 1050]
        # [  0,  85, 170, 255, 340, 425, 510, 595]
        self.increments_w = torch.arange(
            self.loss_approx_factor, device=self.device) * self.w_block
        self.increments_h = torch.arange(
            self.loss_approx_factor, device=self.device) * self.h_block
        c, r = torch.meshgrid(self.increments_w, self.increments_h)
        self.c, self.r = c.t(), r.t()
        # 每个block的起始位置
        self.increments_single = torch.stack((r, c), dim=2).view(-1, 2)


    def set_directions(self):
        # 尺寸为[1, 680, 1200, 3]，确定每个像素的ray
        self.dirs_C = geometry.transform.ray_dirs_C(
            1,
            self.H,
            self.W,
            self.fx,
            self.fy,
            self.cx,
            self.cy,
            self.device,
            depth_type="z",
        )

        # 尺寸为[1, 8160, 3]，确定每个像素的ray 68 x 120 = 8160
        self.dirs_C_vis = geometry.transform.ray_dirs_C(
            1,
            self.H_vis,
            self.W_vis,
            self.fx_vis,
            self.fy_vis,
            self.cx_vis,
            self.cy_vis,
            self.device,
            depth_type="z",
        ).view(1, -1, 3)

    def load_networks(self):
        # 位置编码，把可以把size 3的位置编码为size 255的embedding
        positional_encoding = embedding.PostionalEncoding(
            min_deg=0,
            max_deg=self.n_embed_funcs,
            scale=self.scale_input,
            transform=self.inv_bounds_transform,
        )

        # 网络模型，和论文描述基本一致
        self.sdf_map = fc_map.SDFMap(
            positional_encoding,
            hidden_size=self.hidden_feature_size,
            hidden_layers_block=self.hidden_layers_block,
            scale_output=self.scale_output,
        ).to(self.device)

        # 网络参数优化器，输入现有参数，学习率，权重衰减 
        self.optimiser = optim.AdamW(
            self.sdf_map.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

    def load_checkpoint(self, checkpoint_load_file):
        checkpoint = torch.load(checkpoint_load_file)
        self.sdf_map.load_state_dict(checkpoint["model_state_dict"])
        # self.optimiser.load_state_dict(checkpoint["optimizer_state_dict"])

    def load_gt_sdf(self):
        # shape 727, 314, 1300
        sdf_grid = np.load(self.gt_sdf_file)
        if self.dataset_format == "ScanNet":
            sdf_grid = np.abs(sdf_grid)
        # 加载变换
        self.sdf_transform = np.loadtxt(self.sdf_transf_file)
        # 得到了一个差值函数，输入坐标，可以得到近似值
        self.gt_sdf_interp = sdf_util.sdf_interpolator(sdf_grid, self.sdf_transform)
        # 加载位置的尺寸
        self.sdf_dims = torch.tensor(sdf_grid.shape)


    # Visualisation methods -----------------------------------

    def update_vis_vars(self):
        # 关键帧中的深度和图像
        depth_batch_np = self.frames.depth_batch_np
        im_batch_np = self.frames.im_batch_np
        if self.gt_depth_vis is None:
            # 如果还没有可视化的，第一次
            updates = depth_batch_np.shape[0]
        else:
            # 否则看看有哪些新的需要展示的
            diff_size = depth_batch_np.shape[0] - self.gt_depth_vis.shape[0]
            updates = diff_size + 1
        for i in range(updates, 0, -1):
            # 深度矩阵的真值
            prev_depth_gt = depth_batch_np[-i]
            # rgb图像的真值
            prev_im_gt = im_batch_np[-i]
            # 把图像resize一下，改成可视化大小的，缩小十倍
            prev_depth_gt_resize = imgviz.resize(
                prev_depth_gt, width=self.W_vis,
                height=self.H_vis,
                interpolation="nearest")[None, ...]
            prev_im_gt_resize = imgviz.resize(
                prev_im_gt, width=self.W_vis,
                height=self.H_vis)[None, ...]
            replace = False
            if i == updates:
                # 如果是第一个，用replace，因为需要替换之前的最后一个（最新帧但不一定是关键帧）
                replace = True
            # 加入到可视化的数据中
            self.gt_depth_vis = data_util.expand_data(
                self.gt_depth_vis,
                prev_depth_gt_resize,
                replace=replace)
            self.gt_im_vis = data_util.expand_data(
                self.gt_im_vis,
                prev_im_gt_resize,
                replace=replace)

    def frames_vis(self):
        # 把所有当前的关键帧位置根据预测渲染为深度图
        view_depths = self.render_depth_vis()
        # 真正的深度图
        gt_depth_ims = self.gt_depth_vis
        # 真正的rgb图
        im_batch_np = self.gt_im_vis
        views = []
        for batch_i in range(len(self.frames)):
            # 渲染深度的展示
            depth = view_depths[batch_i]
            depth_viz = imgviz.depth2rgb(depth)
            # 真值深度的展示
            gt = gt_depth_ims[batch_i]
            gt_depth = imgviz.depth2rgb(gt)
            # 渲染和真值的差异
            loss = np.abs(gt - depth)
            # 如果gt都是0的话，就不需要计算loss了
            loss[gt == 0] = 0
            loss_viz = imgviz.depth2rgb(loss)
            # 可视化的三个图像
            visualisations = [gt_depth, depth_viz, loss_viz]
            # if im_batch_np is not None:
            #     # 再加上rgb的图像
            #     visualisations.append(im_batch_np[batch_i])
            # viz = np.d
            # 按照垂直方向排序，组成新的数组，所以效果里面是竖着的
            viz_1 = np.vstack((gt_depth, depth_viz))
            viz_2 = np.vstack((loss_viz, im_batch_np[batch_i]))
            viz = np.hstack((viz_1, viz_2))
            views.append(viz)
        # 不同帧水平排列
        viz = np.hstack(views)
        return viz

    def render_depth_vis(self):
        # 渲染深度图像
        view_depths = []
        # 这是输入的真值深度
        depth_gt = self.frames.depth_batch_np
        # 输入的真值轨迹
        T_WC_batch = self.frames.T_WC_batch
        if self.frames.T_WC_track:
            T_WC_batch = self.frames.T_WC_track
        # 设置不要反向传播
        with torch.set_grad_enabled(False):
            for batch_i in range(len(self.frames)):  # loops through frames
                T_WC = T_WC_batch[batch_i].unsqueeze(0)
                # 真值深度图变换一下
                depth_sample = depth_gt[batch_i]
                depth_sample = cv2.resize(depth_sample, (self.W_vis, self.H_vis))
                depth_sample = torch.FloatTensor(depth_sample).to(self.device)
                # 渲染的最大深度，每一帧不超过表面之后的0.8m
                max_depth = (depth_sample + 0.8).flatten()
                # 只进行均匀采样，采样时用的还是缩小之后的，以加快速度吧，但是要对每个像素都采样了
                pc, z_vals, _, _ = sample.sample_along_rays(
                    T_WC,
                    self.min_depth,
                    max_depth,
                    self.n_strat_samples,
                    n_surf_samples=0,
                    dirs_C=self.dirs_C_vis[0],
                    gt_depth=None,
                    grad=False,
                )
                # 设置不要反向传播，计算sdf值
                with torch.set_grad_enabled(False):
                    sdf = self.sdf_map(pc)
                # 把所有像素进行利用采样点的预测结果渲染
                view_depth = render.sdf_render_depth(z_vals, sdf)
                # 改成图像大小
                view_depth = view_depth.view(self.H_vis, self.W_vis)
                # view_depths存储所有需要展示的
                view_depths.append(view_depth)
            view_depths = torch.stack(view_depths)
            view_depths = view_depths.cpu().numpy()
        # 返回np格式，方便cv展示
        return view_depths

    def draw_3D(
        self,
        show_pc=False,
        show_grid_pc=False,
        show_mesh=False,
        draw_cameras=False,
        show_gt_mesh=False,
        camera_view=True,
    ):
        # 绘制用于展示的mesh
        # 一个专门展示场景mesh的类
        scene = trimesh.Scene()
        # 还可以设置相机视角观看结果
        scene.set_camera()
        scene.camera.focal = (self.fx, self.fy)
        scene.camera.resolution = (self.W, self.H)
        # 关键帧的位姿，用来绘制
        T_WC_np = self.frames.T_WC_batch_np
        # 如果
        if self.frames.T_WC_track:
            T_WC_np = self.frames.T_WC_track.cpu().numpy()
        # 如果需要绘制相机的位姿
        if draw_cameras:
            n_frames = len(self.frames)
            # 绘制相机，在scene中绘制，颜色为绿色
            draw3D.draw_cams(n_frames, T_WC_np, scene, color=(0.0, 1.0, 0.0, 1.0))
            # 用的不是真值轨迹，其实是的
            if self.frames.T_WC_gt:  
                draw3D.draw_cams(n_frames, self.frames.T_WC_gt, scene, color=(1.0, 0.0, 1.0, 0.8))
                draw3D.draw_cams(n_frames, self.frames.T_WC_batch_np, scene, color=(1., 0., 0., 0.8))
            # 增量式需要绘制相机轨迹
            if self.incremental:
                trajectory_gt = self.frames.T_WC_batch_np[:, :3, 3]
                if self.frames.T_WC_gt is not None:
                    trajectory_gt = self.frames.T_WC_gt[:, :3, 3]
                # 绘制红色轨迹，就是把这些位置连接起来
                visualisation.draw3D.draw_trajectory(trajectory_gt, scene, color=(1.0, 0.0, 0.0))
        if show_pc:
            # 如果要展示点云的话，用缩小后的图像的投影点来展示
            pcs_cam = geometry.transform.backproject_pointclouds(self.gt_depth_vis, self.fx_vis, self.fy_vis, self.cx_vis, self.cy_vis)
            draw3D.draw_pc(n_frames, pcs_cam, T_WC_np, self.gt_im_vis, scene)
        if show_grid_pc:
            draw3D.draw_add_pc(self.grid_points, scene)
        if show_mesh:
            # 如果要展示mesh的话，找到zero level set
            sdf_mesh = self.mesh_rec()
            # scene中添加mesh
            scene.add_geometry(sdf_mesh)
        if show_gt_mesh:
            # 如果要展示真值mesh
            gt_mesh = trimesh.load(self.scene_file)
            # 透明度50%
            # gt_mesh.visual.material.image.putalpha(50)
            scene.add_geometry(gt_mesh)
        if not camera_view:
            # 如果不需要相机视角，就需要确定从上往下看
            cam_pos = self.scene_center + self.up * 12 + np.array([3., 0., 0.])
            R, t = geometry.transform.look_at(cam_pos, self.scene_center, -self.up)
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = t
            # 观看场景的姿态
            scene.camera_transform = geometry.transform.to_trimesh(T)
        else:
            view_idx = -1
            # 如果用相机视角，使用最近的相机的位姿
            scene.camera_transform = geometry.transform.to_trimesh(T_WC_np[view_idx])
            scene.camera_transform = (scene.camera_transform @ trimesh.transformations.translation_matrix([0, 0, 0.1]))
        return scene

    def draw_local_3D(
        self,
        show_pc=False,
        show_mesh=False,
        draw_cameras=False,
        show_gt_mesh=False,
        camera_view=True,
    ):
        # 绘制用于展示的mesh
        # 一个专门展示场景mesh的类
        scene = trimesh.Scene()
        # 还可以设置相机视角观看结果
        scene.set_camera()
        scene.camera.focal = (self.fx, self.fy)
        scene.camera.resolution = (self.W, self.H)
        # 关键帧的位姿，用来绘制
        T_WC_np = self.frames.T_WC_batch_np
        # 如果
        if self.frames.T_WC_track:
            T_WC_np = self.frames.T_WC_track.cpu().numpy()
        # 如果需要绘制相机的位姿
        if draw_cameras:
            n_frames = len(self.frames)
            # 绘制相机，在scene中绘制，颜色为绿色
            draw3D.draw_cams(n_frames, T_WC_np, scene, color=(0.0, 1.0, 0.0, 1.0))
            # 用的不是真值轨迹，其实是的
            if self.frames.T_WC_gt:  
                draw3D.draw_cams(n_frames, self.frames.T_WC_gt, scene, color=(1.0, 0.0, 1.0, 0.8))
                draw3D.draw_cams(n_frames, self.frames.T_WC_batch_np, scene, color=(1., 0., 0., 0.8))
            # 增量式需要绘制相机轨迹
            if self.incremental:
                trajectory_gt = self.frames.T_WC_batch_np[:, :3, 3]
                if self.frames.T_WC_gt is not None:
                    trajectory_gt = self.frames.T_WC_gt[:, :3, 3]
                # 绘制红色轨迹，就是把这些位置连接起来
                visualisation.draw3D.draw_trajectory(trajectory_gt, scene, color=(1.0, 0.0, 0.0))
        if show_pc:
            # 如果要展示点云的话，用缩小后的图像的投影点来展示
            pcs_cam = geometry.transform.backproject_pointclouds(self.gt_depth_vis, self.fx_vis, self.fy_vis, self.cx_vis, self.cy_vis)
            draw3D.draw_pc(n_frames, pcs_cam, T_WC_np, self.gt_im_vis, scene)
        if show_mesh:
            # 如果要展示mesh的话，找到zero level set
            sdf_mesh = self.mesh_local_rec()
            # scene中添加mesh
            scene.add_geometry(sdf_mesh)
        if show_gt_mesh:
            # 如果要展示真值mesh
            gt_mesh = trimesh.load(self.scene_file)
            # 透明度50%
            gt_mesh.visual.material.image.putalpha(50)
            scene.add_geometry(gt_mesh)
        if not camera_view:
            # 如果不需要相机视角，就需要确定从上往下看
            cam_pos = self.scene_center + self.up * 12 + np.array([3., 0., 0.])
            R, t = geometry.transform.look_at(cam_pos, self.scene_center, -self.up)
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = t
            # 观看场景的姿态
            scene.camera_transform = geometry.transform.to_trimesh(T)
        else:
            view_idx = -1
            # 如果用相机视角，使用最近的相机的位姿
            scene.camera_transform = geometry.transform.to_trimesh(T_WC_np[view_idx])
            scene.camera_transform = (scene.camera_transform @ trimesh.transformations.translation_matrix([0, 0, 0.1]))
        return scene

    def draw_obj_3D(self, show_gt_mesh=True):
        # 没用上，是具体绘制某个物体的
        if self.obj_bounds_file is not None:
            scene = trimesh.Scene()
            scene.set_camera()
            scene.camera.focal = (self.fx, self.fy)
            scene.camera.resolution = (self.W, self.H)
            if show_gt_mesh:
                gt_mesh = trimesh.load(self.scene_file)
                gt_mesh.visual.material.image.putalpha(50)
            obj_bounds = metrics.get_obj_eval_bounds(
                self.obj_bounds_file, self.up_ix,
                expand_m=0.2, expand_down=True)
            for i, bounds in enumerate(obj_bounds):
                x = torch.linspace(bounds[0, 0], bounds[1, 0], 128)
                y = torch.linspace(bounds[0, 1], bounds[1, 1], 128)
                z = torch.linspace(bounds[0, 2], bounds[1, 2], 128)
                xx, yy, zz = torch.meshgrid(x, y, z)
                pc = torch.cat(
                    (xx[..., None], yy[..., None], zz[..., None]), dim=3)
                pc = pc.view(-1, 3).to(self.device)
                with torch.set_grad_enabled(False):
                    sdf = fc_map.chunks(
                        pc, self.chunk_size, self.sdf_map,
                        # surf_dists=gt_dist,
                    )
                T = np.eye(4)
                T[:3, 3] = bounds[0] + 0.5 * (bounds[1] - bounds[0])
                sdf = sdf.view(128, 128, 128)
                obj_mesh = draw3D.draw_mesh(
                    sdf, 0.5 * (bounds[1] - bounds[0]), T)
                obj_mesh.visual.face_colors = [160, 160, 160, 160]
                scene.add_geometry(obj_mesh)
                if show_gt_mesh:
                    box = trimesh.primitives.Box(
                        extents=bounds[1] - bounds[0], transform=T)
                    crop = gt_mesh.slice_plane(
                        box.facets_origin, -box.facets_normal)
                    crop.visual.face_colors = [0, 160, 50, 160]
                    scene.add_geometry(crop)
            scene.set_camera()
            return scene
        return None

    def get_sdf_grid(self):
        # 得到256 x 256 x 256的grid尺寸下的所有sdf值，可用于切片和scene可视化
        with torch.set_grad_enabled(False):
            sdf = fc_map.chunks(self.grid_pc, self.chunk_size, self.sdf_map,)
            dim = self.grid_dim
            sdf = sdf.view(dim, dim, dim)
        return sdf


    def mesh_rec(self):
        """
        Generate mesh reconstruction.
        生成网格形状的重构，即零面
        """
        # 首先得到256 x 256 x 256个sdf值的栅格
        sdf = self.get_sdf_grid()
        # 全部显示
        sdf_mesh = draw3D.draw_mesh(sdf,  self.scene_scale_np, self.bounds_transform_np,)
        # 方法一：按照法线着色
        # face_vertices = sdf_mesh.vertices[sdf_mesh.faces]
        # face_normals = np.cross((face_vertices[:,0]-face_vertices[:,1]),(face_vertices[:,1]-face_vertices[:,2]))
        # face_normals_dir = np.linalg.norm(face_normals, axis=1)
        # face_normals = face_normals/face_normals_dir[:,None]
        # face_normals = (face_normals+1)/2
        # for i in range(sdf_mesh.faces.shape[0]):
        #     sdf_mesh.visual.face_colors[i] = 255*np.array([face_normals[i,0],face_normals[i,1],face_normals[i,2],1])
            
        if "gt_mesh_dir" not in self.config["dataset"]:
            # 方法二：按照面中心高度着色
            face_vertices = sdf_mesh.vertices[sdf_mesh.faces].mean(axis=1)
            # 对于replica数据集
            face_z = face_vertices[:,1]
            # 对于scannet数据集
            if self.dataset_format == "ScanNet":
                face_z = face_vertices[:,2]
            cmap = sdf_util.get_cost_colormap(range=[face_z.min(),face_z.max()])
            face_rgba = cmap.to_rgba(face_z.flatten(), alpha=1., bytes=False)
            for i in range(sdf_mesh.faces.shape[0]):
                sdf_mesh.visual.face_colors[i] = 255*face_rgba[i]
        return sdf_mesh


    def mesh_local_rec(self):
        """
        Generate mesh reconstruction.
        生成网格形状的重构，即零面
        """
        # grid_sdf = self.grid_sdf[:, 0]
        # grid_sdf[~(torch.nonzero(self.grid_activate)[:,0])] = None
        # grid_sdf = grid_sdf.reshape(self.grid_dim, self.grid_dim, self.grid_dim)
        sdf = self.get_sdf_grid()
        sdf_local = self.grid_sdf[:, 0].reshape(self.grid_dim, self.grid_dim, self.grid_dim)
        grid_activate = self.grid_activate.reshape(self.grid_dim,self.grid_dim,self.grid_dim)
        grid_activate = grid_activate == 1
        sdf[grid_activate] = sdf_local[grid_activate]

        # N = sdf_local.shape[0]
        # coords = torch.stack(torch.meshgrid(torch.arange(N), torch.arange(N), torch.arange(N)), dim=-1)
        # distance_map, nearest_indices = distance_transform_edt(~grid_activate.cpu().numpy(), return_indices=True)
        # print(nearest_indices)
        # nearest_indices = torch.tensor(nearest_indices, dtype=torch.long, device="cuda")
        # nearest_values = sdf_local[tuple(nearest_indices)][~grid_activate]
        # filled_tensor = sdf_local.clone()
        # # filled_tensor[~grid_activate] = sdf_local[tuple(nearest_indices)]
        # filled_tensor[~grid_activate] = nearest_values
        # sdf_mesh = draw3D.draw_mesh(filled_tensor,  self.scene_scale_np, self.bounds_transform_np,)
        sdf_mesh = draw3D.draw_mesh(sdf,  self.scene_scale_np, self.bounds_transform_np, grid_activate,)
        # sdf_mesh = draw3D.draw_mesh(sdf,  self.scene_scale_np, self.bounds_transform_np,)
        return sdf_mesh

    def write_mesh(self, filename, save_local = False, im_pose=None):
        mesh = self.mesh_rec()
        data = trimesh.exchange.ply.export_ply(mesh)
        out = open(filename, "wb+")
        out.write(data)
        out.close()
        if save_local:
            mesh = self.mesh_local_rec()
            data = trimesh.exchange.ply.export_ply(mesh)
            out = open(filename+"_local.ply", "wb+")
            out.write(data)
            out.close()
        if im_pose is not None:
            scene = trimesh.Scene(mesh)
            im = draw3D.capture_scene_im(
                scene, im_pose, tm_pose=True)
            cv2.imwrite(filename[:-4] + ".png", im[..., :3][..., ::-1])

    def compute_slices(
        self, z_ixs=None, n_slices=6,
        include_gt=False, include_diff=False, include_chomp=False,
        draw_cams=False, sdf_range=[-2, 2],
    ):
        # 看看切片长什么样子
        # 计算要查询的点
        if z_ixs is None:
            # 30:6:(256-30)
            z_ixs = torch.linspace(30, self.grid_dim - 30, n_slices)
            z_ixs = torch.round(z_ixs).long()
        z_ixs = z_ixs.to(self.device)
        # 得到每个grid的真实坐标
        pc = self.grid_pc.reshape(self.grid_dim, self.grid_dim, self.grid_dim, 3)
        # 得到pc，得到俯视每个切片所有grid的坐标，第一维代表z [6, 256, 256, 3]
        pc = torch.index_select(pc, self.up_ix, z_ixs)
        if not self.up_aligned:
            # 如果没有向上对齐，不存在的，所以不用看咯
            indices = np.arange(len(z_ixs))[::-1]
            indices = torch.from_numpy(indices.copy()).to(self.device)
            pc = torch.index_select(pc, self.up_ix, indices)
        # 根据sdf_range范围得到颜色图的类
        cmap = sdf_util.get_colormap(sdf_range=sdf_range)
        # grid_shape 是 [6, 256, 256]
        grid_shape = pc.shape[:-1]
        # z的尺寸,6
        n_slices = grid_shape[self.up_ix]
        # [256 x 256 x6, 3]
        pc = pc.reshape(-1, 3)
        # x和y的尺寸
        scales = torch.cat([self.scene_scale[:self.up_ix], self.scene_scale[self.up_ix + 1:]])
        # 图像尺寸以最小的那条边为256像素
        im_size = 256 * scales / scales.min()
        im_size = im_size.int().cpu().numpy()
        # 切片类
        slices = {}
        # 设置不反向传播
        with torch.set_grad_enabled(False):
            # 获取所有栅格的sdf
            sdf = fc_map.chunks(pc, self.chunk_size, self.sdf_map)
            sdf = sdf.detach().cpu().numpy()
        # 将这些sdf折叠为一维数组，把标量映射为rgba
        sdf_viz = cmap.to_rgba(sdf.flatten(), alpha=1., bytes=False)
        # 变为rgb的图像
        sdf_viz = (sdf_viz * 255).astype(np.uint8)[..., :3]
        # 换成[6, 256, 256, 3]
        sdf_viz = sdf_viz.reshape(*grid_shape, 3)
        sdf_viz = [
            cv2.resize(np.take(sdf_viz, i, self.up_ix), im_size[::-1])
            for i in range(n_slices)
        ]
        slices["pred_sdf"] = sdf_viz
        # 碰撞成本图
        if include_chomp:
            # 利用sdf计算的代价图
            cost = metrics.chomp_cost(sdf, epsilon=2.)
            # 变为彩色图
            cost_viz = imgviz.depth2rgb(cost.reshape(self.grid_dim, -1), min_value=0., max_value=1.5)
            # 成为展示的尺寸[6, 256, 256, 3]
            cost_viz = cost_viz.reshape(*grid_shape, 3)
            cost_viz = [
                cv2.resize(np.take(cost_viz, i, self.up_ix), im_size[::-1])
                for i in range(n_slices)
            ]
            slices["pred_cost"] = cost_viz
        # 把pc变回来[6, 256, 256, 3]
        pc = pc.reshape(*grid_shape, 3)
        pc = pc.detach().cpu().numpy()
        # 如果需要展示真值图像
        if include_gt:
            # 查找真值sdf图中pc位置的
            gt_sdf = sdf_util.eval_sdf_interp(self.gt_sdf_interp, pc, handle_oob='fill')
            # 设置了透明度
            gt_sdf_viz = cmap.to_rgba(gt_sdf.flatten(), alpha=1., bytes=False)
            # 成为展示的尺寸[6, 256, 256, 4]
            gt_sdf_viz = gt_sdf_viz.reshape(*grid_shape, 4)
            # (6, 256, 256, 3)
            gt_sdf_viz = (gt_sdf_viz * 255).astype(np.uint8)[..., :3]
            gt_sdf_viz = [
                cv2.resize(np.take(gt_sdf_viz, i, self.up_ix), im_size[::-1])
                for i in range(n_slices)
            ]
            slices["gt_sdf"] = gt_sdf_viz
            # 计算真值的cost
            if include_chomp:
                gt_costs = metrics.chomp_cost(gt_sdf, epsilon=2.)
                gt_cost_viz = imgviz.depth2rgb(gt_costs.reshape(self.grid_dim, -1), min_value=0., max_value=1.5)
                gt_cost_viz = gt_cost_viz.reshape(*grid_shape, 3)
                gt_cost_viz = [
                    cv2.resize(
                        np.take(gt_cost_viz, i, self.up_ix), im_size[::-1])
                    for i in range(n_slices)
                ]
                slices["gt_cost"] = gt_cost_viz
        # 计算差值
        if include_diff:
            sdf = sdf.reshape(*grid_shape)
            diff = np.abs(gt_sdf - sdf)
            # print("The mean diff between sdf and diff is: ",  np.mean(np.mean(diff, 1),1), ", and the all mean is: ", np.mean(diff))
            diff = diff.reshape(self.grid_dim, -1)
            # 不知道为什么展示不出来
            diff_viz = imgviz.depth2rgb(diff, min_value=0., max_value=1.5)
            # diff_viz = diff_viz.reshape(-1, 3)
            # viz = np.full(diff_viz.shape, 255, dtype=np.uint8)
            viz = diff_viz.reshape(*grid_shape, 3)
            viz = [
                cv2.resize(np.take(viz, i, self.up_ix), im_size[::-1])
                for i in range(n_slices)
            ]
            slices["diff"] = viz
        # 在切片中展示相机轨迹
        if draw_cams: 
            # 相机的关键帧的坐标
            cam_xyz = self.frames.T_WC_batch[:, :3, 3].cpu()
            # 相机的关键帧的坐标在图像中的位置
            cam_td = self.to_topdown(cam_xyz, im_size)
            # 相机的旋转矩阵
            cam_rots = self.frames.T_WC_batch[:, :3, :3].cpu().numpy()
            angs = []
            for rot in cam_rots:
                # 得到角度
                ang = np.arctan2(rot[0, 2], rot[0, 0])
                angs.append(ang)
            # 将cam标记添加到预测的sdf切片
            for i, im in enumerate(slices["pred_sdf"]):
                if self.incremental:
                    # 相机轨迹
                    trajectory_gt = self.frames.T_WC_batch_np[:, :3, 3]
                    if self.frames.T_WC_gt is not None:
                        # 如果用的真值轨迹
                        trajectory_gt = self.frames.T_WC_gt[:, :3, 3]
                    # 轨迹在图像上展示结果
                    traj_td = self.to_topdown(trajectory_gt, im_size)
                    for j in range(len(traj_td) - 1):
                        if not (traj_td[j] == traj_td[j + 1]).all():
                            im = im.astype(np.uint8) / 255
                            # opencv绘制线条，关键帧之间直接直线连接
                            im = cv2.line(im, traj_td[j][::-1], traj_td[j + 1][::-1], [1., 0., 0.], 2)
                            im = (im * 255).astype(np.uint8)
                for (p, ang) in zip(cam_td, angs):
                    draw.draw_agent(im, p, agent_rotation=ang, agent_radius_px=12)
                slices["pred_sdf"][i] = im
        return slices

    def write_slices(
        self, save_path, prefix="", n_slices=6,
        include_gt=False, include_diff=False, include_chomp=False,
        draw_cams=False, sdf_range=[-2, 2],
    ):
        # 保存切片
        slices = self.compute_slices(
            z_ixs=None,
            n_slices=n_slices,
            include_gt=include_gt,
            include_diff=include_diff,
            include_chomp=include_chomp,
            draw_cams=draw_cams,
            sdf_range=sdf_range,
        )
        for s in range(n_slices):
            cv2.imwrite(
                os.path.join(save_path, prefix + f"pred_{s}.png"),
                slices["pred_sdf"][s][..., ::-1])
            if include_gt:
                cv2.imwrite(
                    os.path.join(save_path, prefix + f"gt_{s}.png"),
                    slices["gt_sdf"][s][..., ::-1])
            if include_diff:
                cv2.imwrite(
                    os.path.join(save_path, prefix + f"diff_{s}.png"),
                    slices["diff"][s][..., ::-1])
            if include_chomp:
                cv2.imwrite(
                    os.path.join(save_path, prefix + f"pred_cost_{s}.png"),
                    slices["pred_cost"][s][..., ::-1])
                cv2.imwrite(
                    os.path.join(save_path, prefix + f"gt_cost_{s}.png"),
                    slices["gt_cost"][s][..., ::-1])

    def slices_vis(self, n_slices=6):
        include_gt = True
        if self.myself:
            include_gt = False
        # 计算切片可视化
        slices = self.compute_slices(
            z_ixs=None,
            n_slices=n_slices,
            include_gt=include_gt,
            include_diff=False,
            include_chomp=False,
            draw_cams=False,
        )
        if self.myself:
            pred_sdf = np.hstack((slices["pred_sdf"]))
            viz = pred_sdf
        else:
            # 切片可视化
            gt_sdf = np.hstack((slices["gt_sdf"]))
            # gt_cost = np.hstack((slices["gt_cost"]))
            pred_sdf = np.hstack((slices["pred_sdf"]))
            viz = np.vstack((gt_sdf, pred_sdf))
            # pred_cost = np.hstack((slices["pred_cost"]))
            # diff = np.hstack((slices["diff"]))
            # viz = np.vstack((gt_sdf, pred_sdf, diff))
            # viz = np.vstack((gt_sdf, gt_cost, pred_sdf, pred_cost, diff))
        return viz

    def to_topdown(self, pts, im_size):
        # 相机从上往下看怎么绘制轨迹
        cam_homog = np.concatenate([pts, np.ones([pts.shape[0], 1])], axis=-1)
        inv_bt = np.linalg.inv(self.bounds_transform_np)
        cam_td = np.matmul(cam_homog, inv_bt.T)
        cam_td = cam_td[:, :3] / self.scene_scale.cpu().numpy()
        cam_td = cam_td / 2 + 0.5  # [-1, 1] -> [0, 1]
        cam_td = np.concatenate((cam_td[:, :self.up_ix], cam_td[:, self.up_ix + 1:]), axis=1)
        cam_td = cam_td * im_size
        cam_td = cam_td.astype(int)
        return cam_td

    def obj_slices_vis(self, n_slices=6):
        # 展示物体，也没用
        if self.obj_bounds_file is not None:
            up_ix = 1
            obj_bounds = metrics.get_obj_eval_bounds(self.obj_bounds_file, up_ix)
            cmap = sdf_util.get_colormap(sdf_range=[-0.5, 0.5])
            all_slices = []
            for bounds in obj_bounds:
                dims = [256, 256, 256]
                dims[up_ix] = n_slices
                x = torch.linspace(bounds[0, 0], bounds[1, 0], dims[0])
                y = torch.linspace(bounds[0, 1], bounds[1, 1], dims[1])
                z = torch.linspace(bounds[0, 2], bounds[1, 2], dims[2])
                xx, yy, zz = torch.meshgrid(x, y, z)
                pc = torch.cat(
                    (xx[..., None], yy[..., None], zz[..., None]), dim=3
                ).to(self.device)
                sdf = self.sdf_map(pc)
                col = cmap.to_rgba(
                    sdf.detach().cpu().numpy().flatten(),
                    alpha=1., bytes=False)
                col = (col * 255).astype(np.uint8)[..., :3]
                col = col.reshape(*pc.shape[:-1], 3)
                col = np.hstack([col[:, i] for i in range(n_slices)])
                gt_sdf = sdf_util.eval_sdf_interp(
                    self.gt_sdf_interp, pc.cpu(), handle_oob='fill')
                gt_col = cmap.to_rgba(gt_sdf.flatten(), alpha=1., bytes=False)
                gt_col = gt_col.reshape(*pc.shape[:-1], 4)
                gt_col = (gt_col * 255).astype(np.uint8)[..., :3]
                gt_col = np.hstack([gt_col[:, i] for i in range(n_slices)])
                slices = np.vstack((col, gt_col))
                all_slices.append(slices)
            return np.vstack((all_slices))
        return None

    # Evaluation methods ------------------------------------

    def eval_sdf(self, samples=200000, visible_region=True):
        """ If visible_region is True then choose random samples along rays
            in the frames. Otherwise choose random samples in the volume
            where the GT sdf is defined.
        """
        # 如果visible_region为True，则沿帧中的光线选择随机采样。否则，在定义GT sdf的体积中选择随机样本。
        # start, end = start_timing()
        # 设置每次采样每个方法是一样的，需要注意在场景范围内采样
        seed = float(self.tot_step_time//self.eval_freq_s)
        torch.manual_seed(seed)
        if visible_region:
            # 在可见区域随机采样光线，而且墙后面的也不行，所以得到的准一点
            sdf, eval_pts = self.eval_sdf_visible(samples)
        else:
            # 在整个区域随机采样光线，并得到预测的sdf
            sdf, eval_pts = self.eval_sdf_volume(samples)
        # 获得真值的sdf
        gt_sdf, valid_mask = sdf_util.eval_sdf_interp(self.gt_sdf_interp, eval_pts.cpu().detach().numpy(), handle_oob='mask')
        # gt sdf gives value 0 inside the walls. Don't include this in loss
        # gt sdf在墙内给出值0，不包括在损失中
        valid_mask = np.logical_and(gt_sdf != 0., valid_mask)
        # 得到不在墙内的sdf的gt值
        gt_sdf = gt_sdf[valid_mask]
        # 得到不在墙内的sdf的预测值
        sdf = sdf[valid_mask]
        gt_sdf = torch.from_numpy(gt_sdf).to(self.device)
        # 计算差异，获得多种类型的loss，比如l1，分层，碰撞成本
        with torch.set_grad_enabled(False):
            sdf_diff = sdf - gt_sdf
            sdf_diff = torch.abs(sdf_diff)
            l1_sdf = sdf_diff.mean()
            # 判断哪些损失在截断距离内，对这部分损失给予更好的权重
            near_surf_mask = gt_sdf < self.trunc_distance
            near_surf_sdf_diff = sdf_diff[near_surf_mask]
            far_surf_sdf_diff = sdf_diff[~near_surf_mask]
            near_surf_l1_sdf = near_surf_sdf_diff.mean()
            far_surf_l1_sdf = far_surf_sdf_diff.mean()
            weight_for_near = self.trunc_weight/(self.trunc_weight+1)
            surf_l1_sdf = near_surf_l1_sdf*weight_for_near+far_surf_l1_sdf*(1-weight_for_near)
            bins_loss = metrics.binned_losses(sdf_diff, gt_sdf)
            epsilons = [1., 1.5, 2.]
            l1_chomp_costs = [
                torch.abs(
                    metrics.chomp_cost(sdf, epsilon=epsilon) -
                    metrics.chomp_cost(gt_sdf, epsilon=epsilon)
                ).mean().item() for epsilon in epsilons
            ]
        res = {
            'av_l1': l1_sdf.item(),
            'surf_l1': surf_l1_sdf.item(),
            'binned_l1': bins_loss,
            'l1_chomp_costs': l1_chomp_costs,
        }
        # 返回当前时刻的所有loss值
        return res

    def eval_sdf_visible(self, samples=20000):
        # if self.incremental:
        #     frame_ixs = np.arange(int(self.tot_step_time * self.fps))
        #     # 目前位置的可见区域的数据
        #     sample = self.cached_dataset[frame_ixs]
        # else:
        # 都是用整个区域可见的
        sample = self.cached_dataset.get_all()
        # 目前可见位置的深度和相机位姿
        depth_batch = torch.FloatTensor(sample["depth"]).to(self.device)
        T_WC_batch = torch.FloatTensor(sample["T"]).to(self.device)
        # 每帧需要采样的数目
        rays_per_frame = samples // depth_batch.shape[0]
        dist_behind_surf = self.dist_behind_surf
        if self.dataset_format == "ScanNet":
            # For scanNet only evaluate in visible region
            dist_behind_surf == 0
        # 在这写可见部分上面进行采样
        sample_pts = self.sample_points(
            depth_batch, T_WC_batch,
            n_rays=rays_per_frame, dist_behind_surf=dist_behind_surf,
            n_strat_samples=1, n_surf_samples=0)
        pc = sample_pts["pc"]

        # # 只考虑在场景范围内的点
        # dist_to_first = pc.reshape(-1,3) - self.grid_pc[0]
        # dist_to_first = torch.matmul(self.bounds_transform[:3,:3].inverse(), dist_to_first.T).T
        # dist_to_first = dist_to_first - self.grid_scale/2
        # axis_id_3d = torch.ceil(dist_to_first/self.grid_scale)
        # out_scene = (axis_id_3d[:,0] >= self.grid_dim-1)|(axis_id_3d[:,0] <= 0)|(axis_id_3d[:,1] >= 
        #             self.grid_dim-1)|(axis_id_3d[:,1] <= 0)|(axis_id_3d[:,2] >= self.grid_dim-1)|(axis_id_3d[:,2] <= 0)
        # # print(pc.shape)
        # pc = pc[~out_scene]
        # # print(pc.shape)
        

        with torch.set_grad_enabled(False):
            # 获得这些采样点的sdf值，这个时候noise_std又成为0
            sdf = self.sdf_map(pc, noise_std=0)
        sdf = sdf.flatten()
        eval_pts = pc.squeeze()
        # 返回采样的200k个点的sdf和坐标
        return sdf, eval_pts

    def eval_sdf_volume(self, samples=20000):
        """ Sample random points in gt sdf grid volume
        """
        # 在gt的sdf的整个区域内随机采样点做评估
        eval_pts = torch.rand(samples, 3)
        eval_pts = eval_pts * (self.sdf_dims - 1)
        eval_pts = eval_pts * self.sdf_transform[0, 0]
        eval_pts = eval_pts + self.sdf_transform[:3, 3]
        # Filter points for evaluation
        if self.dataset_format == "replicaCAD":
            if self.stage_sdf_interp is None:
                stage_sdf = np.load(self.stage_sdf_file)
                transf = np.loadtxt(self.sdf_transf_file)
                self.stage_sdf_interp = sdf_util.sdf_interpolator(stage_sdf, transf)
            eval_stage_sdf = self.stage_sdf_interp(eval_pts)
            # discard_pts = eval_pts[eval_stage_sdf <= 0]
            eval_pts = eval_pts[eval_stage_sdf > 0]
            min_xy = np.loadtxt(self.seq_dir + 'bounds.txt')
            islands = np.loadtxt(self.seq_dir + 'unnavigable.txt')
            px = torch.floor((eval_pts[:, 0] - min_xy[0]) / min_xy[2])
            py = torch.floor((eval_pts[:, 2] - min_xy[1]) / min_xy[2])
            px = torch.clamp(px, min=0, max=islands.shape[1] - 1).int()
            py = torch.clamp(py, min=0, max=islands.shape[0] - 1).int()
            # discard2_pts = eval_pts[islands[py, px] == 1]
            eval_pts = eval_pts[islands[py, px] == 0]
        with torch.set_grad_enabled(False):
            eval_pts = eval_pts.float().to(self.device)
            sdf = self.sdf_map(eval_pts)
            sdf = torch.squeeze(sdf)
        # # Vis evaluation points
        # mesh_gt = trimesh.load(self.scene_file)
        # scene = trimesh.Scene(mesh_gt)
        # pc = trimesh.PointCloud(eval_pts.cpu().numpy(), [0, 255, 0, 255])
        # pc1 = trimesh.PointCloud(discard_pts.cpu().numpy(), [255, 0, 0, 255])
        # pc2 = trimesh.PointCloud(discard2_pts.cpu().numpy(), [255, 0, 0, 255])
        # scene.add_geometry([pc, pc1, pc2])
        # scene.show()
        return sdf, eval_pts

    def eval_object_sdf(self, samples=10000):
        # 如果这个序列是以对象为中心的，则注重于评估这个obj，没有用
        errors = None
        if self.obj_bounds_file is not None:
            obj_bounds = metrics.get_obj_eval_bounds(
                self.obj_bounds_file, self.up_ix)
            obj_bounds = torch.FloatTensor(obj_bounds).to(self.device)
            # is object visible yet
            offsets = torch.rand(100, 3).to(self.device)
            extents = obj_bounds[:, 1] - obj_bounds[:, 0]
            pts = obj_bounds[:, 0] + offsets[:, None] * extents
            if self.incremental:
                frame_ixs = np.arange(int(self.tot_step_time * self.fps))
                sample = self.cached_dataset[frame_ixs]
            else:
                sample = self.cached_dataset.get_all()
            depth_batch = torch.FloatTensor(sample["depth"]).to(self.device)
            T_WC_batch = torch.FloatTensor(sample["T"]).to(self.device)
            visible = geometry.frustum.is_visible_torch(
                pts.view(-1, 3), T_WC_batch, depth_batch,
                self.H, self.W, self.fx, self.fy, self.cx, self.cy,
                trunc=0.05)
            visible = visible.detach().cpu().numpy().sum(axis=0) > 0
            visible = visible.reshape(100, len(obj_bounds))
            visible_prop = visible.sum(axis=0) / 100
            visible = visible_prop > 0.5
            errors = []
            for i in range(len(obj_bounds)):
                if visible[i]:
                    offsets = torch.rand(samples, 3).to(self.device)
                    bounds = obj_bounds[i]
                    extents = bounds[1] - bounds[0]
                    pts = bounds[0] + offsets * extents[None, :]
                    gt_sdf, valid_mask = sdf_util.eval_sdf_interp(
                        self.gt_sdf_interp, pts.cpu().numpy(),
                        handle_oob='mask')
                    with torch.set_grad_enabled(False):
                        sdf = self.sdf_map(pts)
                        sdf = torch.squeeze(sdf)
                    gt_sdf = gt_sdf[valid_mask]
                    sdf = sdf[valid_mask].cpu().numpy()
                    l1 = np.abs(gt_sdf - sdf).mean()
                    errors.append(l1)
                else:
                    errors.append(np.nan)
        return errors

    def eval_mesh(self, samples=200000):
        # 评估mesh，判断精度和完成度
        # 加载真值mesh，.obj文件
        # 设置每次采样每个方法是一样的
        seed = float(self.tot_step_time//self.eval_freq_s)
        torch.manual_seed(seed)
        mesh_gt = trimesh.load(self.scene_file)
        # 当前预测的mesh
        sdf_mesh = self.mesh_rec()
        # 计算mesh的精度和完成度
        acc, comp = metrics.accuracy_comp(mesh_gt, sdf_mesh, samples=samples)
        return acc, comp

    def sdf_fn(self, pts):
        # sdf的计算，返回pts的sdf值
        with torch.set_grad_enabled(False):
            pts = torch.FloatTensor(pts).to(self.device)
            sdf = self.sdf_map(pts)
        return sdf.detach().cpu().numpy()

    def grad_fn(self, pts):
        # grad的计算，返回pts处的sdf值
        pts = torch.FloatTensor(pts).to(self.device)
        pts.requires_grad_()
        sdf = self.sdf_map(pts)
        sdf_grad = fc_map.gradient(pts, sdf)
        return sdf_grad.detach().cpu().numpy()

    def eval_fixed(self):
        # 取出一个评估时刻，这是评估voxblox用的，我觉得不用
        t = self.eval_times.pop(0)
        # 把sdf和grad的计算方式都传递过去，进行评估
        return eval_pts.fixed_pts_eval(
            self.sdf_fn, t, self.eval_pts_dir,
            self.seq_dir, self.dataset_format,
            self.cached_dataset, self.dirs_C.cpu(),
            self.gt_sdf_interp, self.eval_pts_root,
            len(self.scene_dataset), grad_fn=self.grad_fn,
        )



    # Data methods ---------------------------------------

    def load_data(self):
        # 图像预处理
        rgb_transform = transforms.Compose(
            [image_transforms.BGRtoRGB()])
        depth_transform = transforms.Compose(
            [image_transforms.DepthScale(self.inv_depth_scale),
             image_transforms.DepthFilter(self.max_depth)])
        if self.dataset_format == "ScanNet":
            dataset_class = dataset.ScanNetDataset
            col_ext = ".jpg"
            self.up = np.array([0., 0., 1.])
            ims_file = self.scannet_dir
        elif self.dataset_format == "replica":
            dataset_class = dataset.ReplicaDataset
            col_ext = ".jpg"
            self.up = np.array([0., 1., 0.])
            ims_file = self.ims_file
        elif self.dataset_format == "replicaCAD":
            dataset_class = dataset.ReplicaDataset
            col_ext = ".png"
            # 图像到俯视图需要进行的变换
            self.up = np.array([0., 1., 0.])
            ims_file = self.ims_file
        # 数据集的类
        self.scene_dataset = dataset_class(
            ims_file,
            self.traj_file,
            rgb_transform=rgb_transform,
            depth_transform=depth_transform,
            col_ext=col_ext,
            noisy_depth=self.noisy_depth,
        )
        if self.incremental is False:
            # 如果不是增量式的话，要自己构建索引
            if "im_indices" not in self.config["dataset"]:
                if "n_random_views" in self.config["dataset"]:
                    n_random_views = self.config["dataset"]["n_random_views"]
                    if n_random_views > 0:
                        n_dataset = len(self.scene_dataset)
                        self.indices = np.random.choice(
                            np.arange(0, n_dataset),
                            size=n_random_views,
                            replace=False)
            print("Frame indices", self.indices)
            idxs = self.indices
            frame_data = self.get_data(idxs)
            self.add_data(frame_data)

    def get_data(self, idxs):
        # 获取某帧数据的数据类，用于把数据都加载进来
        frames_data = FrameData()
        for idx in idxs:
            # 返回rgb和深度图像的矩阵信息，以及pose
            sample = self.scene_dataset[idx]
            im_np = sample["image"][None, ...]
            depth_np = sample["depth"][None, ...]
            T_np = sample["T"][None, ...]
            # rgb图像归一化了一下
            im = torch.from_numpy(im_np).float().to(self.device) / 255.
            depth = torch.from_numpy(depth_np).float().to(self.device)
            T = torch.from_numpy(T_np).float().to(self.device)
            # 一个数据操作的类
            data = FrameData(
                frame_id=np.array([idx]),
                im_batch=im,
                im_batch_np=im_np,
                depth_batch=depth,
                depth_batch_np=depth_np,
                T_WC_batch=T,
                T_WC_batch_np=T_np,
            )
            if self.do_normal:
                # 需要计算法线，是需要的，得到整个点云，为点云计算法线，得到每个点的法线
                pc = geometry.transform.pointcloud_from_depth_torch(depth[0], self.fx, self.fy, self.cx, self.cy)
                normals = geometry.transform.estimate_pointcloud_normals(pc)
                normals = geometry.transform.adjust_normals_outward(normals, pc, T[0])
                data.normal_batch = normals[None, :]
                # dyndyn：计算渲染和深度的得分
                # 渲染得分
                normals_render = torch.sum(self.dirs_C[0] * normals, axis = 2)
                normal_block = normals_render.view(self.loss_approx_factor, self.h_block, self.loss_approx_factor, self.w_block)
                normal_loss = normal_block.var(dim=(1,3))
                normal_loss[torch.isnan(normal_loss) ] = 0.
                normal_loss = (normal_loss-normal_loss.min())/(normal_loss.max()-normal_loss.min())  
                # 深度得分
                depth_block = depth.view(self.loss_approx_factor, self.h_block, self.loss_approx_factor, self.w_block)
                depth_loss = depth_block.var(dim=(1,3))
                depth_loss = (depth_loss-depth_loss.min())/(depth_loss.max()-depth_loss.min())  
                # 总得分并归一化
                all_loss = 0.3 * normal_loss + 0.7 * depth_loss
                scores = all_loss / all_loss.sum()
                data.score_batch = scores[None, :]
            if self.gt_traj is not None:
                # 如果有真正的轨迹的话，其实用的不是真值
                data.T_WC_gt = self.gt_traj[idx][None, ...]
            # 添加数据，一定不是替换最后一个，因为这是一个空的数据类型
            frames_data.add_frame_data(data, False)
        return frames_data

    def add_data(self, data):
        # 新的时刻添加新的数据进行，如果最后一帧不是关键帧，则新帧将替换批次中的最后一帧
        self.frames.add_frame_data(data, True)

    def add_frame(self, frame_data):
        # 如果不管上一帧率是不是关键帧，都要计算新的帧的loss
        self.frozen_sdf_map = copy.deepcopy(self.sdf_map)
        # 添加数据进去
        self.add_data(frame_data)
        self.steps_since_frame = 0
        self.last_is_keyframe = False
        self.optim_frames = self.iters_per_frame
        self.noise_std = self.noise_frame
        self.rand_id_ok = None

    # Keyframe methods ----------------------------------

    # def is_keyframe(self, T_WC, depth_gt):
    #     # 判断某一帧是否为关键帧
    #     # 对其进行采样
    #     sample_pts = self.sample_points(depth_gt, T_WC, n_rays=self.n_rays_is_kf, dist_behind_surf=0.8)
    #     pc = sample_pts["pc"]
    #     z_vals = sample_pts["z_vals"]
    #     depth_sample = sample_pts["depth_sample"]
    #     #评估采样点的sdf
    #     with torch.set_grad_enabled(False):
    #         # 把添加了关键帧之后的模型冷冻一下，作为判断是否需要添加新的关键帧的依据
    #         sdf = self.frozen_sdf_map(pc, noise_std=self.noise_std)
    #     z_vals, ind1 = z_vals.sort(dim=-1)
    #     ind0 = torch.arange(z_vals.shape[0])[:, None].repeat(1, z_vals.shape[1])
    #     # sdf安装z_vals排了一下序, 但数目有限
    #     sdf = sdf[ind0, ind1]
    #     # 利用预测的sdf渲染图像，但只能得到光线位置的渲染 n x 27
    #     view_depth = render.sdf_render_depth(z_vals, sdf)
    #     # 计算渲染得到的图像与真值图像的采样位置的误差
    #     loss = torch.abs(view_depth - depth_sample) / depth_sample
    #     # 判断有哪些采样光线的loss低于设置的损失阈值
    #     below_th = loss < self.kf_dist_th
    #     size_loss = below_th.shape[0]
    #     # loss较低的像素占比有多少
    #     below_th_prop = below_th.sum().float() / size_loss
    #     # 如果loss较低的比较少，也就是loss差异较大的比较多，那就是关键帧
    #     is_keyframe = below_th_prop.item() < self.kf_pixel_ratio
    #     # # 给出当前帧是否应该作为关键帧的依据
    #     # print(
    #     #     "Proportion of loss below threshold",
    #     #     below_th_prop.item(),
    #     #     "for KF should be less than",
    #     #     self.kf_pixel_ratio,
    #     #     ". Therefore is keyframe:",
    #     #     is_keyframe
    #     # )
    #     return is_keyframe

    # def check_keyframe_latest(self):
    #     """
    #     returns whether or not to add a new frame.
    #     返回是否需要添加一个新的帧进来
    #     """
    #     add_new_frame = False
    #     if self.last_is_keyframe:
    #         # 如果已经是关键帧，就不要在设定这是关键帧了
    #         add_new_frame = True
    #     else:
    #         # Check if latest frame should be a keyframe.
    #         # 检查最新帧是否应为关键帧。
    #         T_WC = self.frames.T_WC_batch[-1].unsqueeze(0)
    #         depth_gt = self.frames.depth_batch[-1].unsqueeze(0)
    #         # 判断最新帧是否为关键帧
    #         self.last_is_keyframe = self.is_keyframe(T_WC, depth_gt)
    #         # 如果是关键帧的话，要把优化次数纠正一下
    #         if self.last_is_keyframe:
    #             self.optim_frames = self.iters_per_kf
    #             # 噪声，不是很懂
    #             self.noise_std = self.noise_kf
    #         else:
    #             add_new_frame = True
    #     return add_new_frame


    # Main training methods ----------------------------------

    
    def sample_points(
        self,
        depth_batch,
        T_WC_batch,
        norm_batch=None,
        score_batch=None,
        active_loss_approx=None,
        n_rays=None,
        dist_behind_surf=None,
        n_strat_samples=None,
        n_surf_samples=None,
    ):
        # 得到所有的采样数据，用于训练
        # 首先采样像素，然后沿反向投影光线采样深度，从而采样点。
        if n_rays is None:
            # 采样的光线数量
            n_rays = self.n_rays
        if dist_behind_surf is None:
            # 考虑的障碍物后面的距离
            dist_behind_surf = self.dist_behind_surf
        if n_strat_samples is None:
            # 均匀采样的个数
            n_strat_samples = self.n_strat_samples
        if n_surf_samples is None:
            # 高斯采样的个数
            n_surf_samples = self.n_surf_samples
        # 这一批次用来训练的图像个数
        n_frames = depth_batch.shape[0]
        if active_loss_approx is None:
            # 获取采样点的像素
            if score_batch is not None:
                # indices_b, indices_h, indices_w = sample.sample_pixels( n_rays, n_frames, self.H, self.W, device=self.device)
                indices_b, indices_h, indices_w = sample.sample_pixels_score(score_batch, n_rays, n_frames, self.H, self.W, self.h_block, self.w_block, self.loss_approx_factor, self.c, self.r,  device=self.device)
            else:
                indices_b, indices_h, indices_w = sample.sample_pixels( n_rays, n_frames, self.H, self.W, device=self.device)
        else:
            raise Exception('Active sampling not currently supported.')

        # 上面只是得到采样像素，还要得到采样光线，对深度为0的采样点过滤，并获得了光线方向dirs_C_sample
        get_masks = active_loss_approx is None
        (
            dirs_C_sample,
            depth_sample,
            norm_sample,
            T_WC_sample,
            binary_masks,
            indices_b,
            indices_h,
            indices_w
        ) = sample.get_batch_data(
            depth_batch,
            T_WC_batch,
            self.dirs_C,
            indices_b,
            indices_h,
            indices_w,
            norm_batch=norm_batch,
            get_masks=get_masks,
        )

        # 最大深度，为真实深度基础上加上高斯采样的范围
        max_depth = depth_sample + dist_behind_surf
        # 还要在光线上采样，得到了世界坐标下的点云数据
        pc, z_vals, pc_surf, z_vals_surf = sample.sample_along_rays(
            T_WC_sample,
            self.min_depth,
            max_depth,
            n_strat_samples,
            n_surf_samples,
            dirs_C_sample,
            gt_depth=depth_sample,
            grad=False,
        )
        # 这是训练所需的所有东西
        sample_pts = {
            "depth_batch": depth_batch,
            "pc": pc,
            "z_vals": z_vals,
            "indices_b": indices_b,
            "indices_h": indices_h,
            "indices_w": indices_w,
            "dirs_C_sample": dirs_C_sample,
            "depth_sample": depth_sample,
            "T_WC_sample": T_WC_sample,
            "norm_sample": norm_sample,
            "binary_masks": binary_masks,
        }
        return sample_pts
    

    def find_visable_grids(self, depth, grid_pcs, w2c, down=5):
        if_time = False
        if if_time:
            start_time = time.time()
        '''
        输入深度图像和图像的全局姿态，返回可见的栅格，并生成当前帧的点云
        '''
        # 1. 生成栅格的索引
        grid_pcs_h = torch.cat([grid_pcs, torch.ones(grid_pcs.shape[0], 1, device='cuda')], dim=1)
        if if_time:
            end_time = time.time()
            print("The gen time is", end_time-start_time)
            start_time = time.time()
        # 2. 将世界坐标系的点转换为相机坐标系
        cam_points = (w2c @ grid_pcs_h.T).T
        img_points = (self.k @ cam_points[:, :3].T).T
        img_points = img_points[:, :2] / img_points[:, 2:]
        # 3. 保留在图像范围内的点
        valid_mask = (img_points[:, 0] >= 0) & (img_points[:, 0] < self.W) & (img_points[:, 1] >= 0) & (img_points[:, 1] < self.H)
        valid_img_points = img_points[valid_mask]
        valid_cam_points = cam_points[valid_mask]
        valid_all_grid_indices = self.all_grid_indices[valid_mask]
        if if_time:
            end_time = time.time()
            print("The ima time is", end_time-start_time)
            start_time = time.time()
        # 4. 根据像素位置提取深度值
        pixel_x = valid_img_points[:, 0].long()
        pixel_y = valid_img_points[:, 1].long()
        depth_values = depth[pixel_y, pixel_x]
        if if_time:
            end_time = time.time()
            print("The depth time is", end_time-start_time)
            start_time = time.time()
        # 5. 生成完整的点云（当前帧），仅保留深度值 > 0 的像素
        valid_depth_mask = depth > 0
        valid_depth_mask = valid_depth_mask[::down, ::down]
        u, v = torch.meshgrid(torch.arange(0, self.H, down, device='cuda'), torch.arange(0, self.W, down, device='cuda'), indexing='ij')
        u = u[valid_depth_mask].float()
        v = v[valid_depth_mask].float()
        depth_valid = depth[::down, ::down][valid_depth_mask]  # 过滤掉深度为 0 的像素
        # 将有效深度像素转换为相机坐标
        x = (v - self.cx) * depth_valid / self.fx
        y = (u - self.cy) * depth_valid / self.fy
        z = depth_valid
        current_frame_pcd = torch.stack((x, y, z), dim=-1)  # 点云 (有效像素数量, 3)
        current_frame_pcd_h = torch.cat([current_frame_pcd, torch.ones(current_frame_pcd.shape[0], 1, device='cuda')], dim=1)  # 增加齐次坐标
        current_frame_pcd = (w2c.inverse() @ current_frame_pcd_h.T).T[:, :3]  # 转换到世界坐标系
        if if_time:
            end_time = time.time()
            print("The current time is", end_time-start_time)
            start_time = time.time()
        # 6. 判断哪些点满足深度范围
        point_depths = valid_cam_points[:, 2]
        valid_depth_mask = (point_depths > 0) & (point_depths < (depth_values + self.dist_behind_surf))  & (depth_values>0)
        valid_all_grid_indices = valid_all_grid_indices[valid_depth_mask]
        # 在后面的部分给出索引，是在所有可视栅格的索引
        behind_index = point_depths[valid_depth_mask] > depth_values[valid_depth_mask]
        # near_face_index = abs(point_depths[valid_depth_mask] - depth_values[valid_depth_mask]) < self.trunc_distance
        near_face_index = abs(point_depths[valid_depth_mask] - depth_values[valid_depth_mask]) < 0.1
        if if_time:
            end_time = time.time()
            print("The near time is", end_time-start_time)
            start_time = time.time()
        # 返回可见栅格索引、可见点云、以及根据深度图像生成的当前帧的点云
        return valid_all_grid_indices, grid_pcs[valid_all_grid_indices], behind_index, near_face_index, current_frame_pcd, u.long(), v.long()

    def compute_local_variance(self, gradient_field, valid_mask, window_size=3):
        padding = window_size // 2
        # 0. 梯度归一化
        norm = torch.norm(gradient_field, dim=-1, keepdim=True)  # 计算每个点的L2范数
        gradient_field = torch.where(norm > 0, gradient_field / norm, torch.zeros_like(gradient_field))
        # 1. 计算局部均值，首先对无效梯度进行掩码处理
        masked_gradients = gradient_field * valid_mask.unsqueeze(-1)  # (N, N, N, 3)
        # 检查 masked_gradients 的维度
        if masked_gradients.shape[0] == 0 or masked_gradients.shape[1] == 0 or masked_gradients.shape[2] == 0:
            raise ValueError("Masked gradients are empty.")
        # 2. 计算有效梯度的局部均值
        masked_gradients = masked_gradients.permute(3, 0, 1, 2).unsqueeze(0)  # (1, 3, N, N, N)
        # 2. 计算局部均值，只考虑有效值
        local_sum = F.avg_pool3d(masked_gradients, 
                                kernel_size=window_size, 
                                stride=1, 
                                padding=padding)
        valid_count = F.avg_pool3d(valid_mask.unsqueeze(0).unsqueeze(0).float(), 
                                kernel_size=window_size, 
                                stride=1, 
                                padding=padding) + 1e-8  # 避免除零
        # local_mean = local_sum / valid_count  # 只对有效值计算均值
        local_mean = local_sum
        # 3. 计算局部方差
        squared_diff = (masked_gradients - local_mean) ** 2
        local_variance_sum = F.avg_pool3d(squared_diff, 
                                        kernel_size=window_size, 
                                        stride=1, 
                                        padding=padding)
        local_variance = local_variance_sum / valid_count  # 根据有效值数量归一化
        # 最终返回局部方差，去掉多余的维度
        var_value = torch.mean(local_variance.squeeze(0), dim=0)
        var_value = var_value.flatten()
        return var_value


    def sdf_eval_and_loss_new(
        self,
        depth_this,
        T_WC_this,
        t,
        norm_this=None,
        score_this=None,
        active_loss_approx=None,
        n_rays=None,
        dist_behind_surf=None,
        n_strat_samples=None,
        n_surf_samples=None,
        rand_id=None
    ):
        if_time = False
        if if_time:
            start_time = time.time()

        if rand_id is None:
            visable_index, visable_grids, behind_index, near_face_index, gt_pc, gt_pc_pixel_u,  gt_pc_pixel_v= self.find_visable_grids(depth_this, self.grid_pc, T_WC_this.inverse(),down=2)
            if self.all_gt_pc is not None:
                self.all_gt_pc = torch.cat((self.all_gt_pc, gt_pc))
            else:
                self.all_gt_pc = gt_pc
            if if_time:
                end_time = time.time()
                print("The find visable time is", end_time-start_time)
                start_time = time.time()
            # 本来就是顺序分布的，直接降采样即可
            # visable_index = visable_index[::300]
            # visable_grids = visable_grids[::300]
            # behind_index = behind_index[::300]
            # visable_index = torch.cat([visable_index[~near_face_index][::500], visable_index[near_face_index][::50]], dim=0)
            # visable_grids = torch.cat([visable_grids[~near_face_index][::500], visable_grids[near_face_index][::50]], dim=0)
            # behind_index = torch.cat([behind_index[~near_face_index][::500], behind_index[near_face_index][::50]], dim=0)
            visable_index = torch.cat([visable_index[~near_face_index][::500], visable_index[near_face_index][::50]], dim=0)
            visable_grids = torch.cat([visable_grids[~near_face_index][::500], visable_grids[near_face_index][::50]], dim=0)
            behind_index = torch.cat([behind_index[~near_face_index][::500], behind_index[near_face_index][::50]], dim=0)
            
            # 使用所有的栅格和所有的点云太多了
            diff = visable_grids[:, None, :] - gt_pc[None, :, :]  # M x N x 3
            dists = diff.norm(p=2, dim=-1)  # 计算欧几里得距离 M x N
            dists, closest_ixs = dists.min(dim=-1)  # 对每个 visable_grid 找到最近的 gt_pc，返回最小距离和索引
            dists[behind_index] *= -1  # 对 behind_index 中的点，将距离标记为负数
        
            # 计算方向向量，但这样计算出来的梯度其实在空闲区域是不准确的
            # directions = gt_pc[closest_ixs] - visable_grids  # M x 3
            directions = visable_grids - gt_pc[closest_ixs]  # M x 3
            eps = 1e-8  # 避免除以零
            this_grad = directions / (torch.norm(directions, dim=1, keepdim=True) + eps)  # 单位方向向量
            this_grad[behind_index] *= -1  
            
            if if_time:
                end_time = time.time()
                print("The cal time is", end_time-start_time)
                start_time = time.time()

            self.grid_activate[visable_index] = 1
            weight_old = self.grid_sdf[visable_index, 1]
            # 使用bound的大小作为权重
            weight_now = torch.exp(-abs(dists)*5)
            weight_now[weight_now<1e-3] = 1e-3
            # weight_now = 1-abs(bounds)
            # weight_now[weight_now<0.1] = 0.1
            # 使用1作为权重
            # weight_now = torch.ones(bounds.shape).to(self.device)
            weight_new = weight_old + weight_now
            # 对这部分grid的sdf进行更新
            visable_index_old_sdf = self.grid_sdf[visable_index, 0]
            
            # self.grid_sdf[visable_index, 0] = (self.grid_sdf[visable_index, 0] * weight_old + dists * weight_now)/weight_new
            # self.grid_sdf[visable_index, 3:6] = (self.grid_sdf[visable_index, 3:6] * weight_old[:,None] + this_grad * weight_now[:,None])/weight_new[:,None]
            # self.grid_sdf[visable_index, 1] = weight_new
            
            # 消融实验，不更新
            self.grid_sdf[visable_index, 0] = torch.where(
                weight_old > 0,  self.grid_sdf[visable_index, 0], dists  # 当 weight_old == 0 时，使用最新值 dists
            )
            self.grid_sdf[visable_index, 3:6] = torch.where(
                weight_old[:, None] > 0, self.grid_sdf[visable_index, 3:6],  this_grad  # 当 weight_old == 0 时，使用最新的 this_grad
            )
            self.grid_sdf[visable_index, 1] = torch.where(
                weight_old > 0,  weight_old, weight_new   # 使用最新值
            )
            
            visable_index_new_sdf = self.grid_sdf[visable_index, 0]
            
            if if_time:
                end_time = time.time()
                print("The fusion time is", end_time-start_time)
                start_time = time.time()

            # 对于当前帧，只引入变化大的
            visable_index_change = visable_index[abs(visable_index_old_sdf-visable_index_new_sdf)>0.005]
            activate_id = torch.nonzero(self.grid_activate)[:,0]
            # near_face = (self.grid_sdf[:,0] < self.trunc_distance) & (self.grid_activate>0)
            near_face = (self.grid_sdf[:,0] < 0.1) & (self.grid_activate>0)
            # surface_id = torch.nonzero(near_face.bool(), as_tuple=False)[:,0]
            # 计算梯度局部变化大的区域，一般是复杂的薄几何和精细物体
            local_variance = self.compute_local_variance(self.grid_sdf[:, 3:6].reshape(self.grid_dim,self.grid_dim,self.grid_dim,-1),
                                                                     self.grid_activate.reshape(self.grid_dim,self.grid_dim,self.grid_dim),window_size=7)
            activated_values = local_variance[near_face]  # 提取被激活的值
            threshold_index = int(len(activated_values) * 0.2)  # 前面一些的索引
            if threshold_index == 0:
                raise ValueError("No activated values to process.")
            sorted_values, sorted_indices = torch.sort(activated_values, descending=True)
            top_percent_indices = sorted_indices[:threshold_index]
            surface_id = torch.nonzero(near_face.bool(), as_tuple=False)[:,0][top_percent_indices]
            rand_id = torch.cat((visable_index_change, activate_id[::50], surface_id[::10]))
            # rand_id = torch.cat((activate_id[::50], surface_id[::10]))
            # rand_id = torch.cat((visable_index_change, surface_id[::10]))
            # rand_id = torch.cat((visable_index_change, activate_id[::50], ))
            print(len(visable_index_change))
            print(len(activate_id[::50]))
            print(len(surface_id[::20]))
            
            # # # debug 可视化一下
            # non_indices = sorted_indices[threshold_index:]  # 取出剩下的索引
            # non_coords = torch.nonzero(near_face.bool(), as_tuple=False)[:, 0][non_indices]
            # if (len(surface_id)>0):
            #     near_face_grids = self.grid_pc[surface_id[::10]].cpu().numpy()
            #     nf_pcd = o3d.geometry.PointCloud()
            #     nf_pcd.points = o3d.utility.Vector3dVector(near_face_grids)
            #     nf_pcd.colors = o3d.utility.Vector3dVector([[1, 0, 0] for _ in range(len(near_face_grids))])
            #     near_grids = self.grid_pc[non_coords[::10]].cpu().numpy()
            #     grid_pcd = o3d.geometry.PointCloud()
            #     grid_pcd.points = o3d.utility.Vector3dVector(near_grids)
            #     grid_pcd.colors = o3d.utility.Vector3dVector([[0.5, 0.5, 0.5] for _ in range(len(near_grids))])
            #     o3d.visualization.draw_geometries([nf_pcd, grid_pcd])

        # debug,测试rand_id只在表面附近分布
        # near_face = (self.grid_sdf[:,0] < 0.15) & (self.grid_activate>0)
        # activate_id = torch.nonzero(near_face)[:,0]
        # rand_id = activate_id[::10] 
        # # 可视化一下这部分

        # # debug 可视化一下表面区域的法线对不对
        # if t%300 == 0:
        #     near_face = (self.grid_sdf[:,0] < 0.01) & (self.grid_activate>0)
        #     near_face_id = torch.nonzero(near_face)[:,0]
        #     use_grids = self.grid_pc[near_face_id].cpu().numpy()
        #     grad = self.grid_sdf[near_face_id,3:6].cpu().numpy()
        #     grid_pcd = o3d.geometry.PointCloud()
        #     grid_pcd.points = o3d.utility.Vector3dVector(use_grids)
        #     grid_pcd.colors = o3d.utility.Vector3dVector([[0, 1, 0] for _ in range(len(use_grids))])
        #     lines = []
        #     points_for_lines = []
        #     for i, point in enumerate(use_grids):
        #         closest_point = point+grad[i]/10
        #         points_for_lines.append(point)
        #         points_for_lines.append(closest_point)
        #         lines.append([2 * i, 2 * i + 1])  # 每次添加两个点作为一条线的起点和终点
        #     # 将 points 和 lines 传入 LineSet
        #     lineset = o3d.geometry.LineSet()
        #     lineset.points = o3d.utility.Vector3dVector(np.array(points_for_lines))
        #     lineset.lines = o3d.utility.Vector2iVector(np.array(lines))
        #     # 为每条线指定颜色
        #     colors = [[1, 0, 0] for _ in range(len(lines))]  # 将线条设为红色
        #     lineset.colors = o3d.utility.Vector3dVector(colors)
        #     # 可视化点云和连线
        #     o3d.visualization.draw_geometries([grid_pcd, lineset])
        # if t%300 == 0:
        #     near_face = (self.grid_sdf[:,0] < 0.01) & (self.grid_activate>0)
        #     activate_id = torch.nonzero(near_face)[:,0]
        #     grid_pcd = o3d.geometry.PointCloud()
        #     near_face_grids = self.grid_pc[activate_id].cpu().numpy()
        #     grid_pcd.points = o3d.utility.Vector3dVector(near_face_grids)
        #     # 可视化点云和连线
        #     o3d.visualization.draw_geometries([grid_pcd])
        
        grid_usd = self.grid_pc[rand_id]
        do_sdf_grad = self.eik_weight != 0 or self.grad_weight != 0

        if do_sdf_grad:
            # 需要为张量计算梯度
            grid_usd.requires_grad_()
        # 把这些点云输入网络中，得到sdf值，这是输出加上噪声然后又缩放的结果
        # grid_usd = torch.cat((grid_usd, gt_pc))
        # print(len(grid_usd))
        # print(len(gt_pc))
        sdf = self.sdf_map(grid_usd, noise_std=self.noise_std)
        # 求取误差
        sdf_grad = None
        if do_sdf_grad:
            # 利用这些采样点，和对应的sdf，计算对当前网络输出的法线，也是[n,27,3]，是在每个点的三维方向
            sdf_grad = fc_map.gradient(grid_usd, sdf)
        # 计算sdf损失，trunc_distance用来判断是不是表面点，论文中有，free和surf的sdf计算方式还不一样，得到[n, 27]的损失
        sdf_loss_mat, free_space_ixs = loss.sdf_loss(sdf, self.grid_sdf[rand_id, 0], self.trunc_distance, loss_type=self.loss_type)
        # sdf_loss_mat, free_space_ixs = loss.sdf_loss(sdf, self.grid_sdf[rand_id, 0], 0.1, loss_type=self.loss_type)
        # 计算eik的损失，就是看预测的grad法向量是不是1
        eik_loss_mat = None
        if self.eik_weight != 0:
            eik_loss_mat = torch.abs(sdf_grad.norm(2, dim=-1) - 1)
        # 计算grad的损失
        grad_loss_mat = None
        if self.grad_weight != 0:
            # 计算grad损失就是我们认为的真值和sdf
            grad_loss_mat = 1 - self.cosSim(self.grid_sdf[rand_id, 3:6], sdf_grad )
            if self.orien_loss:
                grad_loss_mat = (grad_loss_mat > 1).float()

        #  可以在表面附近计算正负关系，保证表面精度，正为正，负为负
        # difference = sdf[-len(near_face_id):] * self.grid_sdf[near_face_id, 0]
        # sign_loss = torch.mean(torch.sigmoid(-difference * 1e+5))

        # 计算总体损失，total_loss为最终平均值，total_loss_mat为每个点的值，losses为一个类，用来终端打印
        total_loss, total_loss_mat, losses = loss.tot_loss(
            sdf_loss_mat, grad_loss_mat, eik_loss_mat,
            free_space_ixs, self.grid_sdf[rand_id, 0], self.eik_apply_dist,
            self.trunc_weight, self.grad_weight, self.eik_weight,
        )

        # self.grid_sdf[rand_id, 2] = total_loss_mat.detach()

        if if_time:
            end_time = time.time()
            print("The loss time is", end_time-start_time)
            start_time = time.time()

        # 返回这一训练批次的总loss，需要发布的loss类，分块的loss[n_used_kf, 8, 8]，分帧的loss[n_used_kf]
        return (
            total_loss,
            losses,
            rand_id
        )


    def step(self, t=0):
        # 处理一次迭代的主要函数，计算loss
        start, end = start_timing()
        # 所有的关键帧的数据
        depth_batch = self.frames.depth_batch
        T_WC_batch = self.frames.T_WC_batch
        # dyndyn：需要使用到frames的score
        score_batch = self.frames.score_batch if self.do_normal else None
        norm_batch = self.frames.normal_batch if self.do_normal else None
        n_frames = len(self.frames)
        idxs = [n_frames-1]
        # 这些是选取出来进行训练的图像和轨迹数据
        depth_this = depth_batch[idxs][0]
        score_batch = score_batch[idxs][0]
        T_WC_this = T_WC_batch[idxs][0]
        norm_this = norm_batch[idxs][0]
        total_loss, losses, rand_id_ok =  self.sdf_eval_and_loss_new(depth_this, T_WC_this,t, norm_this=norm_this, rand_id=self.rand_id_ok)
        self.rand_id_ok = rand_id_ok
        total_loss.backward()
        # 优化器对参数进行优化
        self.optimiser.step()
        # total_params = sum(p.numel() for p in self.sdf_map.parameters())
        # total_params += sum(p.numel() for p in self.sdf_map.buffers())
        # print(f'{total_params:,} total parameters.')
        # print(f'{total_params/(1024*1024):.2f}M total parameters.')
        for param_group in self.optimiser.param_groups:
            params = param_group["params"]
            for param in params:
                param.grad = None
        # 记录现在时间
        step_time = end_timing(start, end)
        # 预热阶段不做时间处理
        if t<200:
            step_time = 0
        time_s = step_time / 1000.
        # 模拟速度为1，所以总时间就是加上花费的真实时间
        self.tot_step_time += (1 / self.frac_time_perception) * time_s
        # 处理的帧数加一
        self.steps_since_frame += 1
        return losses, step_time