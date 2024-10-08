# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

#!/usr/bin/env python
import torch
import numpy as np
import json
import os
from datetime import datetime
import argparse
from torch.utils.tensorboard import SummaryWriter

import sys
sys.path.append("/code/dyn/SDF/LGSDF")

from lgsdf import visualisation
from lgsdf.modules import trainer
from lgsdf.eval.metrics import start_timing, end_timing



def train(
    device,
    config_file,
    chkpt_load_file=None,
    incremental=True,
    # vis
    if_vis = True,
    show_obj=False,
    update_im_freq=50,
    update_mesh_freq=200,
    # save
    save_path=None,
    use_tensorboard=False,
):
    # init trainer-------------------------------------------------------------
    # 初始化一个训练器，加载了所有必要参数
    lgsdf_trainer = trainer.Trainer(
        device,
        config_file,
        chkpt_load_file=chkpt_load_file,
        incremental=incremental,
    )

    # saving init--------------------------------------------------------------
    # 建立保存目录
    save = save_path is not None
    save_final = True
    save_grid_pc = False
    eval_final = True
    if save_grid_pc:
        grid_pc_file = "../../results/"+"grid_sdf.npy"
    if save:
        # 把config文件写入文件夹
        with open(save_path + "/config.json", "w") as outfile:
            json.dump(lgsdf_trainer.config, outfile, indent=4)
        # 如果需要保存模型，见了checkpoints子文件夹
        if lgsdf_trainer.save_checkpoints:
            checkpoint_path = os.path.join(save_path, "checkpoints")
            os.makedirs(checkpoint_path)
        # 如果需要保存切片，建立slices子文件夹
        if lgsdf_trainer.save_slices:
            slice_path = os.path.join(save_path, 'slices')
            os.makedirs(slice_path)
            # lgsdf_trainer.write_slices(
            #     slice_path, prefix="0.000_", include_gt=True)
        # 如果需要保存mesh，建立mesh子文件夹
        if lgsdf_trainer.save_meshes:
            mesh_path = os.path.join(save_path, 'meshes')
            os.makedirs(mesh_path)

        writer = None
        if use_tensorboard:
            writer = SummaryWriter(save_path)
            
    if save_final:
        if save:
            save_final_path = save_path
            slice_final_path = slice_path
            mesh_final_path = mesh_path
        else:
            now = datetime.now()
            time_str = now.strftime("%m-%d-%y_%H-%M-%S")
            save_final_path = "../../results/" + time_str
            slice_final_path = os.path.join(save_final_path, 'slices')
            os.makedirs(slice_final_path)
            mesh_final_path = os.path.join(save_final_path, 'meshes')
            os.makedirs(mesh_final_path)

    # eval init--------------------------------------------------------------
    # 如果要进行评估的话，设置好评估内容res
    if lgsdf_trainer.do_eval:
        res = {}
        if lgsdf_trainer.sdf_eval:
            res['sdf_eval'] = {}
        if lgsdf_trainer.mesh_eval:
            res['mesh_eval'] = {}
    # 这个是对voxblox的评估
    if lgsdf_trainer.do_vox_comparison:
        vox_res = {}

    last_eval = 0

    # main  loop---------------------------------------------------------------
    # 主循环部分，最多训练20000步，一般到不了
    print("Starting training for max", lgsdf_trainer.n_steps, "steps...")
    # 通过看轨迹长度，知道图像多少
    size_dataset = len(lgsdf_trainer.scene_dataset)

    break_at = -1

    for t in range(lgsdf_trainer.n_steps):
        # break at end -------------------------------------------------------
        # 如果到了最后阶段
        if t == break_at and len(lgsdf_trainer.eval_times) == 0:
            if save_final:
                if lgsdf_trainer.save_slices:
                    # 把当前slices保存下来
                    lgsdf_trainer.write_slices(slice_final_path, prefix="",
                        include_gt=False, include_diff=False,
                        include_chomp=False, draw_cams=False)
                if lgsdf_trainer.save_meshes:
                    lgsdf_trainer.write_mesh(mesh_final_path + "/mesh.ply", save_local =False)
                if lgsdf_trainer.do_eval:
                    # 把当前评估结果保存下来
                    kf_list = lgsdf_trainer.frames.frame_id[:-1].tolist()
                    res['kf_indices'] = kf_list
                    with open(os.path.join(save_path, 'res.json'), 'w') as f:
                        json.dump(res, f, indent=4)
            if eval_final:
                visible_res = lgsdf_trainer.eval_sdf(samples=1000000, visible_region=True)
                print("Time -------------------------------------------------------------------------------------------->", lgsdf_trainer.tot_step_time)
                print("Visible region SDF error: {:.4f}".format(visible_res["av_l1"]))
                print("Visible region SDF error using surf dis: {:.4f}".format(visible_res["surf_l1"]))
                print("Visible region Bins error: ", visible_res["binned_l1"])
                print("Visible region Chomp error: ", visible_res["l1_chomp_costs"])
                acc, comp = lgsdf_trainer.eval_mesh()
                print("Mesh accuracy and completion:", acc, comp)
                print("have saved to", save_path)

            if save_grid_pc:
                sdf = lgsdf_trainer.get_sdf_grid().cpu().numpy()
                np.save(grid_pc_file,sdf)

            break

        # get/add data---------------------------------------------------------
        # 判断是否迭代结束
        finish_optim =  lgsdf_trainer.steps_since_frame == lgsdf_trainer.optim_frames
        # 如果是增量式，并且需要优化或第一帧
        if incremental and (finish_optim or t == 0):
            # After n steps with new frame, check whether to add it to kf set.
            # 使用新帧执行n步后，检查是否将其添加到关键帧集合。
            if t == 0:
                # 如果是第一帧，一定是关键帧
                add_new_frame = True
            else:
                # 如果上一帧是关键帧，设置迭代次数再进行，否则加进来一个新的帧
                # add_new_frame = lgsdf_trainer.check_keyframe_latest()
                add_new_frame = True
            # 如果需要加入进来一个新的关键帧
            if add_new_frame:
                # 如果已经达到关键帧指标，要加入最新的当前时刻的帧（太模拟实时了吧）
                new_frame_id = lgsdf_trainer.get_latest_frame_id()
                if new_frame_id >= size_dataset:
                    # 如果这个最新的已经超过数据个数了，就是结束了,此时再迭代400次停止
                    break_at = t + 600
                    print("**************************************",
                          "End of sequence",
                          "**************************************")
                else:
                    # 如果还没有结束，说明当前帧所处位置，和帧的id，其实是有关联的，差一个帧率作为倍数
                    # print("Total step time", lgsdf_trainer.tot_step_time)
                    print("frame______________________", new_frame_id)
                    # 得到当前帧的数据，一个数据类型
                    frame_data = lgsdf_trainer.get_data([new_frame_id])
                    # 在训练器中添加当前的数据
                    lgsdf_trainer.add_frame(frame_data)
                    if t == 0:
                        # 如果是第一帧，需要优化200次
                        lgsdf_trainer.last_is_keyframe = True
                        lgsdf_trainer.optim_frames = 200
        # optimisation step---------------------------------------------
        # 优化步骤，根据loss优化一次网络参数
        losses, step_time = lgsdf_trainer.step(t=t)
        # 返回状态
        status = [k + ': {:.6f}  '.format(losses[k]) for k in losses.keys()]
        status = "".join(status) + '-- Step time: {:.2f}  '.format(step_time)
        loss = losses['total_loss']
        print(t, status)
        # writer is none，所以不进去
        if save and writer is not None:
            for key in losses.keys():
                writer.add_scalar("losses/{key}", losses[key], t)

        # visualisation----------------------------------------------------------
        # 可视化部分
        # 如果需要更新图像
        if if_vis:
            if update_im_freq is not None and (t % update_im_freq == 0):
                # 如果t达到40倍数
                display = {}
                # 更新需要展示的图像，是所有关键帧，和最新的一帧
                lgsdf_trainer.update_vis_vars()
                # keyframes的数据，是个大数组，直接可以可视化
                display["keyframes"] = lgsdf_trainer.frames_vis()
                # slices的数据，是个大数组，可以直接可视化
                display["slices"] = lgsdf_trainer.slices_vis()
                if show_obj:
                    # 就不展示了，不然显存不够用
                    obj_slices_viz = lgsdf_trainer.obj_slices_vis()
                # 如果需要更新mesh，所以update_mesh_freq一定要是update_im_freq的倍数
                if update_mesh_freq is not None and (t % update_mesh_freq == 0):
                    # 绘制mesh
                    # 返回得到scene
                    scene = lgsdf_trainer.draw_3D(show_pc=False, show_grid_pc = False, show_mesh = t > 200, draw_cameras= t <= 200, camera_view=False, show_gt_mesh=False)
                    # scene = lgsdf_trainer.draw_3D(show_pc=False, show_grid_pc = False, show_mesh = False, draw_cameras= True, camera_view=False, show_gt_mesh=False)
                    # grid_scene = lgsdf_trainer.draw_local_3D(show_pc=False, show_mesh =  t > 200, draw_cameras= t <= 200, camera_view=False, show_gt_mesh=False)
                    if show_obj:
                        try:
                            obj_scene = lgsdf_trainer.draw_obj_3D()
                        except:
                            print('Failed to draw mesh')
                display["scene"] = scene
                # display["local_scene"] = grid_scene
                if show_obj and obj_scene is not None:
                    display["obj_scene"] = obj_scene
                if show_obj and obj_slices_viz is not None:
                    display["obj_slices"] = obj_slices_viz
                # 返回一次当前scene，用于可视化，但循环会继续
                yield display

        # save ----------------------------------------------------------------
        # 如果要保存的话，就不用了,save_perio保存一次
        # if save and len(lgsdf_trainer.save_times) > 0:
        #     if lgsdf_trainer.tot_step_time > lgsdf_trainer.save_times[0]:
        #         save_t = f"{lgsdf_trainer.save_times.pop(0):.3f}"
        #         print(
        #             f"Saving at {save_t}s",
        #             f" --  model {lgsdf_trainer.save_checkpoints} ",
        #             f"slices {lgsdf_trainer.save_slices} ",
        #             f"mesh {lgsdf_trainer.save_meshes} "
        #         )
        #         if lgsdf_trainer.save_checkpoints:
        #             torch.save(
        #                 {
        #                     "step": t,
        #                     "model_state_dict":
        #                         lgsdf_trainer.sdf_map.state_dict(),
        #                     "optimizer_state_dict":
        #                         lgsdf_trainer.optimiser.state_dict(),
        #                     "loss": loss.item(),
        #                 },
        #                 os.path.join(
        #                     checkpoint_path, "step_" + save_t + ".pth")
        #             )
        #         if lgsdf_trainer.save_slices:
        #             lgsdf_trainer.write_slices(
        #                 slice_path, prefix=save_t + "_",
        #                 include_gt=False, include_diff=False,
        #                 include_chomp=False, draw_cams=True)
        #         if lgsdf_trainer.save_meshes:
        #             lgsdf_trainer.write_mesh(mesh_path + f"/{save_t}.ply")

        # evaluation -----------------------------------------------------
        # 如果评估vox的话，评估很多东西的话
        if len(lgsdf_trainer.eval_times) > 0:
            # 如果达到需要评估的时间点，即voxblox在这个时候评估了
            if lgsdf_trainer.tot_step_time > lgsdf_trainer.eval_times[0]:
                eval_t = lgsdf_trainer.eval_times[0]
                # 得到当前时刻的当前网络的评估结果
                print("voxblox eval at ----------------------------->", eval_t)
                vox_res[lgsdf_trainer.tot_step_time] = lgsdf_trainer.eval_fixed()
                if save:
                    with open(os.path.join(save_path, 'vox_res.json'), 'w') as f:
                        json.dump(vox_res, f, indent=4)
        # 如果达到评估时间了
        elapsed_eval = lgsdf_trainer.tot_step_time - last_eval
        if lgsdf_trainer.do_eval and elapsed_eval > lgsdf_trainer.eval_freq_s:
            last_eval = lgsdf_trainer.tot_step_time - lgsdf_trainer.tot_step_time % lgsdf_trainer.eval_freq_s
            # 如果要进行sdf的评估
            if lgsdf_trainer.sdf_eval and lgsdf_trainer.gt_sdf_file is not None:
                visible_res = lgsdf_trainer.eval_sdf(samples=1000000, visible_region=True)
                # obj_errors = lgsdf_trainer.eval_object_sdf()
                print("Time -------------------------------------------------------------------------------------------->", lgsdf_trainer.tot_step_time)
                print("Visible region SDF error: {:.4f}".format(visible_res["av_l1"]))
                print("Visible region SDF error using surf dis: {:.4f}".format(visible_res["surf_l1"]))
                print("Visible region Bins error: ", visible_res["binned_l1"])
                print("Visible region Chomp error: ", visible_res["l1_chomp_costs"])
                # print("Objects SDF error: ", obj_errors)
                # 如果不是增量式，就需要评估整个场景的
                if not incremental:
                    full_vol_res = lgsdf_trainer.eval_sdf(visible_region=True)
                    print("Full region SDF error: {:.4f}".format(full_vol_res["av_l1"]))
                if save:
                    res['sdf_eval'][t] = {
                        'time': lgsdf_trainer.tot_step_time,
                        'rays': visible_res,
                    }
                    # 我注释掉的
                    # if obj_errors is not None:
                    #     res['sdf_eval'][t]['objects_l1'] = obj_errors
            if lgsdf_trainer.mesh_eval:
                acc, comp = lgsdf_trainer.eval_mesh()
                print("Mesh accuracy and completion:", acc, comp)
                if save:
                    res['mesh_eval'][t] = {
                        'time': lgsdf_trainer.tot_step_time,
                        'acc': acc,
                        'comp': comp,
                    }
            if save:
                with open(os.path.join(save_path, 'res.json'), 'w') as f:
                    json.dump(res, f, indent=4)
                if writer is not None:
                    writer.add_scalar(
                        "sdf_error_visible/average", visible_res["av_l1"], t)


# 主函数，最开始运行的地方
if __name__ == "__main__":
    # 如果有gpu的话，使用cuda
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 表示第一堆随机数，桌
    seed = 1
    np.random.seed(seed)
    torch.manual_seed(seed)
    # python用于调取参数的解释器
    parser = argparse.ArgumentParser(description="iSDF.")
    # json配置文件
    parser.add_argument("--config", type=str, help="input json config")
    # 是否增量式，这肯定是呀，所以不加
    parser.add_argument(
        "-ni",
        "--no_incremental",
        action="store_false",
        help="disable incremental SLAM option",
    )
    # 是否是一个无头序列，是多个场景使用的，所以不加
    parser.add_argument(
        "-hd", "--headless",
        action="store_true",
        help="run headless (i.e. no visualisations)"
    )
    args = parser.parse_args()

    config_file = args.config
    headless = args.headless
    incremental = args.no_incremental
    # 加载chkpt，也就是网络
    chkpt_load_file = None

    # vis，可视化使用的，多久更新一次
    show_obj = False
    update_im_freq = 80 #40
    update_mesh_freq = 250 #200
    if headless:
        update_im_freq = None
        update_mesh_freq = None

    # save，保存用的
    save = True #False
    use_tensorboard = False # False
    if save:
        # 如果需要保存，会创建一个文件夹
        now = datetime.now()
        time_str = now.strftime("%m-%d-%y_%H-%M-%S")
        save_path = "../../results/" + time_str
        os.mkdir(save_path)
    else:
        save_path = None

    scenes = train(
        device,
        config_file,
        chkpt_load_file=chkpt_load_file,
        incremental=incremental,
        # vis
        if_vis = False,
        show_obj=show_obj,
        update_im_freq=update_im_freq,
        update_mesh_freq=update_mesh_freq,
        # save
        save_path=save_path,
        use_tensorboard=use_tensorboard,
    )

    
    if headless:
        # 如果是多个场景，不用可视化，继续下一个
        on = True
        while on:
            try:
                out = next(scenes)
            except StopIteration:
                on = False
    else:
        # 如果是一个场景
        n_cols = 2
        if show_obj:
            n_cols = 3
        # 可视化是一个2 x 2的opencv窗口，展示当前场景，但主循环还在继续
        tiling = (2, n_cols)
        visualisation.display.display_scenes(scenes, height=int(680 * 0.7), width=int(1200 * 0.7), tile=tiling
        )
