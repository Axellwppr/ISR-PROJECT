import glob
import os
import sys
import os.path as osp

sys.path.append(os.getcwd())

from scipy.spatial.transform import Rotation as sRot
import numpy as np
import torch
from torch.autograd import Variable
from tqdm import tqdm
import argparse
import joblib

# 假设你已有SMPL_Parser，可从SMPL姿态与形状参数获取关节位置
from smpl_sim.smpllib.smpl_parser import SMPL_Parser, SMPL_BONE_ORDER_NAMES

# 引入pytorch_kinematics
import pytorch_kinematics as pk

# 导入Matplotlib模块
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from utils import compute_rotation_matrix_batch, compute_rotation_matrix


# =========== 函数定义 =========== #
def load_amass_data(data_path):
    entry_data = dict(np.load(open(data_path, "rb"), allow_pickle=True))
    if "mocap_framerate" not in entry_data:
        return None
    framerate = entry_data["mocap_framerate"]

    root_trans = entry_data["trans"]
    pose_aa = np.concatenate(
        [entry_data["poses"][:, :66], np.zeros((root_trans.shape[0], 6))], axis=-1
    )
    betas = entry_data["betas"]
    gender = entry_data["gender"]
    # breakpoint()
    return {
        "pose_aa": pose_aa,
        "gender": gender,
        "trans": root_trans,
        "betas": betas,
        "fps": framerate,
    }


def plot_frame(robot_pos, smpl_pos, iteration, frame_idx, show_vec):
    """
    绘制机器人和SMPL关节位置的3D图，并阻塞优化过程直到窗口关闭。
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # 绘制机器人关节
    ax.scatter(
        robot_pos[:, 0], robot_pos[:, 1], robot_pos[:, 2], c="r", label="Robot Joints"
    )

    # 绘制SMPL关节
    ax.scatter(
        smpl_pos[:, 0], smpl_pos[:, 1], smpl_pos[:, 2], c="b", label="SMPL Joints"
    )

    # 绘制向量
    ax.quiver(
        robot_pos[0, 0],
        robot_pos[0, 1],
        robot_pos[0, 2],
        show_vec[0],
        show_vec[1],
        show_vec[2],
        color="g",
        label="Y Axis",
    )

    # 设置坐标轴标签和标题
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"Iteration {iteration} - Frame {frame_idx}")

    # 添加图例
    ax.legend()

    # 设置坐标轴范围（根据你的数据调整）
    all_points = np.concatenate((robot_pos, smpl_pos), axis=0)
    max_range = (
        np.array(
            [
                all_points[:, 0].max() - all_points[:, 0].min(),
                all_points[:, 1].max() - all_points[:, 1].min(),
                all_points[:, 2].max() - all_points[:, 2].min(),
            ]
        ).max()
        / 2.0
    )

    mid_x = (all_points[:, 0].max() + all_points[:, 0].min()) * 0.5
    mid_y = (all_points[:, 1].max() + all_points[:, 1].min()) * 0.5
    mid_z = (all_points[:, 2].max() + all_points[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # 显示图形并阻塞
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--amass_root", type=str, default="data/AMASS")
    parser.add_argument(
        "--robot_urdf",
        type=str,
        default="/home/axell/desktop/humanoid/humanoid-benchmark-main/isaacLab/manipulation/assets/urdf/rel2/urdf/rel2.urdf",
    )  # 请修改为你的URDF路径
    args = parser.parse_args()

    device = torch.device("cpu")

    # rotate_xbot = torch.tensor(
    #     [[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=torch.float32, device=device
    # )

    # =========== 加载SMPL解析器 ===========
    smpl_parser_n = SMPL_Parser(model_path="data/smpl", gender="neutral")
    smpl_parser_n.to(device)

    # =========== 加载之前拟合好的新机器人形状和比例 ===========
    shape_new, scale = joblib.load("data/new_robot/shape_optimized.pkl")
    shape_new = shape_new.to(device)
    scale = scale.to(device)

    # =========== 加载URDF并建立运动链 ===========
    with open(args.robot_urdf, "rb") as f:
        urdf_str = f.read()
    chain = pk.build_chain_from_urdf(urdf_str)

    # 获得URDF中的joint和link信息
    all_link_names = chain.get_link_names()
    all_joint_names = chain.get_joint_parameter_names()
    print(all_joint_names)
    # breakpoint()
    chain.print_tree()

    rotate_axis = []
    joint_data = chain.get_joints()

    for idx, joint in enumerate(joint_data):
        if joint.joint_type == "revolute":
            rotate_axis.append(joint.axis)
        else:
            print(joint.joint_type)

    rotate_axis = torch.stack(rotate_axis, dim=0)

    print(rotate_axis)

    # =========== 定义关节映射 ===========
    from link_trans import smpl_joint_pick, new_robot_joint_pick, joint_enable

    smpl_joint_pick_idx = [SMPL_BONE_ORDER_NAMES.index(j) for j in smpl_joint_pick]
    new_robot_joint_pick_idx = [all_link_names.index(ln) for ln in new_robot_joint_pick]
    joint_enable_idx = [all_joint_names.index(ln) for ln in joint_enable]

    amass_root = args.amass_root
    all_pkls = glob.glob(f"{amass_root}/**/*.npz", recursive=True)
    split_len = len(amass_root.split("/"))
    key_name_to_pkls = {
        "0-" + "_".join(data_path.split("/")[split_len:]).replace(".npz", ""): data_path
        for data_path in all_pkls
    }

    if len(key_name_to_pkls) == 0:
        raise ValueError(f"No motion files found in {amass_root}")

    data_dump = {}
    # pbar = tqdm(key_name_to_pkls.keys())

    filter_keys = joblib.load(
        "/home/axell/desktop/11-30-ISR-PROJ/human2humanoid/resources/motions/h1/amass_phc_filtered.pkl"
    ).keys()
    pbar = tqdm(filter_keys)

    # 假设你要绘制的帧索引为10
    FRAME_INDEX_TO_PLOT = 20

    count = 0

    for data_key in pbar:
        if data_key not in key_name_to_pkls:
            print("Not found: ", data_key)
            continue
        amass_data = load_amass_data(key_name_to_pkls[data_key])
        if amass_data is None:
            continue

        count += 1
        if count > 20:
            break

        skip = int(amass_data["fps"] // 30) if amass_data["fps"] >= 30 else 1
        trans = torch.from_numpy(amass_data["trans"][::skip]).float().to(device)
        # print(trans)
        N = trans.shape[0]
        pose_aa_walk = (
            torch.from_numpy(
                np.concatenate(
                    (amass_data["pose_aa"][::skip, :66], np.zeros((N, 6))), axis=-1
                )
            )
            .float()
            .to(device)
        )
        # breakpoint()
        # 从SMPL获取关节位置
        verts, joints = smpl_parser_n.get_joints_verts(
            pose_aa_walk, torch.zeros((1, 10)).to(device), trans
        )
        # 计算offset，将SMPL根部平移对齐机器人参考
        offset = joints[:, 0] - trans
        root_trans_offset = trans + offset

        gt_root_rot = pose_aa_walk[:, :3]  # (N,3)
        root_rot_mats_np = (
            sRot.from_rotvec(gt_root_rot.cpu().numpy())
            * sRot.from_matrix(np.array([[[0, 0, 1], [1, 0, 0], [0, 1, 0]]])).inv()
        ).as_matrix()  # (N,3,3)
        root_rot_mats = torch.from_numpy(root_rot_mats_np).float().to(device)  # (N,3,3)

        ### 初始化
        M = len(joint_enable)
        dof_pos_new = Variable(
            torch.zeros((1, N, M), dtype=torch.float32, device=device),
            requires_grad=True,
        )

        optimizer_pose = torch.optim.Adadelta([dof_pos_new], lr=60)

        ### 计算目标形状
        # 使用之前计算的形状参数
        verts_opt, joints_opt = smpl_parser_n.get_joints_verts(
            pose_aa_walk, shape_new, trans
        )
        # 将SMPL关节缩放
        root_pos_opt = joints_opt[:, 0].unsqueeze(1)
        scaled_joints_opt = (joints_opt - root_pos_opt) * scale + root_pos_opt

        # 定义目标SMPL关节位置
        target_smpl_pos = scaled_joints_opt[
            :, smpl_joint_pick_idx
        ]  # Shape: (N, len, 3)
        y_vec = target_smpl_pos[:, 5, :] - target_smpl_pos[:, 8, :]  # 左右肩膀定义的Y轴
        x_vec = target_smpl_pos[:, -1, :] - target_smpl_pos[:, 0, :]

        diff_max = float("inf")
        best_loss = float("inf")
        best_dof = None
        for iteration in range(500):
            joint_value = torch.zeros((N, len(all_joint_names)), dtype=torch.float32)
            joint_value[:, joint_enable_idx] = dof_pos_new[0]  # Shape: (N, M)
            # 批量前向计算机器人FK
            transforms = chain.forward_kinematics(joint_value)

            robot_positions = torch.stack(
                [
                    transforms[all_link_names[idx]].get_matrix()[:, :3, 3]
                    for idx in new_robot_joint_pick_idx
                ],
                dim=1,
            )  # Shape: (N, len(new_robot_joint_pick_idx), 3)
            robot_positions = (robot_positions - robot_positions[:, 0:1, :]).transpose(
                1, 2
            )
            # waist rot
            waist_rot = transforms["waist_yaw_link"].get_matrix()[:, :3, :3]
            rel_rot = torch.matmul(root_rot_mats, torch.linalg.inv(waist_rot))
            # rel_rot = torch.matmul(head_rot, rel_rot)
            robot_positions_world = torch.matmul(rel_rot, robot_positions).transpose(
                1, 2
            )

            y_vec_xbot = (
                robot_positions_world[:, 5, :] - robot_positions_world[:, 8, :]
            )  # 左右肩膀定义的Y轴
            x_vec_xbot = (
                robot_positions_world[:, -1, :] - robot_positions_world[:, 0, :]
            )

            head_rot = compute_rotation_matrix(x_vec_xbot, y_vec_xbot, x_vec, y_vec)

            # breakpoint()
            rel_rot = torch.matmul(head_rot, root_rot_mats)
            robot_positions_world = (
                torch.matmul(root_rot_mats, robot_positions).transpose(1, 2)
                + root_pos_opt
            )

            # 计算差异
            diff = robot_positions_world - target_smpl_pos

            # 计算损失
            loss = diff.norm(dim=-1).mean()
            if loss.item() < best_loss:
                best_dof = dof_pos_new.clone()
                diff_max = diff.norm(dim=-1).max().item()
                best_loss = loss.item()

            pbar.set_description_str(
                f"Iteration {iteration} Loss: {loss.item()*1000:.4f}, Max Diff: {diff_max*1000:.4f}"
            )

            # 反向传播和优化
            optimizer_pose.zero_grad()
            loss.backward()
            optimizer_pose.step()

            # 每20次迭代绘制一次图形，并暂停优化直到窗口关闭
            if iteration % 100 == 0:
                if N > FRAME_INDEX_TO_PLOT:
                    frame_idx = FRAME_INDEX_TO_PLOT

                    robot_pos_frame = (
                        robot_positions_world[frame_idx].detach().cpu().numpy()
                    )  # (len, 3)
                    smpl_pos_frame = (
                        target_smpl_pos[frame_idx].detach().cpu().numpy()
                    )  # (len, 3)

                    print(f"Iteration {iteration}: Plotting frame {frame_idx}")

                    # 绘制并阻塞
                    plot_frame(
                        robot_pos_frame,
                        smpl_pos_frame,
                        iteration,
                        frame_idx,
                        x_vec[FRAME_INDEX_TO_PLOT].detach().cpu().numpy(),
                    )

        if diff_max > 0.3:
            print(f"Max Diff: {diff_max}, Skip {data_key}")
            continue

        if best_loss > 0.08:
            print(f"Loss: {best_loss}, Skip {data_key}")
            continue

        # 优化结束后，保存数据
        root_rot_quat = sRot.from_matrix(rel_rot.detach().cpu().numpy()).as_quat()

        root_trans_offset_dump = root_pos_opt.squeeze(1).clone()
        z_min_robot = robot_positions_world[:, :, 2].min().item()
        root_trans_offset_dump[..., 2] -= z_min_robot - 0.06

        best_joint_value = torch.zeros((N, len(all_joint_names)), dtype=torch.float32)
        best_joint_value[:, joint_enable_idx] = dof_pos_new[0]

        # breakpoint()
        best_pose_aa = torch.cat(
            [
                torch.from_numpy(
                    sRot.from_matrix(root_rot_mats.detach().cpu().numpy()).as_rotvec()
                ).unsqueeze(1),
                rotate_axis.unsqueeze(0) * best_joint_value.unsqueeze(-1),
            ],
            axis=1,
        )
        # breakpoint()
        data_dump[data_key] = {
            "root_trans_offset": root_trans_offset_dump.cpu().detach().numpy(),
            "dof": best_joint_value.detach().cpu().numpy(),
            "pose_aa": best_pose_aa.cpu().detach().numpy(),
            "root_rot": root_rot_quat,
            "fps": 30,
        }

        # print(
        #     f"dumping {data_key} for testing, remove the line if you want to process all data"
        # )
        # joblib.dump(data_dump, "data/new_robot/test.pkl")
        # 移除调试断点
        # import ipdb
        # ipdb.set_trace()

    # 移除调试断点
    # import ipdb
    # ipdb.set_trace()
    joblib.dump(data_dump, "data/new_robot/amass_all.pkl")
