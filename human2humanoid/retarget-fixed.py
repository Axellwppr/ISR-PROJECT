import glob
import os
import sys
import os.path as osp
import joblib
import torch
from torch.autograd import Variable
from tqdm import tqdm
import numpy as np
from scipy.spatial.transform import Rotation as sRot
from smpl_sim.smpllib.smpl_parser import SMPL_Parser, SMPL_BONE_ORDER_NAMES
import pytorch_kinematics as pk
from utils import compute_rotation_matrix


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
    return {
        "pose_aa": pose_aa,
        "gender": gender,
        "trans": root_trans,
        "betas": betas,
        "fps": framerate,
    }


if __name__ == "__main__":
    # 加载SMPL解析器
    smpl_parser_n = SMPL_Parser(model_path="data/smpl", gender="neutral")
    smpl_parser_n.to("cpu")

    # 读取机器人形状和比例参数
    shape_new, scale = joblib.load("data/new_robot/shape_optimized.pkl")
    shape_new = shape_new.to("cpu")
    scale = scale.to("cpu")

    # 加载URDF
    robot_urdf = "/home/axell/desktop/humanoid/humanoid-benchmark-main/isaacLab/manipulation/assets/urdf/rel2/urdf/rel2.urdf"  # 替换为实际URDF路径
    chain = pk.build_chain_from_urdf(open(robot_urdf, "rb").read())
    all_link_names = chain.get_link_names()
    all_joint_names = chain.get_joint_parameter_names()

    # 定义关节映射和相关索引
    from link_trans import smpl_joint_pick, new_robot_joint_pick, joint_enable

    smpl_joint_pick_idx = [SMPL_BONE_ORDER_NAMES.index(j) for j in smpl_joint_pick]
    new_robot_joint_pick_idx = [all_link_names.index(ln) for ln in new_robot_joint_pick]
    joint_enable_idx = [all_joint_names.index(ln) for ln in joint_enable]

    # 读取保存的输出数据
    saved_data = joblib.load("data/new_robot/amass_all_merged_optimized.pkl")
    updated_data_dump = {}

    amass_root = "data/AMASS"  # 原始AMASS数据路径
    all_pkls = glob.glob(f"{amass_root}/**/*.npz", recursive=True)
    key_name_to_pkls = {
        "0-"
        + "_".join(data_path.split("/")[len(amass_root.split("/")) :]).replace(
            ".npz", ""
        ): data_path
        for data_path in all_pkls
    }

    count = 0

    for data_key, data in tqdm(saved_data.items(), desc="Recomputing Values"):
        # count += 1
        # if count > 10:
        #     break
        if data_key not in key_name_to_pkls:
            print(f"Data key {data_key} not found in AMASS dataset")
            continue

        # 从AMASS数据集中加载对应数据
        amass_data = load_amass_data(key_name_to_pkls[data_key])
        if amass_data is None:
            print(f"Failed to load AMASS data for key {data_key}")
            continue
        skip = int(amass_data["fps"] // 30) if amass_data["fps"] >= 30 else 1
        trans = torch.from_numpy(amass_data["trans"][::skip]).float()
        # print(trans)
        N = trans.shape[0]
        pose_aa_walk = torch.from_numpy(
            np.concatenate(
                (amass_data["pose_aa"][::skip, :66], np.zeros((N, 6))), axis=-1
            )
        ).float()

        # 从SMPL获取关节位置
        verts, joints = smpl_parser_n.get_joints_verts(
            pose_aa_walk, torch.zeros((1, 10)), trans
        )
        # 计算offset，将SMPL根部平移对齐机器人参考
        offset = joints[:, 0] - trans
        root_trans_offset = trans + offset

        gt_root_rot = pose_aa_walk[:, :3]  # (N,3)
        root_rot_mats_np = (
            sRot.from_rotvec(gt_root_rot.cpu().numpy())
            * sRot.from_quat([0.5, 0.5, 0.5, 0.5]).inv()
        ).as_matrix()  # (N,3,3)
        root_rot_mats = torch.from_numpy(root_rot_mats_np).float()
        # breakpoint()
        # 从保存的文件中获取dof
        dof = torch.as_tensor(data["dof"], dtype=torch.float32)

        # 从SMPL获取关节位置
        verts_opt, joints_opt = smpl_parser_n.get_joints_verts(
            pose_aa_walk, shape_new, trans
        )
        root_pos_opt = joints_opt[:, 0].unsqueeze(1)
        scaled_joints_opt = (joints_opt - root_pos_opt) * scale + root_pos_opt

        # 定义目标SMPL关节位置
        target_smpl_pos = scaled_joints_opt[:, smpl_joint_pick_idx]

        # 前向计算机器人FK
        joint_value = torch.as_tensor(data["dof"], dtype=torch.float32)
        transforms = chain.forward_kinematics(joint_value)

        robot_positions = torch.stack(
            [
                transforms[all_link_names[idx]].get_matrix()[:, :3, 3]
                for idx in new_robot_joint_pick_idx
            ],
            dim=1,
        )  # Shape: (N, len(new_robot_joint_pick_idx), 3)
        robot_positions = (robot_positions - robot_positions[:, 0:1, :]).transpose(1, 2)

        waist_rot = transforms["waist_yaw_link"].get_matrix()[:, :3, :3]
        rel_rot = torch.matmul(root_rot_mats, torch.linalg.inv(waist_rot))

        robot_positions_world = (
            torch.matmul(rel_rot, robot_positions).transpose(1, 2) + root_pos_opt
        )
        # 重新计算偏移和姿态对齐
        root_trans_offset_dump = root_pos_opt.squeeze(1).clone()
        z_min_robot = robot_positions_world[:, :, 2].min().item()
        root_trans_offset_dump[..., 2] -= z_min_robot - 0.06

        root_rot_quat_dump = (
            sRot.from_rotvec(pose_aa_walk.cpu().numpy()[:, :3])
            * sRot.from_quat([0.5, 0.5, 0.5, 0.5]).inv()
        ).as_quat()

        # 保存重新计算的结果
        updated_data_dump[data_key] = {
            "root_trans_offset": root_trans_offset_dump.cpu().detach().numpy(),
            "dof": data["dof"],
            "pose_aa": data["pose_aa"],
            "root_rot": root_rot_quat_dump,
            "fps": 30,
        }

    # 保存新的数据文件
    joblib.dump(updated_data_dump, "data/new_robot/amass_recomputed.pkl")
    print("Recomputed values saved to data/new_robot/amass_recomputed.pkl")
