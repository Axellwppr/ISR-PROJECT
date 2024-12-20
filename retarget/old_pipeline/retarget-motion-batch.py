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

from smpl_sim.smpllib.smpl_parser import SMPL_Parser, SMPL_BONE_ORDER_NAMES
import pytorch_kinematics as pk
from utils import (
    compute_rotation_matrix_batch,
    compute_rotation_matrix,
    plot_dynamic_points,
)


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


def process_chunk(
    segments,
    all_trans,
    all_pose_aa,
    smpl_parser_n,
    shape_new,
    scale,
    chain,
    smpl_joint_pick_idx,
    rotate_axis,
    joint_enable_idx,
    all_link_names,
    new_robot_joint_pick_idx,
    root_rot_inv_mat,
    all_joint_names,
    max_diff_threshold=0.5,
    loss_threshold=0.08,
    device=torch.device("cpu"),
    joint_min=None,
    joint_max=None,
    pbar=None,
):
    """
    对已经拼合的一整段序列进行优化和输出结果。现在在优化过程中就缓存最好结果。
    """

    data_dump = {}
    if len(segments) == 0 or all_trans.shape[0] == 0:
        return data_dump

    T = all_trans.shape[0]
    trans_t = torch.from_numpy(all_trans).float().to(device)
    pose_aa_t = torch.from_numpy(all_pose_aa).float().to(device)

    verts_opt, joints_opt = smpl_parser_n.get_joints_verts(
        pose_aa_t, shape_new, trans_t
    )
    root_pos_opt = joints_opt[:, 0:1, :]
    scaled_joints_opt = (joints_opt - root_pos_opt) * scale + root_pos_opt
    target_smpl_pos = scaled_joints_opt[:, smpl_joint_pick_idx, :]

    gt_root_rot = pose_aa_t[:, :3]
    root_rot_mats_np = sRot.from_rotvec(gt_root_rot.cpu().numpy())
    inv_convert = sRot.from_matrix(root_rot_inv_mat).inv()
    root_rot_mats_np = (root_rot_mats_np * inv_convert).as_matrix()
    root_rot_mats = torch.from_numpy(root_rot_mats_np).float().to(device)

    M = len(joint_enable_idx)
    dof_pos_new = Variable(
        torch.zeros((T, M), dtype=torch.float32, device=device), requires_grad=True
    )
    optimizer_pose = torch.optim.Adadelta([dof_pos_new], lr=100)

    max_iters = 800
    best_loss = float("inf")
    best_dof = None
    best_robot_positions_world = None
    best_diff = None
    best_head_rot = None

    for iteration in range(max_iters):
        # 前向计算
        joint_value = torch.zeros(
            (T, len(all_joint_names)), dtype=torch.float32, device=device
        )
        joint_value[:, joint_enable_idx] = dof_pos_new
        transforms = chain.forward_kinematics(joint_value)
        robot_positions = torch.stack(
            [
                transforms[all_link_names[idx]].get_matrix()[:, :3, 3]
                for idx in new_robot_joint_pick_idx
            ],
            dim=1,
        )

        robot_positions_rel = robot_positions - robot_positions[:, 0:1, :]
        waist_rot = transforms["waist_yaw_link"].get_matrix()[:, :3, :3]
        rel_rot = torch.matmul(root_rot_mats, torch.linalg.inv(waist_rot))

        robot_positions_world = (
            torch.matmul(rel_rot, robot_positions_rel.transpose(-1, -2)).transpose(
                -1, -2
            )
            + root_pos_opt
        )

        # y_vec = target_smpl_pos[:, 5, :] - target_smpl_pos[:, 8, :]
        # x_vec = target_smpl_pos[:, -1, :] - target_smpl_pos[:, 0, :]
        # y_vec_xbot = robot_positions_world[:, 5, :] - robot_positions_world[:, 8, :]
        # x_vec_xbot = robot_positions_world[:, -1, :] - robot_positions_world[:, 0, :]

        # head_rot = compute_rotation_matrix(x_vec_xbot, y_vec_xbot, x_vec, y_vec)
        # rel_rot = torch.matmul(head_rot, rel_rot)

        # robot_positions_world = (
        #     torch.matmul(rel_rot, robot_positions_rel.transpose(-1, -2)).transpose(
        #         -1, -2
        #     )
        #     + root_pos_opt
        # )

        diff = robot_positions_world - target_smpl_pos
        loss = diff.norm(dim=-1).mean()

        # 优化步骤
        optimizer_pose.zero_grad()
        loss.backward()
        optimizer_pose.step()

        # 关节限制：如果有joint_min和joint_max则clamp
        if joint_min is not None and joint_max is not None:
            with torch.no_grad():
                dof_pos_new.copy_(torch.clamp(dof_pos_new, joint_min, joint_max))

        if pbar is not None:
            pbar.set_postfix_str(f"iter: {iteration}, loss: {loss.item():.4f}")

        # # 更新最佳结果
        # if loss.item() < best_loss:
        #     best_loss = loss.item()
        #     best_dof = dof_pos_new.detach().clone()
        #     best_robot_positions_world = robot_positions_world.detach().clone()
        #     best_diff = diff.detach().clone()
        #     best_head_rot = head_rot.detach().clone()

    # plot_dynamic_points(
    #     robot_positions_world.detach().cpu().numpy(),
    #     target_smpl_pos.detach().cpu().numpy(),
    # )

    best_dof = dof_pos_new.detach().clone()
    best_robot_positions_world = robot_positions_world.detach().clone()
    best_diff = diff.detach().clone()
    best_head_rot = rel_rot.detach().clone()

    # 分段检查
    for data_key, start_idx, end_idx in segments:
        seg_len = end_idx - start_idx
        if seg_len <= 0:
            continue
        diff_final = best_diff[start_idx:end_idx]  # 直接使用best_diff对应的片段
        seg_loss = diff_final.mean().item()
        seg_max_diff = diff_final.max().item()

        if seg_loss > loss_threshold or seg_max_diff > max_diff_threshold:
            print(data_key, "segment loss too high, skipping")
            continue

        # root_trans_offset
        root_trans_offset_dump = root_pos_opt[start_idx:end_idx, 0].clone()
        z_min_robot = best_robot_positions_world[start_idx:end_idx, :, 2].min().item()
        root_trans_offset_dump[..., 2] -= z_min_robot - 0.06

        seg_root_rot_mats = (
            best_head_rot[start_idx:end_idx].detach().cpu().numpy()
        )  # (seg_len, 3,3)
        seg_root_rotvec = sRot.from_matrix(seg_root_rot_mats).as_rotvec()
        seg_best_joint_val = torch.zeros(
            (T, len(all_joint_names)), dtype=torch.float32, device=device
        )
        seg_best_joint_val[:, joint_enable_idx] = best_dof
        seg_best_joint_val = seg_best_joint_val[start_idx:end_idx, :].cpu().numpy()

        # 生成pose_aa
        seg_pose_aa = np.concatenate(
            [
                seg_root_rotvec[:, None, :],
                (
                    rotate_axis.cpu().numpy()[None, :, :]
                    * seg_best_joint_val[:, :, None]
                ),
            ],
            axis=1,
        )

        root_rot_quat = sRot.from_matrix(seg_root_rot_mats).as_quat()

        # 存储结果
        data_dump[data_key] = {
            "root_trans_offset": root_trans_offset_dump.cpu().numpy(),
            "dof": seg_best_joint_val,
            "pose_aa": seg_pose_aa,
            "root_rot": root_rot_quat,
            "fps": 30,
        }

    return data_dump


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--amass_root", type=str, default="data/AMASS")
    parser.add_argument(
        "--robot_urdf",
        type=str,
        default="/home/axell/desktop/humanoid/humanoid-benchmark-main/isaacLab/manipulation/assets/urdf/rel2/urdf/rel2.urdf",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Use cuda or cpu")
    parser.add_argument(
        "--max_chunk_length", type=int, default=10000, help="Max frames per chunk"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="data/new_robot/amass_all_merged_optimized.pkl",
    )
    parser.add_argument(
        "--incremental_save", type=str, default="data/new_robot/temp_incremental.pkl"
    )
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 加载SMPL与机器人模型
    smpl_parser_n = SMPL_Parser(model_path="data/smpl", gender="neutral").to(device)
    shape_new, scale = joblib.load("data/new_robot/shape_optimized.pkl")
    shape_new = shape_new.to(device)
    scale = scale.to(device)

    with open(args.robot_urdf, "rb") as f:
        urdf_str = f.read()
    chain = pk.build_chain_from_urdf(urdf_str).to(device=device)
    all_link_names = chain.get_link_names()
    all_joint_names = chain.get_joint_parameter_names()

    rotate_axis = []
    joint_data = chain.get_joints()
    for idx, joint in enumerate(joint_data):
        if joint.joint_type == "revolute":
            rotate_axis.append(joint.axis)
    rotate_axis = torch.stack(rotate_axis, dim=0).to(device)

    from link_trans import smpl_joint_pick, new_robot_joint_pick, joint_enable

    smpl_joint_pick_idx = [SMPL_BONE_ORDER_NAMES.index(j) for j in smpl_joint_pick]
    new_robot_joint_pick_idx = [all_link_names.index(ln) for ln in new_robot_joint_pick]
    joint_enable_idx = [all_joint_names.index(ln) for ln in joint_enable]

    # 假设这里定义关节限位joint_min, joint_max （需要根据机器人体模具体填写）
    # 这里示例化为全0和全π，一般需要真实数据
    joint_min = torch.tensor(
        chain.get_joint_limits()[0], dtype=torch.float32, device=device
    )[joint_enable_idx]
    joint_max = torch.tensor(
        chain.get_joint_limits()[1], dtype=torch.float32, device=device
    )[joint_enable_idx]

    amass_root = args.amass_root
    all_pkls = glob.glob(f"{amass_root}/**/*.npz", recursive=True)
    split_len = len(amass_root.split("/"))
    key_name_to_pkls = {
        "0-" + "_".join(data_path.split("/")[split_len:]).replace(".npz", ""): data_path
        for data_path in all_pkls
    }

    if len(key_name_to_pkls) == 0:
        raise ValueError(f"No motion files found in {amass_root}")

    filter_keys = joblib.load(
        "/home/axell/desktop/11-30-ISR-PROJ/human2humanoid/resources/motions/h1/amass_phc_filtered.pkl"
    ).keys()
    valid_keys = [k for k in filter_keys if k in key_name_to_pkls]

    MAX_CHUNK_LENGTH = args.max_chunk_length

    # 定义root_rot_inv_mat用于root旋转处理
    root_rot_inv_mat = np.array([[[0, 0, 1], [1, 0, 0], [0, 1, 0]]])

    data_dump_all = {}
    all_trans_list = []
    all_pose_aa_list = []
    segments = []
    current_start = 0
    current_length = 0

    # 增量保存计数器
    flush_count = 0

    def flush_chunk(pbar=None):
        global segments, all_trans_list, all_pose_aa_list, current_start, current_length, data_dump_all, flush_count
        if len(segments) == 0:
            return
        all_trans_arr = np.concatenate(all_trans_list, axis=0)
        all_pose_aa_arr = np.concatenate(all_pose_aa_list, axis=0)
        chunk_result = process_chunk(
            segments,
            all_trans_arr,
            all_pose_aa_arr,
            smpl_parser_n,
            shape_new,
            scale,
            chain,
            smpl_joint_pick_idx,
            rotate_axis,
            joint_enable_idx,
            all_link_names,
            new_robot_joint_pick_idx,
            root_rot_inv_mat,
            all_joint_names,
            device=device,
            joint_min=joint_min,
            joint_max=joint_max,
            pbar=pbar,
        )
        data_dump_all.update(chunk_result)
        # 清空当前chunk缓存
        segments.clear()
        all_trans_list.clear()
        all_pose_aa_list.clear()
        current_start = 0
        current_length = 0

        # 增量式保存
        flush_count += 1
        if flush_count % 50 == 0:
            incremental_file = args.incremental_save.replace(
                ".pkl", f"_{flush_count}.pkl"
            )
            joblib.dump(data_dump_all, incremental_file)
        # 可选释放GPU缓存
        torch.cuda.empty_cache()

    # 按顺序将序列加入chunk
    pbar = tqdm(valid_keys, desc="Loading and chunking data")
    for data_key in pbar:
        # data_key = "0-KIT_424_parkour08_poses"
        amass_data = load_amass_data(key_name_to_pkls[data_key])
        if amass_data is None:
            continue
        fps = amass_data["fps"]
        skip = int(fps // 30) if fps >= 30 else 1
        trans = amass_data["trans"][::skip]
        pose_aa = np.concatenate(
            (amass_data["pose_aa"][::skip, :66], np.zeros((trans.shape[0], 6))), axis=-1
        )
        length = trans.shape[0]
        if length == 0:
            continue

        # 如果加上本序列会超过MAX_CHUNK_LENGTH，则先处理当前chunk
        if current_length + length > MAX_CHUNK_LENGTH and current_length > 0:
            flush_chunk(pbar)

        start_idx = current_length
        end_idx = start_idx + length
        segments.append((data_key, start_idx, end_idx))
        all_trans_list.append(trans)
        all_pose_aa_list.append(pose_aa)
        current_length += length

    # 最后一批未处理数据
    flush_chunk()

    # 最终保存
    joblib.dump(data_dump_all, args.output_file)
    print(f"All sequences processed and results saved to {args.output_file}")
