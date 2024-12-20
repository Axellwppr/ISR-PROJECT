smpl_joint_pick = [
    "L_Knee",
    "L_Ankle",
    "R_Knee",
    "R_Ankle",
    "L_Elbow",
    "L_Hand",
    "R_Elbow",
    "R_Hand",
]

new_robot_joint_pick = [
    "left_knee_link",
    "left_ankle_pitch_link",
    "right_knee_link",
    "right_ankle_pitch_link",
    "left_arm_yaw_link",
    "left_hand_ee_link",
    "right_arm_yaw_link",
    "right_hand_ee_link",
]

import glob
import os
import sys
import os.path as osp

sys.path.append(os.getcwd())

from scipy.spatial.transform import Rotation as sRot
import numpy as np
import torch
from tqdm import tqdm
import joblib

from smpl_sim.smpllib.smpl_parser import SMPL_Parser, SMPL_BONE_ORDER_NAMES
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
    # breakpoint()
    return {
        "pose_aa": pose_aa,
        "gender": gender,
        "trans": root_trans,
        "betas": betas,
        "fps": framerate,
    }


if __name__ == "__main__":
    device = torch.device("cpu")
    
    smpl_parser_n = SMPL_Parser(model_path="/home/axell/desktop/11-30-ISR-PROJ/human2humanoid/data/smpl", gender="neutral")
    smpl_parser_n.to(device)

    shape_new, scale = joblib.load("/home/axell/desktop/11-30-ISR-PROJ/human2humanoid/data/new_robot/shape_optimized.pkl")
    shape_new = shape_new.to(device)
    scale = scale.to(device)
    
    smpl_joint_pick_idx = [SMPL_BONE_ORDER_NAMES.index(j) for j in smpl_joint_pick]
    
    amass_root = "/home/axell/desktop/11-30-ISR-PROJ/human2humanoid/data/AMASS"
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
    
    
    for data_key in pbar:
        if data_key not in key_name_to_pkls:
            print("Not found: ", data_key)
            continue
        amass_data = load_amass_data(key_name_to_pkls[data_key])
        if amass_data is None:
            continue

        skip = int(amass_data["fps"] // 30) if amass_data["fps"] >= 30 else 1
        trans = torch.from_numpy(amass_data["trans"][::skip]).float().to(device)

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
        
        verts, joints = smpl_parser_n.get_joints_verts(
            pose_aa_walk, torch.zeros((1, 10)).to(device), trans
        )
        # 计算offset，将SMPL根部平移对齐机器人参考
        offset = joints[:, 0] - trans
        root_trans_offset = trans + offset
        
        verts_opt, joints_opt = smpl_parser_n.get_joints_verts(
            pose_aa_walk, shape_new, trans
        )
        
        root_rot_quat = sRot.from_rotvec(pose_aa_walk.cpu().numpy()[:, :3]) * sRot.from_quat([0.5, 0.5, 0.5, 0.5]).inv()
        
        # 将SMPL关节缩放
        root_pos_opt = joints_opt[:, 0].unsqueeze(1)
        scaled_joints_opt = (joints_opt - root_pos_opt) * scale
        
        target_smpl_pos = scaled_joints_opt[
            :, smpl_joint_pick_idx
        ]  # Shape: (N, len, 3)
        
        target_smpl_pos = target_smpl_pos.detach().cpu().numpy()
        
        target_smpl_pos = np.einsum('tij,tnj->tni', root_rot_quat.inv().as_matrix(), target_smpl_pos)
        plot_dynamic_points(target_smpl_pos, target_smpl_pos)
        
        data_dump[data_key] = {
            "smpl_pos": target_smpl_pos,
            "root_trans": root_pos_opt.squeeze(1).detach().cpu().detach().numpy(),
            "root_rot": root_rot_quat.as_quat(),
            "fps": 30,
        }
        
    joblib.dump(data_dump,"./amass_pos.pkl")