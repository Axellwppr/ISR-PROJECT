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
from link_trans import smpl_joint_pick, new_robot_joint_pick

if __name__ == "__main__":
    device = torch.device("cpu")
    
    smpl_parser_n = SMPL_Parser(model_path="/home/axell/desktop/11-30-ISR-PROJ/human2humanoid/data/smpl", gender="neutral")
    smpl_parser_n.to(device)

    shape_new, scale = joblib.load("/home/axell/desktop/11-30-ISR-PROJ/human2humanoid/data/new_robot/shape_optimized.pkl")
    shape_new = shape_new.to(device)
    scale = scale.to(device)
    
    smpl_joint_pick_idx = [SMPL_BONE_ORDER_NAMES.index(j) for j in smpl_joint_pick]
    
    smpl_joint_shoulder = [
        SMPL_BONE_ORDER_NAMES.index("L_Shoulder"),
        SMPL_BONE_ORDER_NAMES.index("R_Shoulder")
    ]
    amass_data = joblib.load("./data/train_diffusion_manip_seq_joints24.p")

    pbar = tqdm(amass_data.keys())
    
    data_dump = {}
    
    # breakpoint()
    
    for data_key in pbar:
        data = amass_data[data_key]

        skip = 1
        root_trans = data["trans"]
        # ofst = root_trans[0:1, :].copy()
        # ofst[:, 2] = 0
        # root_trans -= ofst
        # ofst = data["trans2joint"][None,:]
        # root_trans -= ofst
        root_rot = data["root_orient"]
        pose_body = data["pose_body"]
        
        pose_aa = np.concatenate((root_rot, pose_body, np.zeros((pose_body.shape[0], 6))), axis=-1)
        trans = torch.from_numpy(root_trans).float().to(device)

        N = trans.shape[0]
        pose_aa_walk = (
            torch.from_numpy(
                pose_aa
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
        
        target_smpl_pos = scaled_joints_opt.detach().cpu().numpy()
        target_smpl_pos = np.einsum('tij,tnj->tni', root_rot_quat.inv().as_matrix(), target_smpl_pos)
        
        target_smpl_shoulder = target_smpl_pos[:, smpl_joint_shoulder]
        target_smpl_pos = target_smpl_pos[: , smpl_joint_pick_idx] #+ root_pos_opt.detach().cpu().numpy()
        
        shoulder_offset = target_smpl_shoulder.mean(axis=1)[:, None, :] - np.array([[[0, 0, 0.2885]]])
        # breakpoint()
        target_smpl_pos[:, 4:, :] -= shoulder_offset
        
        
        # plot_dynamic_points(target_smpl_pos, target_smpl_pos)
        
        data_dump[data["seq_name"]] = {
            "smpl_pos": target_smpl_pos,
            "root_trans": root_pos_opt.squeeze(1).detach().cpu().detach().numpy(),
            "root_rot": root_rot_quat.as_quat(),
            "fps": 30,
        }
        
    joblib.dump(data_dump,"./gen_pos.pkl")