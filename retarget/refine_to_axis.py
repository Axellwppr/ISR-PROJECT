from tqdm import tqdm  # 修改这行
import joblib

import pytorch_kinematics as pk
from scipy.spatial.transform import Rotation as sRot

from utils import (
    compute_rotation_matrix_batch,
    compute_rotation_matrix,
    plot_dynamic_points,
)

from link_trans import joint_enable, new_robot_joint_pick
import torch
import numpy as np


with open("/home/axell/desktop/11-30-ISR-PROJ/human2humanoid/resources/robots/rel2/urdf/rel2.urdf", "rb") as f:
    urdf_str = f.read()
chain = pk.build_chain_from_urdf(urdf_str)
link_names = chain.get_link_names()
joint_names = chain.get_joint_parameter_names()

joint_enable_idx = [joint_names.index(joint_name) for joint_name in joint_enable]
joint_care_idx = [link_names.index(link_name) for link_name in new_robot_joint_pick]

rotate_axis = []
joint_data = chain.get_joints()
for idx, joint in enumerate(joint_data):
    if joint.joint_type == "revolute":
        rotate_axis.append(joint.axis)
rotate_axis = torch.stack(rotate_axis, dim=0).cpu().numpy()

if __name__ == "__main__":
    data = joblib.load("amass_IK_new.pkl")
    filter_keys = list(data.keys())
    
    pbar = tqdm(filter_keys, desc="merge data")
    
    data_new = {}
    
    for data_key in pbar:
        amass_data = data[data_key]
        
        dof = torch.from_numpy(amass_data["ik_refined"]).float()
        joint_value = torch.zeros((dof.shape[0], len(joint_names)))
        
        joint_value[:, joint_enable_idx] = dof
        
        transforms = chain.forward_kinematics(joint_value)
        robot_positions = torch.stack(
            [transforms[link_names[idx]].get_matrix()[:, :3, 3] for idx in joint_care_idx],
            dim=1,
        ).detach().cpu().numpy()
        
        root_rot = sRot.from_quat(amass_data["root_rot"])
        
        robot_positions_rot = np.einsum('tij,tnj->tni', root_rot.as_matrix(), robot_positions) + amass_data["root_trans"][:, None, :]
        
        z_offset = robot_positions_rot[:, :, 2].min().item() - 0.05
        
        # breakpoint()
        
        pose_aa = np.concatenate(
            [root_rot.as_rotvec()[:, None, :],
            rotate_axis[None, :, :] * joint_value.detach().cpu().numpy()[:,:,None]],
            axis=1
        )
        
        # breakpoint()
        data_new[data_key] = {
            "root_trans_offset": amass_data["root_trans"] - z_offset,
            "dof": joint_value.detach().cpu().numpy(),
            "pose_aa": pose_aa,
            "root_rot": amass_data["root_rot"],
            "fps": 30,
        }
        
        # break
    
    joblib.dump(data_new, "ik_new_final_amass.pkl")
        