import joblib
import torch
import pytorch_kinematics as pk
from scipy.spatial.transform import Rotation as sRot
from torch.autograd import Variable
from tqdm import tqdm
import numpy as np

from utils import (
    compute_rotation_matrix_batch,
    compute_rotation_matrix,
    plot_dynamic_points,
)

from link_trans import joint_enable

def refine_ik_results(
    smpl_pos,
    ik_result,
    chain,
    joint_cares,
    link_names,
    rotate_axis,
    max_iter=500,
    device=torch.device("cpu"),
    pbar=None
):
    all_joint_names = chain.get_joint_parameter_names()
    joint_enable_idx = [all_joint_names.index(j) for j in joint_enable]
    T = smpl_pos.shape[0]
    N = ik_result.shape[1]
    smpl_pos_t = torch.from_numpy(smpl_pos).float().to(device)
    ik_result_t = torch.from_numpy(ik_result).float().to(device)

    dof_pos_new = Variable(ik_result_t[:, joint_enable_idx].clone(), requires_grad=True)
    optimizer_pose = torch.optim.Adadelta([dof_pos_new], lr=100)

    best_loss = float("inf")
    best_dof = None

    for iteration in range(max_iter):
        # breakpoint()
        joint_value = torch.zeros((T, N), device=device)
        joint_value[:, joint_enable_idx] = dof_pos_new
        # transforms = chain.forward_kinematics(torch.zeros_like(dof_pos_new))
        transforms = chain.forward_kinematics(joint_value)
        robot_positions = torch.stack(
            [transforms[link_names[idx]].get_matrix()[:, :3, 3] for idx in joint_cares],
            dim=1,
        )
        # breakpoint()
        diff = robot_positions[:,1:,:] - robot_positions[:, 0:1, :] - smpl_pos_t
        loss = diff.norm(dim=-1).mean()
        
        
        optimizer_pose.zero_grad()
        loss.backward()
        optimizer_pose.step()
        
        if (iteration + 1) % 100 == 0:
            print(f"Iteration {iteration}, Loss: {loss.item()}")

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_dof = dof_pos_new.detach().clone()
    # plot_dynamic_points(robot_positions[:, 1:, :].detach().cpu().numpy(), smpl_pos)

    return best_dof.cpu().numpy()

if __name__ == "__main__":
    data = joblib.load("amass_with_ik.pkl")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open("/home/axell/desktop/11-30-ISR-PROJ/human2humanoid/resources/robots/rel2/urdf/rel2.urdf", "rb") as f:
        urdf_str = f.read()
    chain = pk.build_chain_from_urdf(urdf_str).to(device=device)
    link_names = chain.get_link_names()
    joint_names = chain.get_joint_parameter_names()
    
    print(joint_names)

    rotate_axis = []
    joint_data = chain.get_joints()
    for idx, joint in enumerate(joint_data):
        if joint.joint_type == "revolute":
            rotate_axis.append(joint.axis)
    rotate_axis = torch.stack(rotate_axis, dim=0).to(device)

    joint_enable_idx = [
        link_names.index("base_link"),
        link_names.index("left_knee_link"),
        link_names.index("left_ankle_pitch_link"),
        link_names.index("right_knee_link"),
        link_names.index("right_ankle_pitch_link"),
        link_names.index("left_arm_yaw_link"),
        link_names.index("left_hand_ee_link"),
        link_names.index("right_arm_yaw_link"),
        link_names.index("right_hand_ee_link"),
    ]

    all_pos = []
    all_ik = []
    original_lengths = {}
    sorted_keys = list(data.keys())

    for key in sorted_keys:
        pos_data = data[key]["smpl_pos"]
        ik_init = data[key]["ik_results"]
        original_lengths[key] = pos_data.shape[0]
        all_pos.append(pos_data)
        all_ik.append(ik_init)

    all_pos = np.concatenate(all_pos, axis=0)
    all_ik = np.concatenate(all_ik, axis=0)

    chunk_size = 10000
    T_total = all_pos.shape[0]
    num_chunks = (T_total + chunk_size - 1) // chunk_size

    refined_all = []
    start_idx = 0
    for chunk_idx in range(num_chunks):
        print(f"Processing chunk {chunk_idx + 1}/{num_chunks}")
        end_idx = min(start_idx + chunk_size, T_total)
        pos_chunk = all_pos[start_idx:end_idx]
        ik_chunk = all_ik[start_idx:end_idx]

        refined_chunk = refine_ik_results(
            smpl_pos=pos_chunk,
            ik_result=ik_chunk,
            chain=chain,
            joint_cares=joint_enable_idx,
            link_names=link_names,
            rotate_axis=rotate_axis,
            device=device
        )
        refined_all.append(refined_chunk)
        start_idx = end_idx

    refined_all = np.concatenate(refined_all, axis=0)

    # 将结果拆分回各自序列
    start_idx = 0
    for key in sorted_keys:
        seq_len = original_lengths[key]
        data[key]["ik_refined"] = refined_all[start_idx:start_idx + seq_len]
        start_idx += seq_len

    joblib.dump(data, "amass_with_ik_refined.pkl")
    print("Refined IK results saved to amass_with_ik_refined.pkl")