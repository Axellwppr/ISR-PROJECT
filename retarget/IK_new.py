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

from link_trans import joint_enable, new_robot_joint_pick

def refine_ik_results(
    smpl_pos,
    chain,
    link_cares,
    max_iter=300,
    device=torch.device("cpu"),
    pbar=None
):
    all_joint_names = chain.get_joint_parameter_names()
    joint_enable_idx = [all_joint_names.index(j) for j in joint_enable]
    T = smpl_pos.shape[0]
    N_dof = len(joint_enable_idx)
    N = len(all_joint_names)
    smpl_pos_t = torch.from_numpy(smpl_pos).float().to(device)
    
    # breakpoint()
    init = torch.zeros((T, N_dof), device=device)
    init[:, joint_enable.index("left_leg_pitch_joint")] = torch.pi * 27.5 / 180
    init[:, joint_enable.index("right_leg_pitch_joint")] = - torch.pi * 27.5 / 180
    init[:, joint_enable.index("left_knee_joint")] = torch.pi * 55 / 180
    init[:, joint_enable.index("right_knee_joint")] = - torch.pi * 55 / 180
    
    dof_pos_new = Variable(init, requires_grad=True)
    optimizer_pose = torch.optim.Adam([dof_pos_new], lr=0.2)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer_pose, step_size=50, gamma=0.5)

    best_loss = float("inf")
    best_dof = None
    
    # breakpoint()

    for iteration in range(max_iter):
        joint_value = torch.zeros((T, N), device=device)
        joint_value[:, joint_enable_idx] = dof_pos_new
        
        transforms = chain.forward_kinematics(joint_value)
        robot_positions = torch.stack(
            [transforms[name].get_matrix()[:, :3, 3] for name in link_cares],
            dim=1,
        )
        diff = robot_positions[:,1:,:] - robot_positions[:, 0:1, :] - smpl_pos_t
        loss = diff.norm(dim=-1).mean()
        
        optimizer_pose.zero_grad()
        loss.backward()
        optimizer_pose.step()
        
        if (iteration + 1) % 100 == 0:
            print(f"Iteration {iteration}, Loss: {loss.item()}")
        
        scheduler.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_dof = dof_pos_new.detach().clone()
    plot_dynamic_points((robot_positions[:,1:,:] - robot_positions[:, 0:1, :]).detach().cpu().numpy(), smpl_pos)

    return best_dof.cpu().numpy()

if __name__ == "__main__":
    data = joblib.load("amass_pos.pkl")
    # data = joblib.load("gen_pos.pkl")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open("/home/axell/desktop/11-30-ISR-PROJ/human2humanoid/resources/robots/rel2/urdf/rel2.urdf", "rb") as f:
        urdf_str = f.read()
    chain = pk.build_chain_from_urdf(urdf_str).to(device=device)

    rotate_axis = []
    joint_data = chain.get_joints()
    for idx, joint in enumerate(joint_data):
        if joint.joint_type == "revolute":
            rotate_axis.append(joint.axis)
    rotate_axis = torch.stack(rotate_axis, dim=0).to(device)

    all_pos = []
    original_lengths = {}
    sorted_keys = list(data.keys())

    for key in sorted_keys:
        pos_data = data[key]["smpl_pos"]
        original_lengths[key] = pos_data.shape[0]
        all_pos.append(pos_data)

    all_pos = np.concatenate(all_pos, axis=0)

    chunk_size = 50000
    T_total = all_pos.shape[0]
    num_chunks = (T_total + chunk_size - 1) // chunk_size

    refined_all = []
    start_idx = 0
    for chunk_idx in range(num_chunks):
        print(f"Processing chunk {chunk_idx + 1}/{num_chunks}")
        end_idx = min(start_idx + chunk_size, T_total)
        pos_chunk = all_pos[start_idx:end_idx]

        refined_chunk = refine_ik_results(
            smpl_pos=pos_chunk,
            chain=chain,
            link_cares=new_robot_joint_pick,
            # rotate_axis=rotate_axis,
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

    # joblib.dump(data, "amass_IK_new.pkl")
    # joblib.dump(data, "gen_IK_new.pkl")