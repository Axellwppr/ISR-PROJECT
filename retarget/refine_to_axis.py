import numpy as np
from tqdm import tqdm
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

import pytorch_kinematics as pk
from scipy.spatial.transform import Rotation as sRot

from utils import (
    compute_rotation_matrix_batch,
    compute_rotation_matrix,
    plot_dynamic_points,
)

from link_trans import joint_enable, new_robot_joint_pick, joint_mirror
import torch
import numpy as np


with open("/home/axell/desktop/11-30-ISR-PROJ/human2humanoid/resources/robots/rel2/urdf/rel2.urdf", "rb") as f:
    urdf_str = f.read()
chain = pk.build_chain_from_urdf(urdf_str)
link_names = chain.get_link_names()
joint_names = chain.get_joint_parameter_names()

joint_enable_idx = [joint_names.index(joint_name) for joint_name in joint_enable]
joint_care_idx = [link_names.index(link_name) for link_name in new_robot_joint_pick]
joint_mirror_idx = [joint_enable.index(joint_name) for joint_name in joint_mirror]

rotate_axis = []
joint_data = chain.get_joints()
for idx, joint in enumerate(joint_data):
    if joint.joint_type == "revolute":
        rotate_axis.append(joint.axis)
rotate_axis = torch.stack(rotate_axis, dim=0).cpu().numpy()

def check_state(positions):
    left_foot = positions[:, 2, 2]   # z轴位置
    right_foot = positions[:, 4, 2]
    base_height = positions[:, 0, 2]
    
    # check baseheight
    thres = [0.65, 0.95]
    
    height_flag = True
    if np.any(base_height < thres[0]) or np.any(base_height > thres[1]):
        height_flag = False
        
    threshold = 0.15  # 设置抬脚阈值(米)
    
    # 判断脚的抬起状态
    left_up = left_foot > threshold
    right_up = right_foot > threshold
    
    # 计算各种状态的时间
    total_frames = len(left_foot)
    left_up_frames = np.sum(left_up)
    right_up_frames = np.sum(right_up)
    
    # 判断先抬起的脚
    if np.any(left_up) and np.any(right_up):
        first_left_up = np.where(left_up)[0][0]
        first_right_up = np.where(right_up)[0][0]
        first_up = 'left' if first_left_up < first_right_up else 'right'
    else:
        first_up = 'none'
    
    return {
        'first_up': first_up,
        'left_up_ratio': left_up_frames / total_frames,
        'right_up_ratio': right_up_frames / total_frames,
        'total_frames': total_frames,
        'base_height_frames': base_height,
        'height_flag': height_flag
    }

def plot_statistics(all_stats):
    plt.figure(figsize=(16, 6))
    
    # 先抬脚统计
    plt.subplot(131)
    first_up_data = [[all_stats['first_up_count']['left']], 
                     [all_stats['first_up_count']['right']]]
    sns.heatmap(first_up_data, 
                annot=True, 
                fmt='d',
                yticklabels=['Left First', 'Right First'],
                xticklabels=['Count'],
                cmap='YlOrRd')
    plt.title('First Foot Up Statistics')
    
    # 抬脚时间比例
    plt.subplot(132)
    time_ratio_data = [[all_stats['total_left_up_frames'] / all_stats['total_frames']], 
                       [all_stats['total_right_up_frames'] / all_stats['total_frames']]]
    sns.heatmap(time_ratio_data,
                annot=True,
                fmt='.3f',
                yticklabels=['Left Foot', 'Right Foot'],
                xticklabels=['Up Time Ratio'],
                cmap='YlOrRd')
    plt.title('Foot Up Time Ratio')

    plt.subplot(133)
    sns.histplot(all_stats['base_height_frames'], bins=30, kde=True, color='blue')
    plt.title('Base Height Distribution')
    
    plt.tight_layout()
    plt.savefig('feet_analysis.png')
    plt.close()

def get_forward_vector(root_rot):
    rot_mat = sRot.from_quat(root_rot[0]).as_matrix()
    forward_local = np.array([1, 0, 0])
    forward_global = rot_mat @ forward_local
    forward_global[2] = 0
    return forward_global / np.linalg.norm(forward_global)

def get_plane_normal(root_rot):
    forward_global = get_forward_vector(root_rot)
    up_vec = np.array([0, 0, 1], dtype=float)
    # 使用叉乘获取法向量，以 forward_global 和 up_vec 构建的平面
    plane_normal = np.cross(forward_global, up_vec)
    plane_normal /= np.linalg.norm(plane_normal)
    return plane_normal

mirror_matrix = np.array([
    [1, 0, 0],
    [0, -1, 0],
    [0, 0, 1]]
)

def mirror_data(dof, root_trans, root_rot, rotate_axis):
    # 1) 计算镜像平面的法向量
    n = get_plane_normal(root_rot).reshape(3, 1)
    # 2) 使用 R = I - 2*(n n^T) 构造镜像矩阵
    reflect_mat = np.eye(3) - 2 * (n @ n.T)
    
    root_trans_mirror = root_trans.copy()
    for i in range(root_trans_mirror.shape[0]):
        root_trans_mirror[i] = reflect_mat @ root_trans_mirror[i]
    
    # 计算旋转镜像，这里简单通过对每帧的旋转矩阵做R * R^mirror
    root_rot_mirror = []
    for i in range(root_rot.shape[0]):
        rot_mat = sRot.from_quat(root_rot[i]).as_matrix()
        mirrored_mat = reflect_mat @ rot_mat
        # 为保持右手系，可再乘一次reflect_mat.T
        mirrored_mat = mirrored_mat @ mirror_matrix #reflect_mat.T
        root_rot_mirror.append(sRot.from_matrix(mirrored_mat).as_quat())
    root_rot_mirror = np.array(root_rot_mirror)
    
    dof_mirror = -dof[:, joint_mirror_idx].clone()
    return dof_mirror, root_trans_mirror, root_rot_mirror

def compute_fk_and_stats(dof, root_trans, root_rot, all_stats=None):
    """计算正向运动学并更新统计信息
    Args:
        dof: 关节角度数据
        root_trans: 根节点位置
        root_rot: 根节点旋转(四元数)
        all_stats: 可选的统计信息字典
    Returns:
        dict: 包含处理后的数据
        np.ndarray: 机器人位置数据(用于可视化)
    """
    joint_value = torch.zeros((dof.shape[0], len(joint_names)))
    joint_value[:, joint_enable_idx] = dof
    
    transforms = chain.forward_kinematics(joint_value)
    robot_positions = torch.stack(
        [transforms[link_names[idx]].get_matrix()[:, :3, 3] for idx in joint_care_idx],
        dim=1,
    ).detach().cpu().numpy()
    
    root_rot_mat = sRot.from_quat(root_rot).as_matrix()
    robot_positions_rot = np.einsum('tij,tnj->tni', root_rot_mat, robot_positions) + root_trans[:, None, :]
    
    z_offset = robot_positions_rot[:, :, 2].min().item()
    
    if all_stats is not None:
        stat = check_state(robot_positions_rot - z_offset)
        all_stats['first_up_count'][stat['first_up']] += 1
        all_stats['total_left_up_frames'] += stat['left_up_ratio'] * stat['total_frames']
        all_stats['total_right_up_frames'] += stat['right_up_ratio'] * stat['total_frames']
        all_stats['total_frames'] += stat['total_frames']
        all_stats['base_height_frames'].extend(stat['base_height_frames'].tolist())  
    
    pose_aa = np.concatenate(
        [sRot.from_quat(root_rot).as_rotvec()[:, None, :],
         rotate_axis[None, :, :] * joint_value.detach().cpu().numpy()[:,:,None]],
        axis=1
    )
    
    return {
        "root_trans_offset": root_trans - z_offset,
        "dof": joint_value.detach().cpu().numpy(),
        "pose_aa": pose_aa,
        "root_rot": root_rot,
        "fps": 30,
    }, robot_positions_rot, stat['height_flag']

if __name__ == "__main__":
    data = joblib.load("gen_IK_new.pkl")
    filter_keys = list(data.keys())
    
    # 初始化统计数据
    all_stats = {
        'first_up_count': {'left': 0, 'right': 0, 'none': 0},
        'total_left_up_frames': 0,
        'total_right_up_frames': 0,
        'total_frames': 0,
        'base_height_frames': []
    }
    
    pbar = tqdm(filter_keys, desc="merge data")
    data_new = {}
    
    for data_key in pbar:
        amass_data = data[data_key]
        dof = torch.from_numpy(amass_data["ik_refined"]).float()
        
        amass_data["root_trans"] = amass_data["root_trans"] - amass_data["root_trans"][0:1, :]
        
        # 计算原始数据的FK
        result_orig, pos_orig, flag = compute_fk_and_stats(
            dof, 
            amass_data["root_trans"],
            amass_data["root_rot"],
            all_stats
        )
        
        if not flag:
            print(f"Invalid data: {data_key}")
            continue
        
        # 计算镜像数据的FK
        dof_m, root_trans_m, root_rot_m = mirror_data(
            dof,
            amass_data["root_trans"],
            amass_data["root_rot"],
            rotate_axis
        )
        result_mirror, pos_mirror, _ = compute_fk_and_stats(
            dof_m,
            root_trans_m,
            root_rot_m,
            all_stats  # 同样更新统计信息
        )
        
        # 保存原始数据
        data_new[data_key] = result_orig
        # 可选：保存镜像数据
        data_new[f"{data_key}_mirror"] = result_mirror
    
    # 用封装好的函数来绘制统计结果
    plot_statistics(all_stats)
    
    joblib.dump(data_new, "ik_new_final_gen.pkl")