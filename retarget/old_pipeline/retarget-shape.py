import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import pytorch_kinematics as pk
import matplotlib.pyplot as plt
import os
import joblib


# 假设你已使用smpl_parser解析出SMPL模型与关节坐标
from smpl_sim.smpllib.smpl_parser import SMPL_Parser
from smpl_sim.smpllib.smpl_joint_names import SMPL_BONE_ORDER_NAMES


# 定义绘图函数
def plot_keypoints(robot_points, smpl_points, iteration, save_dir="plots"):
    os.makedirs(save_dir, exist_ok=True)
    robot_points = robot_points.detach().cpu().numpy()[0]
    smpl_points = smpl_points.detach().cpu().numpy()[0]
    planes = [("X", "Y"), ("Y", "Z"), ("X", "Z")]
    robot_color = "blue"
    smpl_color = "red"
    robot_label = "XBot"
    smpl_label = "SMPL"
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    for ax, (axis1, axis2) in zip(axs, planes):
        ax.scatter(
            robot_points[:, {"X": 0, "Y": 1, "Z": 2}[axis1]],
            robot_points[:, {"X": 0, "Y": 1, "Z": 2}[axis2]],
            c=robot_color,
            label=robot_label,
            alpha=0.6,
        )
        ax.scatter(
            smpl_points[:, {"X": 0, "Y": 1, "Z": 2}[axis1]],
            smpl_points[:, {"X": 0, "Y": 1, "Z": 2}[axis2]],
            c=smpl_color,
            label=smpl_label,
            alpha=0.6,
        )
        ax.set_xlabel(f"{axis1}")
        ax.set_ylabel(f"{axis2}")
        ax.set_title(f"{axis1}-{axis2} Plane")
        ax.legend()
        ax.grid(True)
    plt.suptitle(f"Iter-{iteration}")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = os.path.join(save_dir, f"iteration_{iteration}.png")
    plt.savefig(save_path)
    plt.close()


# ================================
# 1. 加载SMPL模型
# ================================
device = torch.device("cpu")
smpl_parser = SMPL_Parser(model_path="data/smpl", gender="neutral")

# 定义一个SMPL标准站姿姿态
pose_aa_stand = torch.zeros((1, 72))
beta = torch.zeros((1, 10))
trans = torch.zeros((1, 3))

verts, smpl_joints = smpl_parser.get_joints_verts(pose_aa_stand, beta, trans)
root_pos = smpl_joints[:, 0]

from link_trans import smpl_joint_pick, new_robot_joint_pick

smpl_joint_pick_idx = [SMPL_BONE_ORDER_NAMES.index(j) for j in smpl_joint_pick]

# ================================
# 2. 使用pytorch_kinematics加载URDF并建立Kinematics Chain
# ================================
with open(
    "/home/axell/desktop/humanoid/humanoid-benchmark-main/isaacLab/manipulation/assets/urdf/rel2/urdf/rel2.urdf",
    "rb",
) as f:
    urdf_str = f.read()

chain = pk.build_chain_from_urdf(urdf_str)
all_joint_names = chain.get_joint_parameter_names()
print(all_joint_names)
# breakpoint()
all_link_names = chain.get_link_names()
breakpoint()
robot_joint_pick_idx = [all_link_names.index(ln) for ln in new_robot_joint_pick]

# ================================
# 3. 定义优化变量：shape与scale
# ================================
shape_new = Variable(
    torch.zeros([1, 10], dtype=torch.float32, device=device), requires_grad=True
)
scale = Variable(
    torch.ones([1], dtype=torch.float32, device=device), requires_grad=True
)

optimizer = optim.Adam([shape_new, scale], lr=0.1)

# ================================
# 4. 前向过程与优化
# ================================
joint_angles = torch.zeros(
    (1, len(all_joint_names)), dtype=torch.float32, device=device, requires_grad=False
)
joint_angles[0, 1] = torch.pi / 2
joint_angles[0, 8] = -torch.pi / 2
joint_angles[0, 18] = torch.pi * 27.5 / 180
joint_angles[0, 24] = -torch.pi * 27.5 / 180
joint_angles[0, 19] = torch.pi * 55 / 180
joint_angles[0, 25] = -torch.pi * 55 / 180


# 定义保存图像的目录
plot_save_dir = "plots"

rotate_xbot = torch.tensor(
    [[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=torch.float32, device=device
)

for iteration in range(1000):
    optimizer.zero_grad()

    # 重新计算SMPL关节坐标（当前shape_new）
    verts_opt, joints_opt = smpl_parser.get_joints_verts(
        pose_aa_stand, shape_new, trans
    )
    root_pos_opt = joints_opt[:, 0]
    scaled_joints_opt = (joints_opt - root_pos_opt) * scale + root_pos_opt

    # 计算机器人FK
    robot_transforms = chain.forward_kinematics(joint_angles[0])

    # 提取感兴趣的link位置
    robot_positions = []
    for idx in robot_joint_pick_idx:
        link_name = all_link_names[idx]
        T = robot_transforms[link_name].get_matrix()[0]  # 4x4
        # breakpoint()
        pos = torch.tensor(
            [T[0, 3], T[1, 3], T[2, 3]], dtype=torch.float32
        )  # 提取位置部分
        robot_positions.append(pos)
    robot_positions = torch.stack(robot_positions, dim=0).unsqueeze(
        0
    )  # (1, len(robot_joint_pick), 3)
    print(robot_positions.shape)
    robot_positions = torch.matmul(robot_positions, rotate_xbot)

    # 从SMPL中提取对应关节
    target_smpl_pos = scaled_joints_opt[
        :, smpl_joint_pick_idx
    ]  # (1, len(smpl_joint_pick), 3)

    # 计算误差
    diff = robot_positions - target_smpl_pos
    loss = diff.norm(dim=-1).mean()

    if iteration % 100 == 0:
        print(f"Iteration {iteration}, loss: {loss.item()}")
        # 调用绘图函数
        plot_keypoints(
            robot_positions, target_smpl_pos, iteration, save_dir=plot_save_dir
        )

    loss.backward()
    optimizer.step()

# 优化完成后保存结果
os.makedirs("data/new_robot", exist_ok=True)
joblib.dump(
    (shape_new.detach().cpu(), scale.detach().cpu()),
    "data/new_robot/shape_optimized.pkl",
)
print("Shape fitting completed and saved to data/new_robot/shape_optimized.pkl")
