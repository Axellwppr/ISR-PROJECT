import torch


def compute_rotation_matrix_batch(A, B, C):
    """
    Computes a batch of rotation matrices that rotate the projections of vectors A
    onto the planes orthogonal to vectors C to the projections of vectors B
    onto the same planes.

    Args:
        A (torch.Tensor): Batch of initial vectors, shape (N, 3).
        B (torch.Tensor): Batch of target vectors, shape (N, 3).
        C (torch.Tensor): Batch of axis vectors, shape (N, 3).

    Returns:
        torch.Tensor: Batch of rotation matrices, shape (N, 3, 3).
    """
    # Ensure tensors are float for computation
    A, B, C = A.float(), B.float(), C.float()

    # Normalize C for plane projection
    C_unit = C / torch.norm(C, dim=1, keepdim=True)

    # Project A and B onto the plane orthogonal to C
    A_proj = A - (torch.sum(A * C_unit, dim=1, keepdim=True) * C_unit)
    B_proj = B - (torch.sum(B * C_unit, dim=1, keepdim=True) * C_unit)

    # Normalize the projections
    A_proj_norm = A_proj / torch.norm(A_proj, dim=1, keepdim=True)
    B_proj_norm = B_proj / torch.norm(B_proj, dim=1, keepdim=True)

    # Compute the rotation angle and axis
    cos_theta = torch.sum(A_proj_norm * B_proj_norm, dim=1).clamp(-1.0, 1.0)
    angle = torch.arccos(cos_theta)
    axis = torch.cross(A_proj_norm, B_proj_norm, dim=1)

    # Handle cases where the axis norm is close to zero
    axis_norm = torch.norm(axis, dim=1, keepdim=True)
    axis = torch.where(axis_norm > 1e-6, axis / axis_norm, torch.zeros_like(axis))

    # Special case: when A_proj and B_proj are aligned or anti-aligned
    is_aligned = axis_norm.squeeze() < 1e-6
    aligned_axis = torch.cross(
        C_unit, torch.tensor([1.0, 0.0, 0.0], device=A.device).expand_as(C_unit)
    )
    aligned_axis = torch.where(
        torch.norm(aligned_axis, dim=1, keepdim=True) < 1e-6,
        torch.cross(
            C_unit, torch.tensor([0.0, 1.0, 0.0], device=A.device).expand_as(C_unit)
        ),
        aligned_axis,
    )
    aligned_axis = aligned_axis / torch.norm(aligned_axis, dim=1, keepdim=True)
    axis = torch.where(is_aligned.unsqueeze(-1), aligned_axis, axis)
    angle = torch.where(
        is_aligned, torch.where(cos_theta > 0, torch.zeros_like(angle), torch.pi), angle
    )

    # Rodrigues' rotation formula
    K = torch.zeros(A.shape[0], 3, 3, device=A.device)
    K[:, 0, 1] = -axis[:, 2]
    K[:, 0, 2] = axis[:, 1]
    K[:, 1, 0] = axis[:, 2]
    K[:, 1, 2] = -axis[:, 0]
    K[:, 2, 0] = -axis[:, 1]
    K[:, 2, 1] = axis[:, 0]

    I = torch.eye(3, device=A.device).unsqueeze(0).repeat(A.shape[0], 1, 1)
    R = (
        I
        + torch.sin(angle).unsqueeze(-1).unsqueeze(-1) * K
        + (1 - torch.cos(angle)).unsqueeze(-1).unsqueeze(-1) * torch.matmul(K, K)
    )

    return R


def compute_rotation_matrix(A, B, C, D):
    """
    计算旋转矩阵 R，使得 R·A 尽可能接近 C，R·B 尽可能接近 D。

    参数:
    A, B, C, D: torch.Tensor
        输入的四个三维向量，形状为 (batch, 3)。不要求单位长度且不一定两两正交。

    返回:
    R: torch.Tensor
        3x3 的旋转矩阵，形状为 (batch, 3, 3)。
    """
    # 1. 归一化向量
    A_norm = A / A.norm(dim=1, keepdim=True)
    B_norm = B / B.norm(dim=1, keepdim=True)
    C_norm = C / C.norm(dim=1, keepdim=True)
    D_norm = D / D.norm(dim=1, keepdim=True)

    # 2. 正交化 A 和 B，C 和 D 使用格拉姆-施密特
    # 正交化 B 相对于 A
    proj_B_on_A = torch.sum(B_norm * A_norm, dim=1, keepdim=True) * A_norm
    B_orth = B_norm - proj_B_on_A
    B_orth = B_orth / B_orth.norm(dim=1, keepdim=True)

    # 正交化 D 相对于 C
    proj_D_on_C = torch.sum(D_norm * C_norm, dim=1, keepdim=True) * C_norm
    D_orth = D_norm - proj_D_on_C
    D_orth = D_orth / D_orth.norm(dim=1, keepdim=True)

    # 3. 计算第三个基向量（确保右手坐标系）
    E = torch.cross(A_norm, B_orth, dim=1)
    F = torch.cross(C_norm, D_orth, dim=1)

    # 4. 构建基矩阵 M1 和 M2
    M1 = torch.stack((A_norm, B_orth, E), dim=2)  # 形状: (batch, 3, 3)
    M2 = torch.stack((C_norm, D_orth, F), dim=2)  # 形状: (batch, 3, 3)

    # 5. 计算初始旋转矩阵 R_initial
    R_initial = torch.bmm(M2, M1.transpose(1, 2))  # 形状: (batch, 3, 3)

    # # 6. 使用 SVD 分解 R_initial
    # U, S, Vh = torch.linalg.svd(R_initial)

    # # 7. 计算旋转矩阵 R = U * Vh
    # R = torch.bmm(U, Vh)

    # # 8. 确保旋转矩阵的行列式为 +1
    # det = torch.det(R)
    # mask = det < 0
    # if mask.any():
    #     # 翻转 U 的最后一列
    #     U_flipped = U.clone()
    #     U_flipped[mask, :, 2] *= -1
    #     # 重新计算 R
    #     R = torch.bmm(U_flipped, Vh)

    return R_initial

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation


def plot_dynamic_points(points1, points2):
    """
    绘制两组点云的动态变化图

    参数:
    points1: shape为[T,N,3]的numpy数组，表示第一组点云
    points2: shape为[T,N,3]的numpy数组，表示第二组点云
    """
    # 检查输入维度
    assert points1.shape == points2.shape, "两组点云的形状必须相同"
    assert (
        len(points1.shape) == 3 and points1.shape[2] == 3
    ), "输入数组必须是[T,N,3]形状"

    T, N, _ = points1.shape

    # 创建图形和3D坐标系
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # 初始化散点图对象
    scatter1 = ax.scatter([], [], [], c="red", marker="o", label="Points 1")
    scatter2 = ax.scatter([], [], [], c="blue", marker="o", label="Points 2")

    # 计算坐标轴范围
    x_min = min(points1[..., 0].min(), points2[..., 0].min())
    x_max = max(points1[..., 0].max(), points2[..., 0].max())
    y_min = min(points1[..., 1].min(), points2[..., 1].min())
    y_max = max(points1[..., 1].max(), points2[..., 1].max())
    z_min = min(points1[..., 2].min(), points2[..., 2].min())
    z_max = max(points1[..., 2].max(), points2[..., 2].max())

    # 设置坐标轴范围和标签
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.set_zlim([z_min, z_max])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()

    # 添加交互提示
    fig.text(
        0.02,
        0.98,
        "交互提示：\n"
        "- 左键拖动：旋转视角\n"
        "- 右键拖动：平移视角\n"
        "- 滚轮：缩放\n"
        "- ESC：退出",
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    # 更新函数
    def update(frame):
        # 更新两组点的位置
        scatter1._offsets3d = (
            points1[frame, :, 0],
            points1[frame, :, 1],
            points1[frame, :, 2],
        )
        scatter2._offsets3d = (
            points2[frame, :, 0],
            points2[frame, :, 1],
            points2[frame, :, 2],
        )

        # 更新标题显示当前帧
        ax.set_title(f"Frame {frame}/{T-1}")
        return scatter1, scatter2

    # 创建动画，设置repeat=True确保循环播放
    anim = FuncAnimation(
        fig,
        update,
        frames=T,
        interval=50,  # 50ms between frames
        blit=False,
        repeat=True,  # 确保循环播放
    )

    # 添加交互控件
    plt.gcf().canvas.mpl_connect(
        "key_press_event", lambda event: plt.close() if event.key == "escape" else None
    )

    # 显示图形
    plt.show()

    return anim
