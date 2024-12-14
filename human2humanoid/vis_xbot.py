import os
import joblib
import numpy as np
from isaacgym import gymapi, gymutil

import torch
from link_trans import joint_enable, joint_enable_idx


def main():
    # ================================
    # 1. 初始化Isaac Gym环境
    # ================================
    # 创建Gym实例
    gym = gymapi.acquire_gym()

    # 配置模拟参数
    sim_params = gymapi.SimParams()
    sim_params.dt = 1.0 / 20.0  # 仿真步长
    sim_params.substeps = 2
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)

    # 创建仿真环境（使用默认的物理引擎和设备）
    sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
    if sim is None:
        print("Failed to create simulation")
        return

    # 创建一个地面平面
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0, 0, 1)
    gym.add_ground(sim, plane_params)

    # 创建视图窗口
    viewer = gym.create_viewer(sim, gymapi.CameraProperties())
    if viewer is None:
        print("Failed to create viewer")
        return

    # ================================
    # 2. 加载机器人URDF
    # ================================
    # 设置机器人URDF路径

    # 加载机器人资产
    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = True  # 机器人基座是自由移动的
    asset_options.armature = 0.0
    asset_options.disable_gravity = True
    asset_options.vhacd_enabled = False  # 根据需要启用V-HACD
    # asset_options.

    robot_asset = gym.load_asset(
        sim,
        "/home/axell/desktop/humanoid/humanoid-benchmark-main/isaacLab/manipulation/assets/urdf/rel2/urdf",
        "rel2.urdf",
        asset_options,
    )
    if robot_asset is None:
        print("Failed to load robot asset")
        return

    # ================================
    # 3. 创建环境并实例化机器人
    # ================================
    # 创建一个环境
    env = gym.create_env(
        sim, gymapi.Vec3(-1.0, -1.0, 0.0), gymapi.Vec3(1.0, 1.0, 1.0), 1
    )

    # ================================
    # 4. 加载轨迹数据
    # ================================
    # 加载test.pkl
    trajectory_path = "data/new_robot/amass_all.pkl"  # 请确保路径正确
    if not os.path.exists(trajectory_path):
        print(f"轨迹文件未找到: {trajectory_path}")
        return

    trajectory_data = joblib.load(trajectory_path)
    if not trajectory_data:
        print("轨迹数据为空")
        return

    # 获取第一条轨迹
    first_key = list(trajectory_data.keys())[13]
    traj = trajectory_data[first_key]
    # root_trans = traj["root_trans_offset"]  # (3,)
    dof_pos = traj["dof"]  # (num_frames, num_dofs)
    root_rot = traj["root_rot"]  # (4,) 四元数 [x, y, z, w]
    fps = traj["fps"]

    num_frames = dof_pos.shape[0]
    num_dofs = dof_pos.shape[1]
    print(f"轨迹帧数: {num_frames}, 关节数: {dof_pos.shape[1]}")

    # ================================
    # 5. 创建机器人实例
    # ================================
    # 设置机器人的初始位置和旋转
    robot_start_pose = gymapi.Transform()
    robot_start_pose.p = gymapi.Vec3(0, 0, 2)
    robot_start_pose.r = gymapi.Quat(0, 0, 0, 1)

    # 创建机器人实例
    robot_handle = gym.create_actor(env, robot_asset, robot_start_pose, "robot", 0, 1)

    # 获取机器人的关节信息
    # robot_dof_props = gym.get_actor_dof_properties(env, robot_handle)
    num_robot_dofs = gym.get_asset_dof_count(robot_asset)
    print(f"机器人关节数: {num_robot_dofs}")

    # 确保轨迹中的关节数与机器人匹配
    if dof_pos.shape[1] != len(joint_enable):
        print(
            f"轨迹关节数({dof_pos.shape[1]})与机器人关节数({len(joint_enable)})不匹配"
        )
        return

    dof_props = gym.get_actor_dof_properties(env, robot_handle)

    # 遍历所有关节，设置为位置控制，并将刚度设置为100
    for i in range(num_robot_dofs):
        dof_props["driveMode"][i] = gymapi.DOF_MODE_POS  # 设置为位置控制
        dof_props["stiffness"][i] = 1000.0  # 设置刚度为100
        dof_props["damping"][i] = 40.0  # 根据需要设置阻尼，可以调整
        # 你也可以根据需要设置`friction`和其他属性

    # 将修改后的关节属性应用回机器人
    gym.set_actor_dof_properties(env, robot_handle, dof_props)

    # ================================
    # 6. 重放轨迹
    # ================================
    # 设置摄像机视角
    gym.viewer_camera_look_at(
        viewer,
        env,
        gymapi.Vec3(1, 0, 1),
        gymapi.Vec3(0, 0, 2),
    )

    frame = 0
    while not gym.query_viewer_has_closed(viewer) and frame < num_frames:
        # 获取当前帧的关节角度

        current_dof_pos = np.zeros((num_robot_dofs))
        current_dof_pos[joint_enable_idx] = dof_pos[frame]
        # breakpoint()
        print(current_dof_pos)

        # 设置机器人的关节角度
        gym.set_actor_dof_position_targets(env, robot_handle, current_dof_pos.tolist())

        # 步进模拟
        gym.simulate(sim)
        gym.fetch_results(sim, True)

        # 渲染
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, True)
        gym.sync_frame_time(sim)

        frame += 1

    # 等待关闭
    while not gym.query_viewer_has_closed(viewer):
        gym.poll_viewer_events(viewer)
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, True)
        gym.sync_frame_time(sim)

    # 清理资源
    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)


if __name__ == "__main__":
    main()
