import os
import numpy as np
from isaacgym import gymapi
from isaacgym import gymutil

# 初始化 Isaac Gym
gym = gymapi.acquire_gym()

# 配置模拟参数
sim_params = gymapi.SimParams()
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
sim_params.physx.solver_type = 1
sim_params.physx.num_position_iterations = 4
sim_params.physx.num_velocity_iterations = 1
sim_params.physx.contact_offset = 0.01
sim_params.physx.rest_offset = 0.0
sim_params.physx.friction_offset_threshold = 0.04
sim_params.physx.friction_correlation_distance = 0.025
sim_params.physx.max_depenetration_velocity = 1.0
sim_params.physx.default_buffer_size_multiplier = 1.0

# 创建模拟
sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
if sim is None:
    raise Exception("Failed to create sim")

# 创建场景
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1)
plane_params.distance = 0
plane_params.static_friction = 1.0
plane_params.dynamic_friction = 1.0
plane_params.restitution = 0.0
gym.add_ground(sim, plane_params)

# 设置图形窗口
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    raise Exception("Failed to create viewer")

# 创建场景
envs = []
env_spacing = 2.0
num_envs = 1
lower = gymapi.Vec3(-env_spacing, -env_spacing, 0.0)
upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)

# 加载URDF或模型路径
asset_root = (
    "/home/axell/desktop/11-30-ISR-PROJ/human2humanoid/resources/robots/rel2/urdf/"
)
asset_file = "rel2.urdf"

# 加载资产（机器人模型）
asset_options = gymapi.AssetOptions()

asset_options.default_dof_drive_mode = 0
asset_options.collapse_fixed_joints = True
asset_options.replace_cylinder_with_capsule = True
asset_options.flip_visual_attachments = False
asset_options.fix_base_link = False
asset_options.density = 0.001
asset_options.angular_damping = 0.0
asset_options.linear_damping = 0.0
asset_options.max_angular_velocity = 1000.0
asset_options.max_linear_velocity = 1000.0
asset_options.armature = 0.0
asset_options.thickness = 0.01
asset_options.disable_gravity = False
asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

humanoid_asset = gym.load_asset(sim, asset_root, asset_file, asset_options)
if humanoid_asset is None:
    raise Exception("Failed to load humanoid asset")
# 设置关节初始位置的字典
initial_joint_positions = {
    "left_shoulder_pitch_joint": 0.0,
    "left_shoulder_roll_joint": 0.0,
    "left_arm_yaw_joint": 0.0,
    "left_elbow_pitch_joint": 0.0,
    "left_elbow_yaw_joint": 0.0,
    "left_wrist_roll_joint": 0.0,
    "left_wrist_yaw_joint": 0.0,
    "right_shoulder_pitch_joint": 0.0,
    "right_shoulder_roll_joint": 0.0,
    "right_arm_yaw_joint": 0.0,
    "right_elbow_pitch_joint": 0.0,
    "right_elbow_yaw_joint": 0.0,
    "right_wrist_roll_joint": 0.0,
    "right_wrist_yaw_joint": 0.0,
    "waist_yaw_joint": 0.0,
    "waist_roll_joint": 0.0,
    "left_leg_roll_joint": 0.0,
    "left_leg_yaw_joint": 0.0,
    "left_leg_pitch_joint": 0.0,
    "left_knee_joint": 0.0,
    "left_ankle_pitch_joint": 0.0,
    "left_ankle_roll_joint": 0.0,
    "right_leg_roll_joint": 0.0,
    "right_leg_yaw_joint": 0.0,
    "right_leg_pitch_joint": 0.0,
    "right_knee_joint": 0.0,
    "right_ankle_pitch_joint": 0.0,
    "right_ankle_roll_joint": 0.0,
}

# 创建环境
for i in range(num_envs):
    env = gym.create_env(sim, lower, upper, num_envs)
    envs.append(env)

    # 加载机器人到环境中
    humanoid_pose = gymapi.Transform()
    humanoid_pose.p = gymapi.Vec3(0.0, 0.0, 1.0)  # 设置初始位置
    humanoid_actor = gym.create_actor(
        env, humanoid_asset, humanoid_pose, "humanoid", i, 0
    )

    # 读取机器人关节
    dof_props = gym.get_actor_dof_properties(env, humanoid_actor)
    dof_props["driveMode"].fill(gymapi.DOF_MODE_POS)  # 将所有关节驱动模式设为POS
    dof_props["stiffness"].fill(1000.0)  # 设置stiffness为1000
    dof_props["damping"].fill(0.0)  # 可根据需要设置damping
    gym.set_actor_dof_properties(env, humanoid_actor, dof_props)

    # 设置关节初始位置
    dof_states = gym.get_actor_dof_states(env, humanoid_actor, gymapi.STATE_ALL)
    dof_names = gym.get_asset_dof_names(humanoid_asset)

    for joint_name, joint_value in initial_joint_positions.items():
        if joint_name in dof_names:
            joint_index = dof_names.index(joint_name)
            dof_states["pos"][joint_index] = joint_value

    gym.set_actor_dof_states(env, humanoid_actor, dof_states, gymapi.STATE_ALL)

# 模拟循环
print("Starting simulation...")
while not gym.query_viewer_has_closed(viewer):
    # 处理用户输入
    gym.poll_viewer_events(viewer)

    # 模拟步骤
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # 更新图形界面
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)

    # 同步时间
    gym.sync_frame_time(sim)

# 清理资源
gym.destroy_sim(sim)
