import numpy as np
import genesis as gs
import joblib
import torch

########################## init ##########################
gs.init(backend=gs.gpu)

########################## create a scene ##########################
scene = gs.Scene(
    show_viewer    = False,
    viewer_options= gs.options.ViewerOptions(
        camera_pos    = (0.0, -2, 1.5),
        camera_lookat = (0.0, 0.0, 0.5),
        camera_fov    = 40,
        max_FPS       = 1,
    ),
    rigid_options=gs.options.RigidOptions(
        enable_joint_limit = False,
    ),
)

########################## entities ##########################
plane = scene.add_entity(
    gs.morphs.Plane(pos=(0, 0, -1)),
)
robot = scene.add_entity(
    gs.morphs.URDF(file='/home/axell/desktop/11-30-ISR-PROJ/human2humanoid/resources/robots/rel2/urdf/rel2.urdf', fixed=True, merge_fixed_links=False, collision=False),
)

# load data
amass = joblib.load("./amass_pos.pkl")
amass_keys = list(amass.keys())

# Sort sequences by length
seq_lengths = [(key, amass[key]['smpl_pos'].shape[0]) for key in amass_keys]
seq_lengths.sort(key=lambda x: x[1])
sorted_keys = [item[0] for item in seq_lengths]

# Parameters
chunk_size = 10000

debug = False  # 若为 True，则只运行一次并输出对应数据

save_joint_sort = ['left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_arm_yaw_joint', 'left_elbow_pitch_joint', 'left_elbow_yaw_joint', 'left_wrist_roll_joint', 'left_wrist_yaw_joint', 'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_arm_yaw_joint', 'right_elbow_pitch_joint', 'right_elbow_yaw_joint', 'right_wrist_roll_joint', 'right_wrist_yaw_joint', 'waist_yaw_joint', 'waist_roll_joint', 'left_leg_roll_joint', 'left_leg_yaw_joint', 'left_leg_pitch_joint', 'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint', 'right_leg_roll_joint', 'right_leg_yaw_joint', 'right_leg_pitch_joint', 'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint']

save_joint_index = [robot.get_joint(joint_name).dof_idx for joint_name in save_joint_sort]

# Concatenate data into chunks
all_data = []
original_lengths = {}

for key in sorted_keys:
    current_data = amass[key]['smpl_pos']
    original_lengths[key] = current_data.shape[0]
    all_data.append(current_data)

all_data = np.concatenate(all_data, axis=0)
num_chunks = (all_data.shape[0] + chunk_size - 1) // chunk_size

scene.build(n_envs=chunk_size, env_spacing=(2.0, 2.0))

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

enabled = [1, 3, 5, 7]

links = [robot.get_link(link_name) for ids, link_name in enumerate(new_robot_joint_pick) if ids in enabled]

start_idx = 0

init_qpos = torch.zeros_like(robot.get_qpos())

for chunk_idx in range(num_chunks):
    end_idx = min(start_idx + chunk_size, all_data.shape[0])
    chunk_data = all_data[start_idx:end_idx]

    pos = torch.from_numpy(chunk_data).float().to(gs.device)  # Shape (T_c, N, 3)
    # padding to chunk_size
    if pos.shape[0] < chunk_size:
        pos = torch.cat((pos, torch.zeros(chunk_size - pos.shape[0], pos.shape[1], pos.shape[2]).to(gs.device)), dim=0)
    
    target_pos = [pos[:, j, :] for j in enabled]

    print(f"Processing chunk {chunk_idx+1}/{num_chunks}, Shape: {pos.shape}")
    # breakpoint()
    q = robot.inverse_kinematics_multilink(
        links=links,
        poss=target_pos,
        pos_tol=0.05,
        init_qpos=init_qpos,
    )
    q = torch.nan_to_num(q, nan=0.0)

    # Store results
    if start_idx == 0:
        q_results = q.clone()
    else:
        q_results = torch.cat((q_results, q.clone()), dim=0)

    # robot.set_qpos(q[-1])  # Set the robot to the last configuration in the chunk
    # scene.step()

    start_idx = end_idx

# Restore results into the original sequences
start_idx = 0
for key in sorted_keys:
    seq_len = original_lengths[key]
    amass[key]['ik_results'] = q_results[start_idx:start_idx + seq_len, save_joint_index].cpu().numpy()
    start_idx += seq_len

# Save processed data
output_file = "amass_with_ik.pkl"
joblib.dump(amass, output_file)
print(f"Results saved to {output_file}")
