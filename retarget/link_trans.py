smpl_joint_pick = [
    "Pelvis",
    # "L_Hip",
    "L_Knee",
    "L_Ankle",
    # "R_Hip",
    "R_Knee",
    "R_Ankle",
    "L_Shoulder",
    "L_Elbow",
    "L_Hand",
    "R_Shoulder",
    "R_Elbow",
    "R_Hand",
    "Head",
]

new_robot_joint_pick = [
    "waist_roll_link",
    # "left_leg_yaw_link",
    "left_knee_link",
    "left_ankle_pitch_link",
    # "right_leg_yaw_link",
    "right_knee_link",
    "right_ankle_pitch_link",
    "left_shoulder_roll_link",
    "left_arm_yaw_link",
    "left_hand_ee_link",
    "right_shoulder_roll_link",
    "right_arm_yaw_link",
    "right_hand_ee_link",
    "neck_pitch_link",
]

joint_enable = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_arm_yaw_joint",
    "left_elbow_pitch_joint",
    "left_wrist_roll_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_arm_yaw_joint",
    "right_elbow_pitch_joint",
    "right_wrist_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "left_leg_roll_joint",
    "left_leg_pitch_joint",
    "left_knee_joint",
    "right_leg_roll_joint",
    "right_leg_pitch_joint",
    "right_knee_joint",
]

# import pytorch_kinematics as pk

# with open(
#     "/home/axell/desktop/humanoid/humanoid-benchmark-main/isaacLab/manipulation/assets/urdf/rel2/urdf/rel2.urdf",
#     "rb",
# ) as f:
#     urdf_str = f.read()

# chain = pk.build_chain_from_urdf(urdf_str)

# all_joint_names = chain.get_joint_parameter_names()

# joint_enable_idx = [all_joint_names.index(j) for j in joint_enable]
