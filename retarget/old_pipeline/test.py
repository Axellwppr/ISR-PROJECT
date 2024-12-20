import os
import numpy as np
from isaacgym import gymapi
from isaacgym import gymutil

# Initialize Isaac Gym
gym = gymapi.acquire_gym()

# Simulation parameters
sim_params = gymapi.SimParams()
sim_params.dt = 0.005
sim_params.substeps = 1
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)

# PhysX settings
sim_params.physx.num_threads = 4
sim_params.physx.solver_type = 1  # 0: pgs, 1: tgs
sim_params.physx.num_position_iterations = 4
sim_params.physx.num_velocity_iterations = 1
sim_params.physx.contact_offset = 0.02
sim_params.physx.rest_offset = 0.02
sim_params.physx.bounce_threshold_velocity = 0.2
sim_params.physx.max_depenetration_velocity = 1.0
sim_params.physx.max_gpu_contact_pairs = 16777210
sim_params.physx.default_buffer_size_multiplier = 10.0
# sim_params.physx.contact_collection = gymapi.ContactCollectionMode.ALL

# Create simulation
sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
if sim is None:
    raise Exception("Failed to create sim")

# Add ground plane
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1)
gym.add_ground(sim, plane_params)

# Create viewer
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    raise Exception("Failed to create viewer")

# Load robot asset
asset_root = (
    "/home/axell/desktop/11-30-ISR-PROJ/human2humanoid/resources/robots/rel2/urdf/"
)
asset_file = "rel2.urdf"
asset_options = gymapi.AssetOptions()
asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
asset_options.collapse_fixed_joints = True
asset_options.override_com = True
asset_options.override_inertia = True
humanoid_asset = gym.load_asset(sim, asset_root, asset_file, asset_options)
if humanoid_asset is None:
    raise Exception("Failed to load humanoid asset")

# Set environment spacing
envs = []
env_spacing = 2.0
num_envs = 1
lower = gymapi.Vec3(-env_spacing, -env_spacing, 0.0)
upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)

# Initialize environments
humanoid_actors = []
dof_targets = []

dof_lower_limits = []
dof_upper_limits = []

for i in range(num_envs):
    env = gym.create_env(sim, lower, upper, num_envs)
    envs.append(env)

    humanoid_pose = gymapi.Transform()
    humanoid_pose.p = gymapi.Vec3(0.0, 0.0, 10.0)
    humanoid_actor = gym.create_actor(
        env, humanoid_asset, humanoid_pose, "humanoid", i, 0
    )
    humanoid_actors.append(humanoid_actor)

    # Set DOF properties for POS control
    dof_props = gym.get_actor_dof_properties(env, humanoid_actor)
    dof_props["driveMode"].fill(gymapi.DOF_MODE_POS)
    dof_props["stiffness"].fill(350.0)  # Adjust stiffness for smoother control
    dof_props["damping"].fill(2.0)
    gym.set_actor_dof_properties(env, humanoid_actor, dof_props)

    # Get DOF limits
    asset_dof_props = gym.get_asset_dof_properties(humanoid_asset)
    dof_lower_limits.append(asset_dof_props["lower"])
    dof_upper_limits.append(asset_dof_props["upper"])

    # Initialize target positions
    dof_targets.append(np.zeros(len(asset_dof_props["lower"]), dtype=np.float32))

# Simulation loop
print("Starting simulation with random target position control...")
while not gym.query_viewer_has_closed(viewer):
    # Handle user input
    gym.poll_viewer_events(viewer)

    for i, env in enumerate(envs):
        # Generate random target positions within limits
        lower_limits = dof_lower_limits[i]
        upper_limits = dof_upper_limits[i]
        dof_targets[i] = np.random.uniform(lower_limits, upper_limits).astype(
            np.float32
        )

        # Apply target positions
        gym.set_actor_dof_position_targets(env, humanoid_actors[i], dof_targets[i])

    # Step simulation
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # Update viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)

    # Sync frame time
    gym.sync_frame_time(sim)

# Cleanup resources
gym.destroy_sim(sim)
