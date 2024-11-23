import omni.isaac.lab.envs.mdp as mdp
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg, RigidObjectCfg
from omni.isaac.lab.envs import DirectRLEnvCfg
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.markers import VisualizationMarkersCfg
from omni.isaac.lab.sensors import ContactSensorCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import PhysxCfg, SimulationCfg
from omni.isaac.lab.utils import configclass

from tasks.ihm.config.hand4f import HAND4F_CFG, HAND5F_CFG, UR5E_HAND_CFG


# Fetch the repository root
import os
import sys

def fetch_repo_root(file_path, repo_name):
    # Split the file path into parts
    path_parts = file_path.split(os.sep)

    # Try to find the repository name in the path
    if repo_name in path_parts:
        # Find the index of the repository name
        repo_index = path_parts.index(repo_name)
        # Join the path components up to the repository name
        repo_root = os.sep.join(path_parts[:repo_index + 1])
        return repo_root
    else:
        raise ValueError("Repository name not found in the file path")

try:
    current_file_path = os.path.abspath(__file__)
    repo_name = "diffuse-plan-learn"
    repo_root = fetch_repo_root(current_file_path, repo_name)
    sys.path.append(repo_root)
    print(f"Repository root '{repo_root}' added to Python path.")
except ValueError as e:
    print(e)


@configclass
class EventCfg:
    """Configuration for environment reset and randomization."""
    # -- robot rigid body properties -- #
    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (1.0, 1.0),
            "dynamic_friction_range": (1.0, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 250,
        },
    )
    # -- object rigid body properties -- #
    object_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object", body_names=".*"),
            "static_friction_range": (1.0, 1.0),
            "dynamic_friction_range": (1.0, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 250,
        },
    )

    # reset
    reset_maze_position = EventTerm(
        func=mdp.reset_scene_to_default,
        mode="reset",
    )


@configclass
class InHandManipulationEnvCfg(DirectRLEnvCfg):
    """ Configuration for the IHM base environment """
    # env
    decimation = 1
    episode_length_s = 25  # 500 timesteps
    max_episode_steps = 500
    control_freq_inv = 1

    # sampling space
    num_q_space = 49

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 20,
        render_interval=decimation,
        disable_contact_processing=True,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="max",  # Defaults to 'average'
            restitution_combine_mode="min",  # Defaults to 'average'
            static_friction=1.0,  # Defaults to 0.5
            dynamic_friction=1.0,  # Defaults to 0.5
            restitution=0.0,  # Defaults to 0.0
        ),
        physx=sim_utils.PhysxCfg(gpu_max_rigid_patch_count=50 * 2**15)
    )

    # robot hand
    # robot_cfg: ArticulationCfg = HAND4F_CFG.replace(prim_path="/World/envs/env_.*/Robot", ).replace(
    robot_cfg: ArticulationCfg = HAND5F_CFG.replace(prim_path="/World/envs/env_.*/Robot", ).replace(
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            joint_pos={
                "finger1_proximal_joint": 0.0,
                "finger1_middle_joint": 0.25,
                "finger1_distal_joint": 0.45,
                "finger2_proximal_joint": 0.0,
                "finger2_middle_joint": 0.25,
                "finger2_distal_joint": 0.45,
                "finger3_proximal_joint": 0.0,
                "finger3_middle_joint": 0.25,
                "finger3_distal_joint": 0.45,
                "finger4_proximal_joint": 0.0,
                "finger4_middle_joint": 0.25,
                "finger4_distal_joint": 0.45,
                "finger5_proximal_joint": 0.0,
                "finger5_middle_joint": 0.25,
                "finger5_distal_joint": 0.45,
            },
        )
    )
    actuated_joint_names = [
        "finger1_proximal_joint",
        "finger2_proximal_joint",
        "finger3_proximal_joint",
        "finger4_proximal_joint",
        "finger5_proximal_joint",

        "finger1_middle_joint",
        "finger2_middle_joint",
        "finger3_middle_joint",
        "finger4_middle_joint",
        "finger5_middle_joint",

        "finger1_distal_joint",
        "finger2_distal_joint",
        "finger3_distal_joint",
        "finger4_distal_joint",
        "finger5_distal_joint",
    ]
    fingertip_body_names = [
        "finger1_distal_tip",
        "finger2_distal_tip",
        "finger3_distal_tip",
        "finger4_distal_tip",
        "finger5_distal_tip",
    ]

    # contact force sensors
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*_distal",
        history_length=3,
        update_period=0.0,
        track_air_time=True,
        debug_vis=True,
    )

    # in-hand object
    object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(repo_root, "tasks/ihm/assets/objects/usd/cube_usd/cube.usd"),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                enable_gyroscopic_forces=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0025,
                max_depenetration_velocity=1000.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(
                mass=0.12,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0, 0, 0.245), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    # goal object
    goal_object_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/goal_marker",
        markers={
            "goal": sim_utils.UsdFileCfg(
                usd_path=os.path.join(repo_root, "tasks/ihm/assets/objects/usd/cube_usd/cube.usd"),
                scale=(1.0, 1.0, 1.0),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
            ),
        },
    )

    # contact position marker
    contact_pos_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/contact_pos",
        markers={
            "contact_pos": sim_utils.SphereCfg(
                radius=0.01,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 0.7)),
            ),
        },
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=2.0, replicate_physics=True)

    # contact position marker
    visualize_contact_pos = True

    # default hand and object configuration
    default_hand_joint_pos = [0, 0, 0, 0, 0, 0.25, 0.25, 0.25, 0.25, 0.25, 0.45, 0.45, 0.45, 0.45, 0.45]
    default_object_pos = [0.0, 0.0, 0.245]
    default_object_quat = [1.0, 0.0, 0.0, 0.0]

    # hand joint limits
    finger_dof_lower_limits = [-0.7, -0.7, -0.7, -0.7, -0.7, -0.39269908169, -0.39269908169, -0.39269908169, -0.39269908169, -0.39269908169, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001]
    finger_dof_upper_limits = [0.7, 0.7, 0.7, 0.7, 0.7, 0.5, 0.5, 0.5, 0.5, 0.5, 1, 1, 1, 1, 1]

    # action scale
    action_scale = 0.1

    # reset
    min_object_height = 0.225
    max_object_height = 0.275
    xy_object_lim = 0.03
    num_contact_constraint = False
    required_ncon = 3  # number of contacts required for task success
    contact_loc_constraint = False
    ftip_on_side_constraint = False
    ftip_on_side_constraint_tolerance = 0.04

    # PRM sample space limits
    sample_obj_pos_lower_limit = [-0.03, -0.03, 0.225]
    sample_obj_pos_upper_limit = [0.03, 0.03, 0.275]
    sample_obj_lin_vel_lower_limit = [-0.03, -0.03, -0.2]
    sample_obj_lin_vel_upper_limit = [0.03, 0.03, 0.05]
    sample_obj_ang_vel_lower_limit = [-1.0, -1.0, -1.0]
    sample_obj_ang_vel_upper_limit = [1.0, 1.0, 1.0]
    sample_joint_vel_lower_limit = [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]
    sample_joint_vel_upper_limit = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

    # PRM valid state limits
    valid_obj_pos_lower_limit = [-0.03, -0.03, 0.225]
    valid_obj_pos_upper_limit = [0.03, 0.03, 0.275]
    valid_obj_lin_vel_lower_limit = [-0.1, -0.1, -0.3]
    valid_obj_lin_vel_upper_limit = [0.1, 0.1, 0.1]
    valid_obj_ang_vel_lower_limit = [-2.0, -2.0, -2.0]
    valid_obj_ang_vel_upper_limit = [2.0, 2.0, 2.0]
    valid_joint_vel_lower_limit = [-3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0]
    valid_joint_vel_upper_limit = [3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0]

    # sampling validity check
    valid_contact_force_upper_limit = 10.0
    valid_req_valid_contact = True  # all contacts must be on the inside of fingers for task success
    valid_req_ncon_lim = True  # must be > n_contact_min contacts for task success
    valid_required_ncon = 3  # number of contacts required for task success

    # domain reset and randomization config
    events: EventCfg = EventCfg()


    ### Parameters for the planning task ###

    # reward
    planner_delta_clip = 0.02  # clip the delta displacement between the current and target object position
    planner_taskCompletionReward = 8.0  # reward for task completion
    planner_deltaReward = 2.0  # reward for changes in object rotation
    planner_outOfBoundsPenalty = 0.0  # penalty for object going out of bounds
    planner_ftipObjDispPenalty = 0.0  # penalty for displacement between fingertip and object
    planner_numContactPenalty = 1.0  # scalar penalty applied when n_contact < n_contact_min
    planner_required_ncon = 3  # number of contacts required for task success
    planner_objPositionPenalty = 1.0  # penalty for object position deviation from default
    planner_contactLocationPenalty = 1.0  # penalty for contact on back of finger

    # success
    planner_angle_threshold = 0.4  # angular tolerance for successful reorientation
    planner_joint_vel_threshold = 0.8  # Tao uses 0.25 but we have more DOF than they do
    planner_object_vel_threshold = 0.04  # object linear velocity threshold for task success
    planner_object_angvel_threshold = 0.5  # object angular velocity threshold for task success
    planner_req_valid_contact = True  # all contacts must be on the inside of fingers for task success
    planner_req_ncon_lim = True  # must be > n_contact_min contacts for task success

    # reset
    planner_num_contact_constraint = False
    planner_contact_loc_constraint = False
    planner_ftip_on_side_constraint = False
    planner_ftip_on_side_constraint_tolerance = 0.04

    # goal extraction config for PRM
    extracted_goal_idx_state = (39, 43)  # indices of the current state in observation (obj_rot)

    def __post_init__(self) -> None:
        """Post initialization."""
        # set up the viewer
        self.viewer.eye = (-0.5, -0.5, 0.3)
        self.viewer.lookat = (0.0, 0.0, 0.2)



@configclass
class ArbitraryReorientEnvCfg(InHandManipulationEnvCfg):
    """
    Configuration for the IHM Arbitrary Reorient environment.

    The configuration below would be overwritten by the configuration in the task yaml config file.
    """
    # env
    num_actions = 12
    num_observations = 39
    num_states = 77

    # sample target rotation within tilt limit of the Finger-gaiting task
    sample_target_rot_within_tilt = True
    rotation_axis = 2  # 0: X, 1: Y, 2: Z
    object_tilt_lim = 0.8

    # reward
    delta_clip = 0.02  # clip the delta displacement between the current and target object position
    taskCompletionReward = 800.0  # reward for task completion
    deltaReward = 100.0  # reward for changes in object rotation
    outOfBoundsPenalty = 0.0  # penalty for object going out of bounds
    ftipObjDispPenalty = 0.0  # penalty for displacement between fingertip and object
    numContactPenalty = 1.0  # scalar penalty applied when n_contact < n_contact_min
    objPositionPenalty = 1.0  # penalty for object position deviation from default
    contactLocationPenalty = 1.0  # penalty for contact on back of finger

    # success
    angle_threshold = 0.4  # angular tolerance for successful reorientation
    joint_vel_threshold = 0.8  # Tao uses 0.25 but we have more DOF than they do
    object_vel_threshold = 0.04  # object linear velocity threshold for task success
    object_angvel_threshold = 0.5  # object angular velocity threshold for task success
    req_valid_contact = True  # all contacts must be on the inside of fingers for task success
    req_ncon_lim = True  # must be > n_contact_min contacts for task success

    # reset
    ftip_on_side_constraint_tolerance = 0.04

    # goal extraction config for PRM
    extracted_goal_idx_policy = (31, 35)  # indices of the current state in observation (obj_rot)
    goal_idx_policy = (35, 39)  # indices of the goal state in observation (goal_rot)
    extracted_goal_idx_critic = (39, 43)  # indices of the current state in critic observation (obj_rot)
    goal_idx_critic = (73, 77)  # indices of the goal state in critic observation (goal_rot)




@configclass
class FingerGaitingEnvCfg(InHandManipulationEnvCfg):
    """
    Configuration for the IHM Finger Gaiting environment.

    The configuration below would be overwritten by the configuration in the task yaml config file.
    """
    # env
    num_actions = 12
    num_observations = 28
    num_states = 73

    # rotation axis for the object
    rotation_axis = 2  # 0: X, 1: Y, 2: Z

    # reward
    ang_vel_clip_max = 0.5
    ang_vel_clip_min = -0.5
    rotation_reward_scale = 1.0
    object_lin_vel_penalty_scale = -0.3
    pose_diff_penalty_scale = -0.0
    torque_penalty_scale = -0.0
    work_penalty_scale = 0.0
    ftip_obj_disp_pen = 0.0
    out_of_bounds_pen = 0.0

    # reset
    ftip_on_side_constraint_tolerance = 0.04  # needed for 10x8x8x4_L (0.035) if states from tree span a large z range
    object_tilt_lim = 0.8