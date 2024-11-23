# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Fetch the repository root
import os

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
except ValueError as e:
    print(e)


"""Configuration for the hand with four fingers."""
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.assets import ArticulationCfg

hand4f_usd = os.path.join(repo_root, "tasks/ihm/assets/hand/usd/hand4f_usd/hand4f.usd")
ur5e_hand4f_usd = os.path.join(repo_root, "tasks/ihm/assets/hand/usd/ur5e_usd/ur5e_hand4f.usd")
hand5f_usd = os.path.join(repo_root, "tasks/ihm/assets/hand/usd/hand5f_usd/hand.usd")


##
# Configuration
##


HAND4F_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=hand4f_usd,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=10.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
            fix_root_link=True,
        ),
        copy_from_source=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        joint_pos={
            "finger1_proximal_joint": 0.0,
            "finger1_middle_joint": 0.25,
            "finger1_distal_joint": 0.3,
            "finger2_proximal_joint": 0.0,
            "finger2_middle_joint": 0.25,
            "finger2_distal_joint": 0.1,
            "finger3_proximal_joint": 0.0,
            "finger3_middle_joint": 0.25,
            "finger3_distal_joint": 0.1,
            "finger4_proximal_joint": 0.0,
            "finger4_middle_joint": 0.25,
            "finger4_distal_joint": 0.1,
        },
    ),
    actuators={
        "robot": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            stiffness=30.0,
            damping=5.0,
        ),
    },
)

UR5E_HAND_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=ur5e_hand4f_usd,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=10.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
            fix_root_link=True,
        ),
        copy_from_source=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        joint_pos={
            "shoulder_pan_joint": 0.0,
            "shoulder_lift_joint": -1.57,
            "elbow_joint": 1.57,
            "wrist_1_joint": -1.57,
            "wrist_2_joint": -1.57,
            "wrist_3_joint": 0.0,
            "finger1_proximal_joint": 0.0,
            "finger1_middle_joint": 0.25,
            "finger1_distal_joint": 0.3,
            "finger2_proximal_joint": 0.0,
            "finger2_middle_joint": 0.25,
            "finger2_distal_joint": 0.1,
            "finger3_proximal_joint": 0.0,
            "finger3_middle_joint": 0.25,
            "finger3_distal_joint": 0.1,
            "finger4_proximal_joint": 0.0,
            "finger4_middle_joint": 0.25,
            "finger4_distal_joint": 0.1,
        },
    ),
    actuators={
        "robot": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            stiffness=3.0,
            damping=0.5,
        ),
    },
)


HAND5F_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=hand5f_usd,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=10.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
            fix_root_link=True,
        ),
        copy_from_source=False,
    ),
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
    ),
    actuators={
        "robot": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            stiffness=30.0,
            damping=5.0,
        ),
    },
)

"""Configuration for the hand with four fingers."""
