import torch
import numpy as np
from time import sleep

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation, ArticulationCfg, AssetBaseCfg, RigidObject
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.envs.common import VecEnvStepReturn
from omni.isaac.lab.sensors import ContactSensor
from omni.isaac.lab.markers import VisualizationMarkers
from omni.isaac.lab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from omni.isaac.lab.utils.math import quat_conjugate, quat_mul, sample_uniform

from tasks.ihm.ihm_env import IHMEnv, randomize_rotation, compute_quat_angle, torch_rand_float, generate_quaternions_within_tilt, my_quat_rotate, quat_to_angle_axis
from tasks.ihm.ihm_env_cfg import ArbitraryReorientEnvCfg
from utils.misc import AverageScalarMeter, to_torch


class ArbitraryReorientEnv(IHMEnv):
    cfg: ArbitraryReorientEnvCfg

    def __init__(self, cfg: DirectRLEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # create angular distance buffer
        self.ang_dist = torch.tensor(
            [torch.nan] * self.num_envs,
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(-1)
        # create previous angular distance buffer
        self.prev_ang_dist = torch.tensor(
            [torch.nan] * self.num_envs,
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(-1)

        # goal orientations
        self.goal = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        self.goal[:, 0] = 1.0
        self.goal_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.goal_pos[:, :] = torch.tensor([0.2, 0.2, 0.40], device=self.device)
        # initialize goal marker
        self.goal_markers = VisualizationMarkers(self.cfg.goal_object_cfg)
        self.goal_dim = 4

        if self.cfg.sample_target_rot_within_tilt == True:
            # The task configuration requires the target rotation to be sampled within the tilt limit
            # Set up rotation axis for the object
            self.rotation_axis = torch.zeros((self.num_envs, 3), device=self.device)
            self.rotation_axis[:, self.cfg.rotation_axis] = 1

        # initialize contact position marker
        self.visualize_contact_pos = self.cfg.visualize_contact_pos
        if self.visualize_contact_pos:
            self.contact_markers = VisualizationMarkers(self.cfg.contact_pos_cfg)

        # Logging success rate for each criteria
        self.success_rate_mode = "eval"
        # split the success rate into train and eval
        self.train_success_rot = torch.zeros_like(self.reset_buf)
        self.train_success_rate_rot = AverageScalarMeter(100)
        self.extras["train_success_rate_rot"] = 0.0
        self.eval_success_rot = torch.zeros_like(self.reset_buf)
        self.eval_success_rate_rot = AverageScalarMeter(100)
        self.extras["eval_success_rate_rot"] = 0.0

    def _get_observations(self) -> dict:
        """ Compute and return the observations for the environment.

        Returns:
            The observations (key: "policy") and states ((key: "critic") for the environment.
        """
        # Refresh the intermediate values after the physics steps
        self._compute_intermediate_values()

        # Concatenate the contact forces on each fingertip
        contact_force = torch.cat(self.fingertip_contact_force, dim=1)
        ftip_contact_pos = torch.cat(self.ftip_contact_pos, dim=1)

        # Observation
        obs_policy = torch.cat(
            (
                self.hand.data.joint_pos.view(self.num_envs, -1),  # joint positions
                self.target_hand_joint_pos,  # target joint positions
                self.contact_bool_tensor,  # fingertip contact bool
                self.object_pos,  # object position
                self.object_orientation,  # object orientation
                self.goal,  # goal orientation
            ),
            dim=-1,
        )
        # States
        obs_critic = torch.cat(
            (
                self.hand.data.joint_pos.view(self.num_envs, -1),  # joint positions
                self.target_hand_joint_pos,  # target joint positions
                contact_force,  # fingertip contact force
                self.object_pos,  # object position
                self.object_orientation,  # object orientation
                self.object_lin_vel,  # object linear velocity
                self.object_ang_vel,  # object angular velocity
                ftip_contact_pos,  # fingertip contact position
                self.torque,  # joint torques
                self.goal,  # goal orientation
            ),
            dim=-1,
        )
        return {"policy": obs_policy, "critic": obs_critic}

    def _get_rewards(self) -> torch.Tensor:
        """ Compute and return the rewards for the environment.

        Returns:
            The rewards for the environment. Shape is (num_envs,).
        """
        # Refresh the intermediate values after the physics steps
        self._compute_intermediate_values()

        # update object angular distance
        self.ang_dist = compute_quat_angle(self.goal, self.object_orientation)
        # compute dense reward
        reward = self.compute_delta_reward()

        # compute out-of-bounds penalty
        oob = self.get_pos_constrants()
        reward += -1 * self.cfg.outOfBoundsPenalty * oob.unsqueeze(1)

        # compute sparse reward
        success = self.check_success()
        reward += self.cfg.taskCompletionReward * success.unsqueeze(1)

        # compute ftip-object distance penalty
        total_ftip_disp = self.compute_ftip_obj_disp()
        reward += -1 * self.cfg.ftipObjDispPenalty * total_ftip_disp.unsqueeze(1)

        # penalize for failing to reach contact minimum
        nconpen = -1 * self.cfg.numContactPenalty * self.ncon_constraint().float()
        reward += nconpen.unsqueeze(1)

        # penalize for object position deviation
        obj_position_dev = torch.linalg.norm(self.object_pos - self.object_default_pos, dim=-1) ** 2
        obj_position_pen = -1 * self.cfg.objPositionPenalty * obj_position_dev
        reward += obj_position_pen.unsqueeze(1)

        # penalize for contact on back of finger
        contact_loc_pen = -1 * self.cfg.contactLocationPenalty * self.contact_location_constraint().float()
        reward += contact_loc_pen.unsqueeze(1)

        # update previous angular distance
        self.prev_ang_dist[:] = self.ang_dist.clone()

        return reward.squeeze(-1)

    def compute_delta_reward(self):
        """ compute dense orientation reward based on change in displacement to target """
        # fetch previous displacement, accounting for the case where ep was reset (nan)
        nan_indices = torch.argwhere(torch.isnan(self.prev_ang_dist).float()).squeeze(-1)
        if len(nan_indices) > 0:
            self.prev_ang_dist[nan_indices] = self.ang_dist[nan_indices]
        # compute delta displacement
        prev_ang_dist = self.prev_ang_dist.clone()
        delta_displacement = torch.clip(self.ang_dist - prev_ang_dist, -self.cfg.delta_clip, self.cfg.delta_clip)
        # orientation reward
        reward = -1 * self.cfg.deltaReward * delta_displacement
        return reward

    def compute_reward_in_walks(self, expand_state) -> torch.Tensor:
        """
        Compute the reward for the state in the PRM walks.

        expand_state[:, 0: self.self.num_joints]: joint position
        expand_state[:, self.num_joints: 2 * self.num_joints]: target joint position
        expand_state[:, 2 * self.num_joints: 3 * self.num_joints]: joint velocity
        expand_state[:, 3 * self.num_joints: 3 * self.num_joints + 3]: object position
        expand_state[:, 3 * self.num_joints + 3: 3 * self.num_joints + 7]: object orientation
        expand_state[:, 3 * self.num_joints + 7: 3 * self.num_joints + 13]: object linear and angular velocity
        expand_state[:, 3 * self.num_joints + 13: 3 * self.num_joints + 16]: previous object orientation
        """
        pass


    def check_success(self):
        """
            Task is complete if the object lies within the angular threshold,
            and all motion criteria are met.
        """
        success = torch.ones_like(self.reset_buf)
        # object orientation criteria
        ang_dist = self.ang_dist.clone().squeeze(-1)
        orientation_crit = torch.where(
            ang_dist < self.cfg.angle_threshold,
            torch.ones_like(self.reset_buf),
            torch.zeros_like(self.reset_buf)
        )
        success = torch.logical_and(success, orientation_crit)
        # finger motion criteria
        q_dot = torch.linalg.norm(self.hand_dof_vel.clone(), dim=-1)
        q_dot_crit = torch.where(
            q_dot < self.cfg.joint_vel_threshold,
            torch.ones_like(self.reset_buf),
            torch.zeros_like(self.reset_buf)
        )
        success = torch.logical_and(success, q_dot_crit)
        # object motion criteria, linear velocity
        object_lin_vel = torch.linalg.norm(self.object_lin_vel, dim=-1)
        object_lin_vel_crit = torch.where(
            object_lin_vel < self.cfg.object_vel_threshold,
            torch.ones_like(self.reset_buf),
            torch.zeros_like(self.reset_buf)
        )
        success = torch.logical_and(success, object_lin_vel_crit)
        # object motion criteria, angular velocity
        object_ang_vel = torch.linalg.norm(self.object_ang_vel, dim=-1)
        object_ang_vel_crit = torch.where(
            object_ang_vel < self.cfg.object_angvel_threshold,
            torch.ones_like(self.reset_buf),
            torch.zeros_like(self.reset_buf)
        )
        success = torch.logical_and(success, object_ang_vel_crit)
        # contact location criteria (all contacts must be on the inside of the fingers)
        if self.cfg.req_valid_contact:
            invalid_contact = self.contact_location_constraint()
            valid_contact = torch.logical_not(invalid_contact)
            success = torch.logical_and(success, valid_contact)
        # contact count criteria (at least 1 contact must be made in 4fhand)
        if self.cfg.req_ncon_lim:
            invalid_ncon = self.ncon_constraint()
            valid_ncon = torch.logical_not(invalid_ncon)
            success = torch.logical_and(success, valid_ncon)

        return success

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """ Compute and return the done flags for the environment.

        Returns:
            A tuple containing the done flags for termination and time-out.
            Shape of individual tensors is (num_envs,).
        """
        # Refresh the intermediate values after the physics steps
        self._compute_intermediate_values()

        # Compute the constraints in object positions
        done = self.get_pos_constrants()

        # Check success criteria
        self.success = self.check_success().int()
        # Update success rate
        if self.success_rate_mode == "train":
            self.train_success_rot = self.check_success().int()
        elif self.success_rate_mode == "eval":
            self.eval_success_rot = self.check_success().int()

        success_idx = self.success.nonzero(as_tuple=False).squeeze(-1)
        # reset goals if the goal has been reached
        if len(success_idx) > 0:
            self._reset_target_pose(success_idx)

        # If task completion should trigger reset, reset the environments that are successful
        done = torch.where(
            self.success.bool(),
            torch.ones_like(self.reset_buf),
            done
        )

        # Check if the contact is made on the front of the fingertips
        if self.cfg.ftip_on_side_constraint:
            done = torch.where(
                self.fingertip_on_side_constraint(
                    upper_lim=self.cfg.ftip_on_side_constraint_tolerance,
                    lower_lim=-1 * self.cfg.ftip_on_side_constraint_tolerance
                ),
                torch.ones_like(self.reset_buf),
                done,
            )
        # Check the number of contacts
        if self.cfg.num_contact_constraint:
            done = torch.where(
                self.ncon_constraint(),
                torch.ones_like(self.reset_buf),
                done,
            )
        # Check if the contact location is valid
        if self.cfg.contact_loc_constraint:
            done = torch.where(
                self.contact_location_constraint(),
                torch.ones_like(self.reset_buf),
                done,
            )

        # Check the time limit
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        return done, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None) -> None:
        """ Reset environments based on specified indices.

        Args:
            env_ids: List of environment ids which must be reset
        """
        # Reset the angular distance buffer
        self.prev_ang_dist[env_ids] = torch.nan

        # Call the parent reset function defined in IHM class
        super()._reset_idx(env_ids)

        # Reset goals
        self._reset_target_pose(env_ids)

        # Update success rate
        if self.success_rate_mode == "train":
            self.train_success_rate_rot.update(self.train_success_rot[env_ids])
            self.extras["train_success_rate_rot"] = self.train_success_rate_rot.get_mean()
            self.train_success_rot[env_ids] = 0.0

        elif self.success_rate_mode == "eval":
            self.eval_success_rate_rot.update(self.eval_success_rot[env_ids])
            self.extras["eval_success_rate_rot"] = self.eval_success_rate_rot.get_mean()
            self.eval_success_rot[env_ids] = 0.0

    def _reset_target_pose(self, env_ids):
        if self.cfg.sample_target_rot_within_tilt == True:
            # goal rotation must satisfy the tilt limit
            sampled_goal_rot = generate_quaternions_within_tilt(
                self.rotation_axis[0].cpu(),
                self.cfg.object_tilt_lim,
                num_samples=env_ids.size(0),
            ).to(self.device)
        else:
            # reset goal rotation
            rand_floats = sample_uniform(-1.0, 1.0, (len(env_ids), 2), device=self.device)
            sampled_goal_rot = randomize_rotation(
                rand_floats[:, 0], rand_floats[:, 1], self.x_unit_tensor[env_ids], self.y_unit_tensor[env_ids]
            )
        # update goal pose and markers
        self.goal[env_ids] = sampled_goal_rot
        goal_pos = self.goal_pos + self.scene.env_origins
        self.goal_markers.visualize(goal_pos, self.goal)


    """ PRM Planner functions """


    def set_goal(self, goal, env_ids: torch.Tensor):
        """ Set the goal state for the environment and update the goal marker """
        self.goal[env_ids, :] = goal
        goal_pos = self.goal_pos + self.scene.env_origins
        self.goal_markers.visualize(goal_pos, self.goal)

    def q_to_goal(self, q: torch.Tensor) -> torch.Tensor:
        """ Extract object rotation from q_state to serve as goal """
        goal = q[:, self.num_joints * 3 + 3: self.num_joints * 3 + 7]  # extract goal rotation
        return goal

    def compute_goal_distance(self, prm_nodes: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        """
            Computes distance in goal state from a specific node to each node in node set.
        """
        # Extract object rotation from goal
        nodes_quat = prm_nodes[:, self.num_joints * 3 + 3: self.num_joints * 3 + 7].view(-1, 4)
        goal_quat = goal[self.num_joints * 3 + 3: self.num_joints * 3 + 7]
        # Compute angular distance
        distances = compute_quat_angle(goal_quat, nodes_quat)
        return distances

    def visualize_walks(self, walks: torch.Tensor) -> None:
        """ Visualize the PRM walks """
        for i in range(walks.shape[0]):
            print("---Walk: ", i, "---")
            # Emtpy walk
            if walks[i, 0, 0] == float('-inf'):
                continue

            # find the valid length of the walk
            indices = (walks[i, :, 0] == -float('inf')).nonzero(as_tuple=False)
            if indices.size(0) > 0:
                valid_len = indices[0][0]
            else:
                valid_len = walks.size(1)

            # Get the start and end object rotation
            start_rot = walks[i, 0, self.num_joints * 3 + 3: self.num_joints * 3 + 7]
            reached_rot = walks[i, valid_len - 1, self.num_joints * 3 + 3: self.num_joints * 3 + 7]
            quat_diff = quat_mul(reached_rot, quat_conjugate(start_rot))
            magnitude, axis = quat_to_angle_axis(quat_diff)
            magnitude = magnitude - 2 * np.pi * (magnitude > np.pi)
            axis_angle = torch.mul(axis, torch.reshape(magnitude, (-1, 1)))
            rotation_axis_reshaped = self.rotation_axis[0: axis_angle.size(0), :]
            angle = (axis_angle * rotation_axis_reshaped).sum(-1)
            print("Length of this walk:", valid_len)
            print("Angular distance between start and end:", angle)

            # set goal as the last valid state in the walk
            reached_state = self.goal.clone()
            reached_state[0, :] = walks[i, valid_len - 1, self.num_joints * 3 + 3: self.num_joints * 3 + 7]
            self.set_goal(reached_state, torch.tensor(list(range(self.num_envs)), device=self.device))

            # set the state in the walk to Environment 0
            for j in range(valid_len):
                # set the current state
                current_state = walks[i, j, :]
                with torch.inference_mode():
                    self.set_env_states(current_state.unsqueeze(0), torch.tensor([0], device=self.device))
                    self.simulate()
                    sleep(0.5)

    def save_episode_context(self):
        """ Saves episode context to switch to planner """
        context = super().save_episode_context()
        context["prev_ang_dist"] = self.prev_ang_dist.clone()
        self.clear_prev_ang_dist()
        return context

    def restore_episode_context(self, context):
        """ Restore episode context from planning to learning """
        self.prev_ang_dist = context["prev_ang_dist"]
        return super().restore_episode_context(context)

    def clear_prev_ang_dist(self):
        self.prev_ang_dist = torch.tensor(
            [torch.nan] * self.num_envs,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(-1)

    def sample_q(self, num_samples) -> torch.Tensor:
        if self.cfg.sample_target_rot_within_tilt == True:
            # Sample target rotation within tilt limit
            # sample random joint positions and object position unifromly
            alpha = torch_rand_float(
                0.0,
                1.0,
                (num_samples, self.pos_sample_space_dim),
                device=self.device,
            )
            x_pos_sample = alpha * self.pos_sample_space_upper + (1 - alpha) * self.pos_sample_space_lower
            # sample random object orientation within the tilt limit from the rotation axis
            x_rot_sample_within_tilt = generate_quaternions_within_tilt(
                self.rotation_axis[0].cpu(),
                self.cfg.object_tilt_lim,
                num_samples=num_samples,
            ).to(self.device)
            # sample random joint velocities, object linear and angular velocities
            alpha_dot = torch_rand_float(
                0.0,
                1.0,
                (num_samples, self.vel_sample_space_dim),
                device=self.device,
            )
            x_vel_sample = alpha_dot * self.vel_sample_space_upper + (1 - alpha_dot) * self.vel_sample_space_lower

            # apply the sampled states to the environment
            x_samples = torch.cat(
                [
                    x_pos_sample[:, :self.num_joints],  # joint position
                    x_pos_sample[:, :self.num_joints],  # target joint position
                    x_vel_sample[:, :self.num_joints],  # joint velocity
                    x_pos_sample[:, self.num_joints: self.num_joints + 3],  # object position
                    x_rot_sample_within_tilt,  # object rotation
                    x_vel_sample[:, self.num_joints: self.num_joints + 6],  # object linear and angularvelocity
                ],
                dim=1
            )

            return x_samples
        else:
            return super().sample_q(num_samples)

    # def is_invalid(self, debug: bool = False) -> torch.Tensor:
    #     invalid, x_start_prime = super().is_invalid(debug)
    #     # Check the object orientation tilt limit if the target rotation is sampled within the tilt limit
    #     if self.cfg.sample_target_rot_within_tilt == True:
    #         # Check if the object is tilted within the limit
    #         obj_axis = my_quat_rotate(self.object_orientation, self.rotation_axis)
    #         invalid_obj_quat = torch.linalg.norm(obj_axis * self.rotation_axis, dim=1) < self.cfg.object_tilt_lim
    #         num_invalid_obj_quat = torch.sum(invalid_obj_quat)
    #         invalid = torch.logical_or(invalid_obj_quat, invalid)
    #         if debug:
    #             print("Invalid Object Quaternions:", num_invalid_obj_quat)
    #     return invalid, x_start_prime

    def is_invalid(self, debug: bool = False) -> (torch.Tensor, torch.Tensor):
        """ Check if the sampled state is valid to be added to the graph """
        self._compute_intermediate_values()

        invalid = torch.zeros_like(self.reset_buf)
        # object position constraint
        invalid_object_pos = torch.logical_or(
            torch.any(self.object_pos < self.valid_obj_pos_lower_limits, dim=1),
            torch.any(self.object_pos > self.valid_obj_pos_upper_limits, dim=1),
        )
        invalid = torch.logical_or(invalid, invalid_object_pos)

        # step the environment with 50 zero actions
        for k in range(5):
            zero_action = self.zero_actions()
            with torch.inference_mode():
                self.step_without_reset(zero_action)
        self._compute_intermediate_values()

        # check the object position constraint again
        invalid_object_pos_after = torch.logical_or(
            torch.any(self.object_pos < self.valid_obj_pos_lower_limits, dim=1),
            torch.any(self.object_pos > self.valid_obj_pos_upper_limits, dim=1),
        )
        invalid = torch.logical_or(invalid, invalid_object_pos_after)
        # object linear velocity constraint
        invalid_object_lin_vel = torch.logical_or(
            torch.any(self.object_lin_vel < self.valid_obj_lin_vel_lower_limits, dim=1),
            torch.any(self.object_lin_vel > self.valid_obj_lin_vel_upper_limits, dim=1),
        )
        invalid = torch.logical_or(invalid, invalid_object_lin_vel)
        # check if the object is tilted within the limit
        if self.cfg.sample_target_rot_within_tilt == True:
            obj_axis = my_quat_rotate(self.object_orientation, self.rotation_axis)
            invalid_obj_quat = torch.linalg.norm(obj_axis * self.rotation_axis, dim=1) < self.cfg.object_tilt_lim
            invalid = torch.logical_or(invalid_obj_quat, invalid)
        # contact location constraint
        if self.cfg.valid_req_valid_contact:
            invalid_contact = self.contact_location_constraint()
            invalid = torch.logical_or(invalid, invalid_contact)
        # contact number constraint
        if self.cfg.valid_req_ncon_lim:
            invalid_ncon = self.ncon_constraint(min_con=self.cfg.valid_required_ncon)
            invalid = torch.logical_or(invalid, invalid_ncon)
        # maximum force constraint on each fingertip
        invalid_ftip_force = torch.zeros_like(self.reset_buf)
        for i, ftip_contact_force in enumerate(self.fingertip_contact_force):
            force_magnitude = torch.linalg.norm(ftip_contact_force, dim=1)
            ftip_in_collision = force_magnitude > self.cfg.valid_contact_force_upper_limit
            invalid_ftip_force = torch.logical_or(invalid_ftip_force, ftip_in_collision)
            invalid = torch.logical_or(invalid, ftip_in_collision)

        x_start_prime = self.get_env_states()

        # print("Stable state percentage: ", 1.0 - (torch.sum(invalid).item() / self.num_envs))

        return invalid, x_start_prime