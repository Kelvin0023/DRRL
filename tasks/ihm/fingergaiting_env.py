from time import sleep
import torch
import numpy as np

from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.utils.math import quat_mul, quat_conjugate, sample_uniform

from tasks.ihm.ihm_env import IHMEnv, quat_to_angle_axis, my_quat_rotate, torch_rand_float, randomize_rotation, generate_quaternions_within_tilt
from tasks.ihm.ihm_env_cfg import FingerGaitingEnvCfg


class FingerGaitingEnv(IHMEnv):
    cfg: FingerGaitingEnvCfg

    def __init__(self, cfg: DirectRLEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Set up rotation axis for the object
        self.rotation_axis = torch.zeros((self.num_envs, 3), device=self.device)
        self.rotation_axis[:, self.cfg.rotation_axis] = 1

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
                # self.object_pos,  # object position
                # self.object_orientation,  # object orientation
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

        # quat_diff = quat_mul(self.object_orientation, quat_conjugate(self.prev_object_orientation))
        # magnitude, axis = quat_to_angle_axis(quat_diff)
        # magnitude = magnitude - 2 * np.pi * (magnitude > np.pi)
        #
        # axis_angle = torch.mul(axis, torch.reshape(magnitude, (-1, 1)))
        # avg_angular_vel = axis_angle / (0.05 * self.cfg.control_freq_inv)  # 0.05 is the dt
        # # function expects inputs of varying size --> need to account for this
        # rotation_axis_reshaped = self.rotation_axis[0: avg_angular_vel.size(0), :]
        # vec_dot = (avg_angular_vel * rotation_axis_reshaped).sum(-1)

        vec_dot = self.object_ang_vel[:, 2]
        rotation_reward = torch.clip(vec_dot, max=self.cfg.ang_vel_clip_max)
        rotation_reward = self.cfg.rotation_reward_scale * rotation_reward
        # Correct the rotation reward for the first and second steps
        rotation_reward = rotation_reward * (torch.logical_and(self.episode_length_buf != 0, self.episode_length_buf != 1))

        # linear velocity penalty
        object_lin_vel_penalty = torch.norm(self.object_lin_vel, p=1, dim=-1)
        # torque penalty
        torque_penalty = (self.torque ** 2).sum(-1)
        work_penalty = ((self.torque * self.hand_dof_vel).sum(-1)) ** 2
        # pose difference penalty
        pose_diff_penalty = ((self.default_hand_joint_pos - self.hand_dof_pos) ** 2).sum(-1)
        object_pose_diff_penalty = ((self.object_default_pos - self.object_pos) ** 2).sum(-1)
        # ftip obj displacement penalty
        ftip_obj_disp = self.compute_ftip_obj_disp()
        # out-of-bounds penalty
        oob = self.get_pos_constrants()

        reward = rotation_reward
        reward = reward + object_lin_vel_penalty * self.cfg.object_lin_vel_penalty_scale
        reward = reward + pose_diff_penalty * self.cfg.pose_diff_penalty_scale
        reward = reward + object_pose_diff_penalty * self.cfg.pose_diff_penalty_scale
        reward = reward + torque_penalty * self.cfg.torque_penalty_scale
        reward = reward + work_penalty * self.cfg.work_penalty_scale
        reward = reward + -1 * ftip_obj_disp * self.cfg.ftip_obj_disp_pen
        reward = reward + -1 * oob * self.cfg.out_of_bounds_pen

        # Update previous object orientation
        self.prev_object_orientation = self.object_orientation.detach().clone()

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

        object_orientation = expand_state[:, 3 * self.num_joints: 3 * self.num_joints + 7]
        object_lin_vel = expand_state[:, 3 * self.num_joints + 7: 3 * self.num_joints + 10]
        prev_object_orientation = expand_state[:, 3 * self.num_joints + 13: 3 * self.num_joints + 16]


        quat_diff = quat_mul(object_orientation, quat_conjugate(prev_object_orientation))
        magnitude, axis = quat_to_angle_axis(quat_diff)
        magnitude = magnitude - 2 * np.pi * (magnitude > np.pi)

        axis_angle = torch.mul(axis, torch.reshape(magnitude, (-1, 1)))
        avg_angular_vel = axis_angle / (0.05 * self.cfg.control_freq_inv)  # 0.05 is the dt
        # function expects inputs of varying size --> need to account for this
        rotation_axis_reshaped = self.rotation_axis[0: avg_angular_vel.size(0), :]
        vec_dot = (avg_angular_vel * rotation_axis_reshaped).sum(-1)
        rotation_reward = torch.clip(vec_dot, max=self.cfg.ang_vel_clip_max)
        rotation_reward = self.cfg.rotation_reward_scale * rotation_reward

        # linear velocity penalty
        object_lin_vel_penalty = torch.norm(object_lin_vel, p=1, dim=-1)

        reward = rotation_reward
        reward = reward + object_lin_vel_penalty * self.cfg.object_lin_vel_penalty_scale

        return reward

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

            obj_axis = my_quat_rotate(self.object_orientation, self.rotation_axis)
            done = torch.where(
                torch.linalg.norm(obj_axis * self.rotation_axis, dim=1) < self.cfg.object_tilt_lim,
                torch.ones_like(self.reset_buf),
                done,
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

    def save_episode_context(self):
        """ Saves episode context to switch to planner """
        context = super().save_episode_context()
        context["prev_object_orientation"] = self.prev_object_orientation.clone()
        self.clear_prev_object_orientation()
        return context

    def restore_episode_context(self, context):
        """ Restore episode context from planning to learning """
        self.prev_object_orientation = context["prev_object_orientation"]
        return super().restore_episode_context(context)

    def clear_prev_object_orientation(self):
        self.prev_object_orientation = self.object_orientation.detach().clone()


    """ PRM Planner functions """


    def sample_q(self, num_samples) -> torch.Tensor:
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

    def compute_goal_distance(self, prm_nodes: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        """
            Computes the rotation around the rotation axis
        """
        # Extract object rotation from goal
        nodes_quat = prm_nodes[:, self.num_joints * 3 + 3: self.num_joints * 3 + 7].view(-1, 4)
        goal_quat = goal[self.num_joints * 3 + 3: self.num_joints * 3 + 7].repeat(prm_nodes.size(0), 1)
        # Compute angular distance
        quat_diff = quat_mul(goal_quat, quat_conjugate(nodes_quat))
        magnitude, axis = quat_to_angle_axis(quat_diff)
        # magnitude = magnitude - 2 * np.pi * (magnitude > np.pi)
        axis_angle = torch.mul(axis, torch.reshape(magnitude, (-1, 1)))
        rotation_axis_reshaped = self.rotation_axis[0, :].repeat(axis_angle.size(0), 1).cpu()
        angle = (axis_angle * rotation_axis_reshaped).sum(-1)
        return angle

    # def is_invalid(self, debug: bool = False) -> (torch.Tensor, torch.Tensor):
    #     """ Check if the sampled state is valid to be added to the graph """
    #     invalid, x_start_prime = super().is_invalid(debug)
    #     # Check if the object is tilted within the limit
    #     obj_axis = my_quat_rotate(self.object_orientation, self.rotation_axis)
    #     invalid_obj_quat = torch.linalg.norm(obj_axis * self.rotation_axis, dim=1) < self.cfg.object_tilt_lim
    #     num_invalid_obj_quat = torch.sum(invalid_obj_quat)
    #     invalid = torch.logical_or(invalid_obj_quat, invalid)
    #     if debug:
    #         print("Invalid Object Quaternions:", num_invalid_obj_quat)
    #     return invalid, x_start_prime

    def is_invalid(self, debug: bool = False) -> (torch.Tensor, torch.Tensor):
        """ Check if the sampled state is valid to be added to the graph """

        invalid = torch.zeros_like(self.reset_buf)
        # object position constraint
        invalid_object_pos = torch.logical_or(
            torch.any(self.object_pos < self.valid_obj_pos_lower_limits, dim=1),
            torch.any(self.object_pos > self.valid_obj_pos_upper_limits, dim=1),
        )
        invalid = torch.logical_or(invalid, invalid_object_pos)

        # step the environment with 5 zero actions
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
        obj_axis = my_quat_rotate(self.object_orientation, self.rotation_axis)
        invalid_obj_quat = torch.linalg.norm(obj_axis * self.rotation_axis, dim=1) < self.cfg.object_tilt_lim
        invalid = torch.logical_or(invalid, invalid_obj_quat)
        # contact location constraint
        if self.cfg.valid_req_valid_contact:
            invalid_contact = self.contact_location_constraint()
            invalid = torch.logical_or(invalid, invalid_contact)
        # contact number constraint
        if self.cfg.valid_req_ncon_lim:
            invalid_ncon = self.ncon_constraint(min_con=self.cfg.valid_required_ncon)
            invalid = torch.logical_or(invalid, invalid_ncon)
        # # maximum force constraint on each fingertip
        # invalid_ftip_force = torch.zeros_like(self.reset_buf)
        # for i, ftip_contact_force in enumerate(self.fingertip_contact_force):
        #     force_magnitude = torch.linalg.norm(ftip_contact_force, dim=1)
        #     ftip_in_collision = force_magnitude > self.cfg.valid_contact_force_upper_limit
        #     invalid_ftip_force = torch.logical_or(invalid_ftip_force, ftip_in_collision)
        #     invalid = torch.logical_or(invalid, ftip_in_collision)

        x_start_prime = self.get_env_states()

        # print("Stable state percentage: ", 1.0 - (torch.sum(invalid).item() / self.num_envs))

        return invalid, x_start_prime


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
            # magnitude = magnitude - 2 * np.pi * (magnitude > np.pi)
            axis_angle = torch.mul(axis, torch.reshape(magnitude, (-1, 1)))
            rotation_axis_reshaped = self.rotation_axis[0: axis_angle.size(0), :]
            angle = (axis_angle * rotation_axis_reshaped).sum(-1)
            print("Length of this walk:", valid_len)
            print("Angular distance between start and end:", angle)

            # set the state in the walk to Environment 0
            for j in range(valid_len):
                # set the current state
                current_state = walks[i, j, :]
                with torch.inference_mode():
                    self.set_env_states(current_state.unsqueeze(0), torch.tensor([0], device=self.device))
                    self.simulate()
                    sleep(0.5)
            sleep(5)