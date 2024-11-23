import math
import torch
import numpy as np

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation, RigidObject
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.sensors import ContactSensor
from omni.isaac.lab.markers import VisualizationMarkers
from omni.isaac.lab.envs.common import VecEnvStepReturn
from omni.isaac.lab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from omni.isaac.lab.utils.math import quat_from_angle_axis, quat_conjugate, quat_mul, sample_uniform


from utils.misc import to_torch


class IHMEnv(DirectRLEnv):
    cfg: DirectRLEnvCfg

    def __init__(self, cfg: DirectRLEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Flags to control reset behavior as the reset behavior needs
        # to alternate between planning and learning
        self.reset_dist_type = "eval"

        # target joint positions
        self.target_hand_joint_pos = to_torch(self.cfg.default_hand_joint_pos, device=self.device).repeat(self.num_envs, 1)

        # default joint positions
        self.default_hand_joint_pos = to_torch(self.cfg.default_hand_joint_pos, device=self.device).repeat(self.num_envs, 1)

        # default object state
        self.object_default_pos = torch.tensor(self.cfg.default_object_pos, device=self.device)
        self.object_default_rot = torch.tensor(self.cfg.default_object_quat, device=self.device)

        # list of actuated joints
        self.actuated_dof_indices = list()
        for joint_name in cfg.actuated_joint_names:
            self.actuated_dof_indices.append(self.hand.joint_names.index(joint_name))
            # print("Joint_index:", joint_name, self.hand.joint_names.index(joint_name))
        self.num_joints = len(self.actuated_dof_indices)

        # finger bodies
        self.finger_bodies = list()
        for body_name in self.cfg.fingertip_body_names:
            self.finger_bodies.append(self.hand.body_names.index(body_name))
        self.num_fingertips = len(self.finger_bodies)

        # object orientation in the previous time step
        self.prev_object_orientation = self.object.data.default_root_state.clone()[:, 3:7]

        # hand dof limits
        self.hand_dof_lower_limits = to_torch(
            self.cfg.finger_dof_lower_limits, device=self.device
        )
        self.hand_dof_upper_limits = to_torch(
            self.cfg.finger_dof_upper_limits, device=self.device
        )

        # Set up position and velocity limits for sampling
        # object position limits
        # sample limits
        self.sample_obj_pos_lower_limits = torch.tensor(self.cfg.sample_obj_pos_lower_limit, device=self.device)
        self.sample_obj_pos_upper_limits = torch.tensor(self.cfg.sample_obj_pos_upper_limit, device=self.device)
        # validity check limits
        self.valid_obj_pos_lower_limits = torch.tensor(self.cfg.valid_obj_pos_lower_limit, device=self.device)
        self.valid_obj_pos_upper_limits = torch.tensor(self.cfg.valid_obj_pos_upper_limit, device=self.device)

        # velocity limits
        # sample limits
        self.sample_joint_vel_lower_limits = torch.tensor(self.cfg.sample_joint_vel_lower_limit, device=self.device)
        self.sample_joint_vel_upper_limits = torch.tensor(self.cfg.sample_joint_vel_upper_limit, device=self.device)
        self.sample_obj_lin_vel_lower_limits = torch.tensor(self.cfg.sample_obj_lin_vel_lower_limit, device=self.device)
        self.sample_obj_lin_vel_upper_limits = torch.tensor(self.cfg.sample_obj_lin_vel_upper_limit, device=self.device)
        self.sample_obj_ang_vel_lower_limits = torch.tensor(self.cfg.sample_obj_ang_vel_lower_limit, device=self.device)
        self.sample_obj_ang_vel_upper_limits = torch.tensor(self.cfg.sample_obj_ang_vel_upper_limit, device=self.device)
        # validity check limits
        self.valid_joint_vel_lower_limits = torch.tensor(self.cfg.valid_joint_vel_lower_limit, device=self.device)
        self.valid_joint_vel_upper_limits = torch.tensor(self.cfg.valid_joint_vel_upper_limit, device=self.device)
        self.valid_obj_lin_vel_lower_limits = torch.tensor(self.cfg.valid_obj_lin_vel_lower_limit, device=self.device)
        self.valid_obj_lin_vel_upper_limits = torch.tensor(self.cfg.valid_obj_lin_vel_upper_limit, device=self.device)
        self.valid_obj_ang_vel_lower_limits = torch.tensor(self.cfg.valid_obj_ang_vel_lower_limit, device=self.device)
        self.valid_obj_ang_vel_upper_limits = torch.tensor(self.cfg.valid_obj_ang_vel_upper_limit, device=self.device)

        # Set up the sampling space [q, q_dot]
        # joint positions + object position
        self.pos_sample_space_lower = torch.cat(
            [self.hand_dof_lower_limits, self.sample_obj_pos_lower_limits],
        )
        self.pos_sample_space_upper = torch.cat(
            [self.hand_dof_upper_limits, self.sample_obj_pos_upper_limits],
        )
        # joint velocities + object linear and angular velocities
        self.vel_sample_space_lower = torch.cat(
            [
                self.sample_joint_vel_lower_limits,
                self.sample_obj_lin_vel_lower_limits,
                self.sample_obj_ang_vel_lower_limits,
            ],
        )
        self.vel_sample_space_upper = torch.cat(
            [
                self.sample_joint_vel_upper_limits,
                self.sample_obj_lin_vel_upper_limits,
                self.sample_obj_ang_vel_upper_limits,
            ],
        )
        self.pos_sample_space_dim = self.pos_sample_space_upper.shape[0]
        self.vel_sample_space_dim = self.vel_sample_space_upper.shape[0]

        # planning state (x) dimension
        self.planning_state_dim = self.num_joints * 3 + 13
        # joint pos (num_joints) + target joint pos (num_joints) + joint vel (num_joints) + object state (13)

        # initialize contact position marker
        self.visualize_contact_pos = self.cfg.visualize_contact_pos
        if self.visualize_contact_pos:
            self.contact_markers = VisualizationMarkers(self.cfg.contact_pos_cfg)

        # unit tensors to set the position of the goal object
        self.x_unit_tensor = torch.tensor(
            [1, 0, 0],
            dtype=torch.float,
            device=self.device
        ).repeat((self.num_envs, 1))
        self.y_unit_tensor = torch.tensor(
            [0, 1, 0],
            dtype=torch.float,
            device=self.device
        ).repeat((self.num_envs, 1))
        self.z_unit_tensor = torch.tensor(
            [0, 0, 1],
            dtype=torch.float,
            device=self.device
        ).repeat((self.num_envs, 1))


        ### Parameters for the planning task ###

        # create angular distance buffer
        self.planner_ang_dist = torch.tensor(
            [torch.nan] * self.num_envs,
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(-1)
        # create previous angular distance buffer
        self.planner_prev_ang_dist = torch.tensor(
            [torch.nan] * self.num_envs,
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(-1)

        # goal for the planner training
        self.planner_goal = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        self.planner_goal[:, 0] = 1.0
        self.planner_goal_dim = 4

    def _setup_scene(self):
        """ Setup the scene with the robot, object, contact sensor, ground plane, and lights  """
        # add hand and in-hand object
        self.hand = Articulation(self.cfg.robot_cfg)
        self.object = RigidObject(self.cfg.object_cfg)
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        # add articultion to scene
        self.scene.articulations["robot"] = self.hand
        self.scene.rigid_objects["object"] = self.object
        self.scene.sensors["contact_sensor"] = self._contact_sensor
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """ Pre-process actions before stepping through the physics. """
        self.actions = actions.clone()

    def _apply_action(self) -> None:
        """ Apply the actions to the robot based on target joint positions. """
        self.target_hand_joint_pos = torch.clamp(
            self.target_hand_joint_pos + self.cfg.action_scale * self.actions,
            self.hand_dof_lower_limits,
            self.hand_dof_upper_limits,
        )
        self.hand.set_joint_position_target(
            self.target_hand_joint_pos,
            joint_ids=self.actuated_dof_indices
        )

    def step_without_reset(self, action: torch.Tensor) -> VecEnvStepReturn:
        """ Execute one time-step of the environment's dynamics without resetting the environment.
            Almost the same as the step() function, but remove the environment reset.
            It would be useful for the PRM planner to simulate the environment without resetting it.
        Args:
            action: The actions to apply on the environment. Shape is (num_envs, action_dim).

        Returns:
            A tuple containing the observations, rewards, resets (terminated and truncated) and extras.
        """
        # process actions
        self._pre_physics_step(action)

        # check if we need to do rendering within the physics loop
        # note: checked here once to avoid multiple checks within the loop
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        # perform physics stepping
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            # set actions into buffers
            self._apply_action()
            # set actions into simulator
            self.scene.write_data_to_sim()
            # simulate
            self.sim.step(render=False)
            # render between steps only if the GUI or an RTX sensor needs it
            # note: we assume the render interval to be the shortest accepted rendering interval.
            #    If a camera needs rendering at a faster frequency, this will lead to unexpected behavior.
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            # update buffers at sim dt
            self.scene.update(dt=self.physics_dt)

        # post-step:
        # -- update env counters (used for curriculum generation)
        self.episode_length_buf += 1  # step in current episode (per env)
        self.common_step_counter += 1  # total step (common for all envs)

        self.reset_terminated[:], self.reset_time_outs[:] = self._get_dones()
        self.reset_buf = self.reset_terminated | self.reset_time_outs
        self.reward_buf = self._get_rewards()

        # post-step: step interval event
        if self.cfg.events:
            if "interval" in self.event_manager.available_modes:
                self.event_manager.apply(mode="interval", dt=self.step_dt)

        # update observations
        self.obs_buf = self._get_observations()

        # add observation noise
        # note: we apply no noise to the state space (since it is used for critic networks)
        if self.cfg.observation_noise_model:
            self.obs_buf["policy"] = self._observation_noise_model.apply(self.obs_buf["policy"])

        # return observations, rewards, resets and extras
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

    def get_observations(self) -> dict:
        """ A public function to access the observations and states of the environment. """
        return self._get_observations()

    def get_pos_constrants(self):
        invalid = torch.zeros_like(self.reset_buf)
        # check x coordinate upper bound
        invalid = torch.where(
            self.object_pos[:, 0] > self.cfg.xy_object_lim,
            torch.ones_like(self.reset_buf),
            invalid,
        )
        # check x coordinate lower bound
        invalid = torch.where(
            self.object_pos[:, 0] < -self.cfg.xy_object_lim,
            torch.ones_like(self.reset_buf),
            invalid,
        )
        # check y coordinate upper bound
        invalid = torch.where(
            self.object_pos[:, 1] > self.cfg.xy_object_lim,
            torch.ones_like(self.reset_buf),
            invalid,
        )
        # check y coordinate lower bound
        invalid = torch.where(
            self.object_pos[:, 1] < -self.cfg.xy_object_lim,
            torch.ones_like(self.reset_buf),
            invalid,
        )
        # check z coordinate upper bound
        invalid = torch.where(
            self.object_pos[:, 2] > self.cfg.max_object_height,
            torch.ones_like(self.reset_buf),
            invalid,
        )
        # check z coordinate lower bound
        invalid = torch.where(
            self.object_pos[:, 2] < self.cfg.min_object_height,
            torch.ones_like(self.reset_buf),
            invalid,
        )
        return invalid

    def _reset_idx(self, env_ids: torch.Tensor | None) -> None:
        """ Reset environments based on specified indices.

        Args:
            env_ids: List of environment ids which must be reset
        """
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.hand._ALL_INDICES

        self.hand.reset(env_ids)
        super()._reset_idx(env_ids)

        if self.reset_dist_type == "train":
            # sample random state from the reset distribution
            # print("Debug: reset state distribution size:", self.reset_state_buf.shape[0])
            sampled_idx = torch.randint(0, self.reset_state_buf.shape[0], (len(env_ids),))
            states = self.reset_state_buf[sampled_idx].to(self.device)
        else:
            # Reset object with default position and orientation
            # Reset object linear and angular velocities to zero
            object_default_state = self.object.data.default_root_state.clone()[env_ids]
            object_default_state[:, :3] = self.object_default_pos.repeat((len(env_ids), 1))
            object_default_state[:, 3:7] = self.object_default_rot.repeat((len(env_ids), 1))
            object_default_state[:, 7:] = torch.zeros_like(self.object.data.default_root_state[env_ids, 7:])

            # Reset hand joint positions and velocities to default
            dof_pos = self.hand.data.default_joint_pos[env_ids]
            dof_vel = self.hand.data.default_joint_vel[env_ids]

            # Set the reset state to the environment
            states = torch.cat(
                [
                    dof_pos,
                    dof_pos,
                    dof_vel,
                    object_default_state,
                ],
                dim=1
            )
        with torch.inference_mode():
            self.set_env_states(states, env_ids)
            self.simulate()

        # Refresh the intermediate values
        self._compute_intermediate_values()

    def reset_idx(self, env_ids: torch.Tensor | None) -> None:
        """ A public function to access the observations and states of the environment. """
        self._reset_idx(env_ids)

    def _compute_intermediate_values(self) -> None:
        """ Compute intermediate values required for the environment. """
        # Fetch data for hand
        fingertip_pos = self.hand.data.body_pos_w[:, self.finger_bodies]
        fingertip_rot = self.hand.data.body_quat_w[:, self.finger_bodies]
        fingertip_pos -= self.scene.env_origins.repeat((1, self.num_fingertips)).reshape(
            self.num_envs, self.num_fingertips, 3
        )
        # Update fingertip position
        self.fingertip_pos = [
            fingertip_pos[:, ftip_idx, :]
            for ftip_idx in range(self.num_fingertips)
        ]
        # Update fingertip orientation
        self.fingertip_rot = [
            fingertip_rot[:, ftip_idx, :]
            for ftip_idx in range(self.num_fingertips)
        ]
        # Update fingertip velocity
        self.fingertip_velocities = self.hand.data.body_vel_w[:, self.finger_bodies]

        # Update hand joint positions and velocities
        self.hand_dof_pos = self.hand.data.joint_pos
        self.hand_dof_vel = self.hand.data.joint_vel
        self.torque = self.hand.data.applied_torque

        # Update object pose
        self.object_pos = self.object.data.root_pos_w - self.scene.env_origins
        self.object_orientation = self.object.data.root_quat_w
        self.object_lin_vel = self.object.data.root_lin_vel_w
        self.object_ang_vel = self.object.data.root_ang_vel_w

        # Update contact sensor data
        # Fingertip Contact Force
        contact_data = self._contact_sensor.data.net_forces_w
        self.fingertip_contact_force = [
            contact_data[:, ftip_idx, :] for ftip_idx in range(self.num_fingertips)
        ]
        # Contact status
        self.contact_bool_tensor = self.compute_contact_bool()
        # Contact positions
        self.ftip_contact_pos = [
            torch.nan_to_num(pos) for pos in self.compute_ftip_contact_pos()
        ]

        # Update the visualization markers of contact position in the finger tips
        if self.visualize_contact_pos:
            contact_pos = torch.cat(self.ftip_contact_pos, dim=1) + self.scene.env_origins.repeat(1, self.num_fingertips)
            contact_pos = contact_pos.reshape(self.num_envs * self.num_fingertips, 3)
            contact_rot = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat((self.num_envs * self.num_fingertips, 1))
            self.contact_markers.visualize(contact_pos, contact_rot)

    def compute_intermediate_values(self) -> None:
        """ A public function to update intermediate values of the environment. """
        self._compute_intermediate_values()

    def compute_contact_bool(self, distal_finger_force_threshold=1):
        force_scalar = [
            torch.linalg.norm(force, dim=1) for force in self.fingertip_contact_force
        ]
        # # Add noise to the tactile force sensor readings
        # if self.cfg["env"]["enableTactileForceNoise"]:
        #     force_scalar = [
        #         force + 0.2 * 2 * (torch.rand(force.shape, device=self.device) - 0.5) + 0.2
        #         for force in force_scalar
        #     ]
        # compute contact boolean tensor
        contact = [
            (force > distal_finger_force_threshold).long() for force in force_scalar
        ]
        contact_bool_tensor = torch.transpose(torch.stack(contact), 0, 1)
        return contact_bool_tensor

    def compute_ftip_contact_pos(self, ftip_radius=0.0185):
        """
            Compute approximate ftip contact position from ftip positions and net contact force.
            The net contact force is in the global frame.
            The net contact force tensor contains the net contact forces (3D) experienced
            by each rigid body during the last simulation step, with the forces expressed as 3D vectors.
            It is a read-only tensor with shape (num_rigid_bodies, 3).
        """
        contact_pos_cuda = []
        for ftip_pos, contact_force in zip(self.fingertip_pos, self.fingertip_contact_force):
            approx_contact_force_normal = contact_force / torch.linalg.norm(
                contact_force, dim=1, keepdim=True
            )
            contact_pos_cuda.append(
                ftip_pos - approx_contact_force_normal * ftip_radius
            )

        return contact_pos_cuda

    def ncon_constraint(self, min_con=3):
        """ Make sure at least one fingertip should be in contact """
        # fetch contact bool tensor
        # sum contact bool tensor along ftip dimension
        contact_bool_tensor = self.compute_contact_bool()
        contact_sum = torch.sum(contact_bool_tensor, dim=1)
        # check if contact sum is less than min_con
        invalid = torch.where(
            contact_sum < min_con,
            torch.ones_like(contact_sum).to(self.device),
            torch.zeros_like(contact_sum).to(self.device)
        )
        return invalid.bool()

    def contact_location_constraint(self):
        """ Contacts should be made on the front of fingertips """
        contact_invalid = torch.zeros_like(self.reset_buf)
        for i, ftip_contact_force in enumerate(self.fingertip_contact_force):
            force_magnitude = torch.linalg.norm(ftip_contact_force, dim=1)
            ftip_in_contact = force_magnitude > 0
            ftip_contact_normal = torch.nan_to_num(ftip_contact_force / force_magnitude.unsqueeze(-1))
            # fetch ftip axis
            ftip_orientation = self.fingertip_rot[i]
            oracle_x = torch.tensor([1, 0, 0], dtype=torch.float, device=self.device).repeat(self.num_envs, 1)
            ftip_x = quat_apply(ftip_orientation, oracle_x)
            # compute dot product between ftip axis and contact force
            dot_product = torch.sum(ftip_x * ftip_contact_normal, dim=1)
            # if dot product is positive (+ threshold), contact is invalid
            contact_invalid = torch.where(
                torch.logical_and(dot_product > 0.2, ftip_in_contact),
                torch.ones_like(self.reset_buf),
                contact_invalid,
            )
        return contact_invalid.bool()

    def compute_ftip_obj_disp(self):
        """ Compute the displacement of fingertips from the object """
        object_pos = self.object_pos.clone().unsqueeze(1)  # (num_envs, 1, 3)
        ftip_pos = torch.stack(self.ftip_contact_pos).transpose(0, 1)  # (num_envs, num_ftips, 3)
        # broadcast object position
        object_pos = object_pos.repeat(1, self.num_fingertips, 1)  # (num_envs, num_ftips, 3)
        # want average disp for each env --> (num_envs, 1)
        ftip_obj_disp = torch.linalg.norm(ftip_pos - object_pos, dim=-1) ** 2
        total_disp = torch.sum(ftip_obj_disp, dim=1)
        return total_disp

    def fingertip_on_side_constraint(self, upper_lim=0.015, lower_lim=-0.015):
        """ Heights of fingertips in contact should be close to the height of the object"""
        contact_invalid = torch.zeros_like(self.reset_buf)
        for i, ftip_contact_force in enumerate(self.fingertip_contact_force):
            force_magnitude = torch.linalg.norm(ftip_contact_force, dim=1)
            ftip_in_contact = force_magnitude > 0
            ftip_pos_invalid = torch.zeros_like(self.reset_buf)
            ftip_pos_invalid = torch.where(
                self.fingertip_pos[i][:, 2] > (self.object_pos[:, 2] + upper_lim),
                torch.ones_like(self.reset_buf),
                ftip_pos_invalid,
            )
            ftip_pos_invalid = torch.where(
                self.fingertip_pos[i][:, 2] < (self.object_pos[:, 2] + lower_lim),
                torch.ones_like(self.reset_buf),
                ftip_pos_invalid,
            )
            contact_invalid = torch.where(
                torch.logical_and(ftip_pos_invalid, ftip_in_contact),
                torch.ones_like(self.reset_buf),
                contact_invalid,
            )
        return contact_invalid.bool()

    def simulate(self) -> None:
        """ Simulate the environment for one step. """
        # check if we need to do rendering within the physics loop
        # note: checked here once to avoid multiple checks within the loop
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        # perform physics stepping
        for _ in range(self.cfg.decimation):
            # set new state and goal into simulator
            self.scene.write_data_to_sim()
            # simulate
            self.sim.step(render=False)
            # render between steps only if the GUI or an RTX sensor needs it
            # note: we assume the render interval to be the shortest accepted rendering interval.
            #    If a camera needs rendering at a faster frequency, this will lead to unexpected behavior.
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            # update buffers at sim dt
            self.scene.update(dt=self.physics_dt)

    def zero_actions(self) -> torch.Tensor:
        """Returns a buffer with zero actions.

        Returns:
            A buffer of zero torch actions
        """
        actions = torch.zeros(
            [self.num_envs, self.cfg.num_actions],
            dtype=torch.float32,
            device=self.device,
        )

        return actions

    def random_actions(self) -> torch.Tensor:
        """ Returns a buffer with random actions drawn from normal distribution

        Returns:
            torch.Tensor: A buffer of random actions torch actions
        """
        mean = torch.zeros(
            [self.num_envs, self.cfg.num_actions],
            dtype=torch.float32,
            device=self.device,
        )
        std = torch.ones(
            [self.num_envs, self.cfg.num_actions],
            dtype=torch.float32,
            device=self.device,
        )
        actions = torch.normal(mean, std)

        return actions

    def set_reset_state_buf(self, buf: torch.Tensor) -> None:
        print("Buffer size: ", buf.size())
        self.reset_state_buf = buf

    def set_reset_dist_type(self, reset_dist_type: str) -> None:
        """ Set the reset distribution type. """
        self.reset_dist_type = reset_dist_type

    def save_episode_context(self):
        """ Saves episode context to switch to planner """
        context = {
            "episode_length_buf": self.episode_length_buf.detach().clone(),
            "reset_buf": self.reset_buf.detach().clone(),
            "reset_terminated": self.reset_terminated.detach().clone(),
            "reset_time_outs": self.reset_time_outs.detach().clone(),
            "env_states": self.get_env_states(),
            "goal": self.goal.detach().clone() if hasattr(self, "goal") else "None",
        }

        return context

    def restore_episode_context(self, context) -> dict:
        """ Restore episode context from planning to learning """
        with torch.no_grad():
            self.episode_length_buf = context["episode_length_buf"]
            self.reset_buf = context["reset_buf"]
            self.reset_terminated = context["reset_terminated"]
            self.reset_time_outs = context["reset_time_outs"]
            with torch.inference_mode():
                self.set_env_states(context["env_states"], torch.arange(self.num_envs, device=self.device))
                self.simulate()
            if hasattr(self, "goal"):
                self.goal = context["goal"]

        return self.get_observations()

    def get_env_states(self) -> torch.Tensor:
        """ Returns the current state of the environment """
        self._compute_intermediate_values()

        return torch.cat(
            [
                self.hand_dof_pos,
                self.target_hand_joint_pos,
                self.hand_dof_vel,
                self.object_pos,
                self.object_orientation,
                self.object_lin_vel,
                self.object_ang_vel,
            ],
            dim=1,
        )


    """ PRM Planner functions """


    def set_env_states(self, states, env_ids: torch.Tensor) -> None:
        hand_dof_pos = states[:, :self.num_joints]
        target_hand_joint_pos = states[:, self.num_joints: 2 * self.num_joints]
        hand_dof_vel = states[:, 2 * self.num_joints: 3 * self.num_joints]
        object_state = states[:, 3 * self.num_joints: 3 * self.num_joints + 13]
        # Add scene origins to object position
        object_state[:, :3] = object_state[:, :3] + self.scene.env_origins[env_ids]

        # Update object previous orientation
        self.prev_object_orientation[env_ids, :] = states[:, 3 * self.num_joints + 3: 3 * self.num_joints + 7]

        # Update target joint positions
        self.target_hand_joint_pos[env_ids, :] = target_hand_joint_pos
        # Update hand joint states
        self.hand.set_joint_position_target(target_hand_joint_pos, env_ids=env_ids, joint_ids=self.actuated_dof_indices)
        self.hand.write_joint_state_to_sim(hand_dof_pos, hand_dof_vel, env_ids=env_ids, joint_ids=self.actuated_dof_indices)
        # Update object states
        self.object.write_root_state_to_sim(object_state, env_ids)

    def get_env_q(self) -> torch.Tensor:
        """ Returns the current q_state of the environment """
        return self.get_env_states()

    def sample_q(self, num_samples) -> torch.Tensor:
        # sample random joint positions and object position unifromly
        alpha = torch_rand_float(
            0.0,
            1.0,
            (num_samples, self.pos_sample_space_dim),
            device=self.device,
        )
        x_pos_sample = alpha * self.pos_sample_space_upper + (1 - alpha) * self.pos_sample_space_lower
        # sample random object orientation
        beta = sample_uniform(-1.0, 1.0, (num_samples, 2), device=self.device)
        x_rot_sample = randomize_rotation(
            beta[:, 0], beta[:, 1], self.x_unit_tensor[list(range(num_samples))],
            self.y_unit_tensor[list(range(num_samples))]
        )
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
                x_rot_sample,  # object rotation
                x_vel_sample[:, self.num_joints: self.num_joints + 6],  # object linear and angularvelocity
            ],
            dim=1
        )

        return x_samples

    def sample_random_goal_state(self, num_goal) -> torch.Tensor:
        """ Sample random goal positions """
        return self.sample_q(num_goal)

    def sample_random_nodes(self, N: int = 32) -> torch.Tensor:
        """ Uniformly sample initial collision-free nodes to be added to the graph """
        sampled_nodes = []

        while len(sampled_nodes) < N:
            # sample random states unifromly
            x_samples = self.sample_q(num_samples=self.num_envs)
            # apply the sampled states to the environment
            with torch.inference_mode():
                self.set_env_states(x_samples, torch.tensor(list(range(self.num_envs)), device=self.device))
                self.simulate()
            self._compute_intermediate_values()

            # perform validity check
            invalid, x_start_prime = self.is_invalid()

            # add valid states to the list
            valid_indices = torch.nonzero(torch.logical_not(invalid), as_tuple=False).squeeze(-1)
            for idx in valid_indices:
                if len(sampled_nodes) >= N:
                    break
                sampled_nodes.append(x_start_prime[idx].clone())

        return torch.stack(sampled_nodes).to(self.device)

    def compute_distance(self, selected_node: torch.Tensor, prm_nodes: torch.Tensor, disable_velocity=False) -> torch.Tensor:
        """ Computes distance from a specific node to each node in node set """
        # Joint position distance
        joint_pos_dist = torch.abs(prm_nodes[:, : self.num_joints] - selected_node[: self.num_joints])
        joint_pos_dist = 0.3 * torch.sum(joint_pos_dist, dim=1)
        # Target joint position distance
        target_joint_pos_dist = torch.abs(prm_nodes[:, self.num_joints: self.num_joints * 2] - selected_node[self.num_joints: self.num_joints * 2])
        target_joint_pos_dist = 0.01 * torch.sum(target_joint_pos_dist, dim=1)
        # Joint velocity distance
        joint_vel_dist = 0.001 * torch.linalg.norm(
            prm_nodes[:, self.num_joints * 2: self.num_joints * 3] - selected_node[self.num_joints * 2: self.num_joints * 3],
            dim=1
        )
        # Object position distance
        obj_pos_dist = 1.0 * torch.linalg.norm(
            prm_nodes[:, self.num_joints * 3: self.num_joints * 3 + 3] - selected_node[self.num_joints * 3: self.num_joints * 3 + 3],
            dim=1
        )
        # Object orientation distance
        selected_node_quat = selected_node[self.num_joints * 3 + 3: self.num_joints * 3 + 7].unsqueeze(0)
        nodes_quat = prm_nodes[:, self.num_joints * 3 + 3: self.num_joints * 3 + 7].view(-1, 4)
        obj_rot_dist = 2.0 * compute_quat_angle(selected_node_quat, nodes_quat).squeeze()
        # Object lin vel and ang vel distance
        obj_vel_dist = 0.005 * torch.linalg.norm(
            prm_nodes[:, self.num_joints * 3 + 7: self.num_joints * 3 + 13] - selected_node[self.num_joints * 3 + 7: self.num_joints * 3 + 13],
            dim=1
        )

        # Compute the total distance
        if disable_velocity:
            total_dist = joint_pos_dist + target_joint_pos_dist+ obj_pos_dist + obj_rot_dist
        else:
            total_dist = joint_pos_dist + target_joint_pos_dist + joint_vel_dist + obj_pos_dist + obj_rot_dist + obj_vel_dist
        return total_dist

    def is_invalid(self, debug: bool = False) -> (torch.Tensor, torch.Tensor):
        """ Check if the sampled state is valid to be added to the graph """
        self._compute_intermediate_values()

        invalid = torch.zeros_like(self.reset_buf)
        # object position constraint
        invalid_object_pos = torch.logical_or(
            torch.any(self.object_pos < self.valid_obj_pos_lower_limits, dim=1),
            torch.any(self.object_pos > self.valid_obj_pos_upper_limits, dim=1),
        )
        num_invalid_object_pos = torch.sum(invalid_object_pos)
        invalid = torch.logical_or(invalid, invalid_object_pos)
        # object linear velocity constraint
        invalid_object_lin_vel = torch.logical_or(
            torch.any(self.object_lin_vel < self.valid_obj_lin_vel_lower_limits, dim=1),
            torch.any(self.object_lin_vel > self.valid_obj_lin_vel_upper_limits, dim=1),
        )
        num_invalid_object_lin_vel = torch.sum(invalid_object_lin_vel)
        invalid = torch.logical_or(invalid, invalid_object_lin_vel)
        # object angular velocity constraint
        invalid_object_ang_vel = torch.logical_or(
            torch.any(self.object_ang_vel < self.valid_obj_ang_vel_lower_limits, dim=1),
            torch.any(self.object_ang_vel > self.valid_obj_ang_vel_upper_limits, dim=1),
        )
        num_invalid_object_ang_vel = torch.sum(invalid_object_ang_vel)
        invalid = torch.logical_or(invalid, invalid_object_ang_vel)
        # joint velocity constraint
        invalid_joint_vel = torch.logical_or(
            torch.any(self.hand_dof_vel < self.valid_joint_vel_lower_limits, dim=1),
            torch.any(self.hand_dof_vel > self.valid_joint_vel_upper_limits, dim=1),
        )
        num_invalid_joint_vel = torch.sum(invalid_joint_vel)
        invalid = torch.logical_or(invalid, invalid_joint_vel)
        # contact location constraint
        if self.cfg.valid_req_valid_contact:
            invalid_contact = self.contact_location_constraint()
            num_invalid_contact = torch.sum(invalid_contact)
            invalid = torch.logical_or(invalid, invalid_contact)
        # contact number constraint
        if self.cfg.valid_req_ncon_lim:
            invalid_ncon = self.ncon_constraint(min_con=self.cfg.valid_required_ncon)
            num_invalid_ncon = torch.sum(invalid_ncon)
            invalid = torch.logical_or(invalid, invalid_ncon)
        # maximum force constraint on each fingertip
        invalid_ftip_force = torch.zeros_like(self.reset_buf)
        for i, ftip_contact_force in enumerate(self.fingertip_contact_force):
            force_magnitude = torch.linalg.norm(ftip_contact_force, dim=1)
            ftip_in_collision = force_magnitude > self.cfg.valid_contact_force_upper_limit
            invalid_ftip_force = torch.logical_or(invalid_ftip_force, ftip_in_collision)
            invalid = torch.logical_or(invalid, ftip_in_collision)

        num_invalid_ftip_force = torch.sum(invalid_ftip_force)

        x_start_prime = self.get_env_states()

        if debug:
            print("Invalid Object Position: ", num_invalid_object_pos)
            print("Invalid Object Lin Vel: ", num_invalid_object_lin_vel)
            print("Invalid Object Ang Vel: ", num_invalid_object_ang_vel)
            print("Invalid Joint Vel: ", num_invalid_joint_vel)
            if self.cfg.req_valid_contact:
                print("Invalid Contact Location: ", num_invalid_contact)
            if self.cfg.req_ncon_lim:
                print("Invalid Contact Number: ", num_invalid_ncon)
            print("Invalid Fingertip Force: ", num_invalid_ftip_force)

        return invalid, x_start_prime


    """ PRM planner policy functions """


    def planner_get_rewards(self):
        """ Compute and return the rewards for the environment.

                Returns:
                    The rewards for the environment. Shape is (num_envs,).
                """
        # Refresh the intermediate values after the physics steps
        self._compute_intermediate_values()

        # update object angular distance
        self.planner_ang_dist = compute_quat_angle(self.planner_goal, self.object_orientation)
        # compute dense reward
        reward = self.planner_compute_delta_reward()

        # compute out-of-bounds penalty
        oob = self.get_pos_constrants()
        reward += -1 * self.cfg.planner_outOfBoundsPenalty * oob.unsqueeze(1)

        # compute sparse reward
        success = self.planner_check_success()
        reward += self.cfg.planner_taskCompletionReward * success.unsqueeze(1)

        # compute ftip-object distance penalty
        total_ftip_disp = self.compute_ftip_obj_disp()
        reward += -1 * self.cfg.planner_ftipObjDispPenalty * total_ftip_disp.unsqueeze(1)

        # penalize for failing to reach contact minimum
        nconpen = -1 * self.cfg.planner_numContactPenalty * self.ncon_constraint(min_con=self.cfg.planner_required_ncon).float()
        reward += nconpen.unsqueeze(1)

        # penalize for object position deviation
        obj_position_dev = torch.linalg.norm(self.object_pos - self.object_default_pos, dim=-1) ** 2
        obj_position_pen = -1 * self.cfg.planner_objPositionPenalty * obj_position_dev
        reward += obj_position_pen.unsqueeze(1)

        # penalize for contact on back of finger
        contact_loc_pen = -1 * self.cfg.planner_contactLocationPenalty * self.contact_location_constraint().float()
        reward += contact_loc_pen.unsqueeze(1)

        # update previous angular distance
        self.planner_prev_ang_dist[:] = self.planner_ang_dist.clone()

        return reward.squeeze(-1)

    def planner_check_success(self):
        """
            Task is complete if the object lies within the angular threshold,
            and all motion criteria are met.
        """
        success = torch.ones_like(self.reset_buf)
        # object orientation criteria
        ang_dist = self.planner_ang_dist.clone().squeeze(-1)
        orientation_crit = torch.where(
            ang_dist < self.cfg.planner_angle_threshold,
            torch.ones_like(self.reset_buf),
            torch.zeros_like(self.reset_buf)
        )
        success = torch.logical_and(success, orientation_crit)
        # finger motion criteria
        q_dot = torch.linalg.norm(self.hand_dof_vel.clone(), dim=-1)
        q_dot_crit = torch.where(
            q_dot < self.cfg.planner_joint_vel_threshold,
            torch.ones_like(self.reset_buf),
            torch.zeros_like(self.reset_buf)
        )
        success = torch.logical_and(success, q_dot_crit)
        # object motion criteria, linear velocity
        object_lin_vel = torch.linalg.norm(self.object_lin_vel, dim=-1)
        object_lin_vel_crit = torch.where(
            object_lin_vel < self.cfg.planner_object_vel_threshold,
            torch.ones_like(self.reset_buf),
            torch.zeros_like(self.reset_buf)
        )
        success = torch.logical_and(success, object_lin_vel_crit)
        # object motion criteria, angular velocity
        object_ang_vel = torch.linalg.norm(self.object_ang_vel, dim=-1)
        object_ang_vel_crit = torch.where(
            object_ang_vel < self.cfg.planner_object_angvel_threshold,
            torch.ones_like(self.reset_buf),
            torch.zeros_like(self.reset_buf)
        )
        success = torch.logical_and(success, object_ang_vel_crit)
        # contact location criteria (all contacts must be on the inside of the fingers)
        if self.cfg.planner_req_valid_contact:
            invalid_contact = self.contact_location_constraint()
            valid_contact = torch.logical_not(invalid_contact)
            success = torch.logical_and(success, valid_contact)
        # contact count criteria (at least 1 contact must be made in 4fhand)
        if self.cfg.planner_req_ncon_lim:
            invalid_ncon = self.ncon_constraint(min_con=self.cfg.planner_required_ncon)
            valid_ncon = torch.logical_not(invalid_ncon)
            success = torch.logical_and(success, valid_ncon)

        return success

    def planner_compute_delta_reward(self):
        """ compute dense orientation reward based on change in displacement to target """
        # fetch previous displacement, accounting for the case where ep was reset (nan)
        nan_indices = torch.argwhere(torch.isnan(self.planner_prev_ang_dist).float()).squeeze(-1)
        if len(nan_indices) > 0:
            self.planner_prev_ang_dist[nan_indices] = self.planner_ang_dist[nan_indices]
        # compute delta displacement
        prev_ang_dist = self.planner_prev_ang_dist.clone()
        delta_displacement = torch.clip(self.planner_ang_dist - prev_ang_dist, -self.cfg.planner_delta_clip, self.cfg.planner_delta_clip)
        # orientation reward
        reward = -1 * self.cfg.planner_deltaReward * delta_displacement
        return reward

    def planner_get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
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
        success = self.planner_check_success()

        # If task completion should trigger reset, reset the environments that are successful
        done = torch.where(
            success.bool(),
            torch.ones_like(self.reset_buf),
            done
        )

        # Check if the contact is made on the front of the fingertips
        if self.cfg.planner_ftip_on_side_constraint:
            done = torch.where(
                self.fingertip_on_side_constraint(
                    upper_lim=self.cfg.planner_ftip_on_side_constraint_tolerance,
                    lower_lim=-1 * self.cfg.planner_ftip_on_side_constraint_tolerance
                ),
                torch.ones_like(self.reset_buf),
                done,
            )
        # Check the number of contacts
        if self.cfg.planner_num_contact_constraint:
            done = torch.where(
                self.ncon_constraint(min_con=self.cfg.planner_required_ncon),
                torch.ones_like(self.reset_buf),
                done,
            )
        # Check if the contact location is valid
        if self.cfg.planner_contact_loc_constraint:
            done = torch.where(
                self.contact_location_constraint(),
                torch.ones_like(self.reset_buf),
                done,
            )

        # Check the time limit
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        return done, time_out

    def planner_step_without_reset(self, action: torch.Tensor) -> VecEnvStepReturn:
        """ Execute one time-step of the environment's dynamics without resetting the environment.
            Almost the same as the step() function, but remove the environment reset.
            It would be useful for the PRM planner to simulate the environment without resetting it.
        Args:
            action: The actions to apply on the environment. Shape is (num_envs, action_dim).

        Returns:
            A tuple containing the observations, rewards, resets (terminated and truncated) and extras.
        """
        # add action noise
        if self.cfg.action_noise_model:
            action = self._action_noise_model.apply(action.clone())
        # process actions
        self._pre_physics_step(action)

        # check if we need to do rendering within the physics loop
        # note: checked here once to avoid multiple checks within the loop
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        # perform physics stepping
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            # set actions into buffers
            self._apply_action()
            # set actions into simulator
            self.scene.write_data_to_sim()
            # simulate
            self.sim.step(render=False)
            # render between steps only if the GUI or an RTX sensor needs it
            # note: we assume the render interval to be the shortest accepted rendering interval.
            #    If a camera needs rendering at a faster frequency, this will lead to unexpected behavior.
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            # update buffers at sim dt
            self.scene.update(dt=self.physics_dt)

        # post-step:
        # -- update env counters (used for curriculum generation)
        self.episode_length_buf += 1  # step in current episode (per env)
        self.common_step_counter += 1  # total step (common for all envs)

        self.reset_terminated[:], self.reset_time_outs[:] = self.planner_get_dones()
        self.reward_buf = self.planner_get_rewards()

        # post-step: step interval event
        if self.cfg.events:
            if "interval" in self.event_manager.available_modes:
                self.event_manager.apply(mode="interval", dt=self.step_dt)

        # update observations
        self.obs_buf = self._get_observations()

        # add observation noise
        # note: we apply no noise to the state space (since it is used for critic networks)
        if self.cfg.observation_noise_model:
            self.obs_buf["policy"] = self._observation_noise_model.apply(self.obs_buf["policy"])

        # return observations, rewards, resets and extras
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

    def set_planner_goal(self, goal, env_ids: torch.Tensor):
        """ Set the goal state for the environment and update the goal marker """
        self.planner_goal[env_ids, :] = goal

    def q_to_planner_goal(self, q: torch.Tensor) -> torch.Tensor:
        """ Extract object rotation from q_state to serve as goal """
        planner_goal = q[:, self.num_joints * 3 + 3: self.num_joints * 3 + 7]  # extract goal rotation
        return planner_goal

    def compute_distances_in_planner_goal(self, prm_nodes: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        """
            Computes distance in goal state from a specific node to each node in node set.
        """
        # Extract object rotation from goal
        nodes_quat = prm_nodes[:, self.num_joints * 3 + 3: self.num_joints * 3 + 7].view(-1, 4)
        goal_quat = goal[self.num_joints * 3 + 3: self.num_joints * 3 + 7]
        # Compute angular distance
        distances = compute_quat_angle(goal_quat, nodes_quat)
        return distances


##
# torch.jit functions
##


@torch.jit.script
def randomize_rotation(rand0, rand1, x_unit_tensor, y_unit_tensor):
    return quat_mul(
        quat_from_angle_axis(rand0 * np.pi, x_unit_tensor), quat_from_angle_axis(rand1 * np.pi, y_unit_tensor)
    )

@torch.jit.script
def quat_apply(a, b):
    shape = b.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 3)

    w = a[:, 0]
    xyz = a[:, 1:]
    t = xyz.cross(b, dim=-1) * 2

    return (b + w.unsqueeze(-1) * t + xyz.cross(t, dim=-1)).view(shape)

@torch.jit.script
def torch_rand_float(lower, upper, shape, device):
    # type: (float, float, Tuple[int, int], str) -> Tensor
    return (upper - lower) * torch.rand(*shape, device=device) + lower

def compute_quat_angle(quat1, quat2):
    # compute angle between two quaternions
    # broadcast quat 1 to quat 2 size
    quat1 = torch.broadcast_to(quat1, quat2.size())
    quat_diff = quat_mul(quat1, quat_conjugate(quat2))
    magnitude, axis = quat_to_angle_axis(quat_diff)
    magnitude = magnitude - 2 * np.pi * (magnitude > np.pi)
    return torch.abs(magnitude).unsqueeze(1)

@torch.jit.script
def quat_to_angle_axis(q):
    # type: (Tensor) -> Tuple[Tensor, Tensor]
    # Computes axis-angle representation from quaternion q
    # q must be normalized
    min_theta = 1e-5
    qw, qx, qy, qz = 0, 1, 2, 3

    sin_theta = torch.sqrt(1 - q[..., qw] * q[..., qw])
    angle = 2 * torch.acos(q[..., qw])
    angle = torch.atan2(torch.sin(angle), torch.cos(angle))  # normalize angle
    angle = angle + 2 * np.pi * (angle < 0)
    sin_theta_expand = sin_theta.unsqueeze(-1)
    axis = q[..., qx:qz+1] / sin_theta_expand

    mask = sin_theta > min_theta
    default_axis = torch.zeros_like(axis)
    default_axis[..., -1] = 1

    angle = torch.where(mask, angle, torch.zeros_like(angle))
    mask_expand = mask.unsqueeze(-1)
    axis = torch.where(mask_expand, axis, default_axis)
    return angle, axis

@torch.jit.script
def normalize_angle(x):
    return torch.atan2(torch.sin(x), torch.cos(x))

@torch.jit.script
def my_quat_rotate(q, v):
    # Rotate along axis v by quaternion q
    shape = q.shape
    q_w = q[:, 0]  # scalar part (w)
    q_vec = q[:, 1:]  # vector part (x, y, z)
    a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * \
        torch.bmm(q_vec.view(shape[0], 1, 3), v.view(
            shape[0], 3, 1)).squeeze(-1) * 2.0
    return a + b + c

def generate_quaternions_within_tilt(axis, threshold_angle_radians, num_samples=1):
    """
    Generate random quaternions with the condition that the dot product of the rotated z-axis
    and the world z-axis is greater than cos(threshold_angle_radians).

    Parameters:
        threshold_angle_radians (float): The maximum allowed tilt angle (in radians).
        num_samples (int): The number of quaternions to generate.

    Returns:
        torch.Tensor: A tensor of shape (num_samples, 4) containing the quaternions [w, x, y, z].
    """
    valid_quaternions = []

    # Compute the cosine of the threshold angle to use for comparison
    cos_threshold = math.cos(threshold_angle_radians)

    while len(valid_quaternions) < num_samples:
        # Generate a random quaternion
        q = torch.rand((1, 4))
        q = q / torch.norm(q, dim=-1, keepdim=True)  # Normalize quaternion

        # Rotate the z-axis [0, 0, 1] by the quaternion
        v = axis.unsqueeze(0)  # z-axis vector
        v_rot = my_quat_rotate(q, v)  # Rotate the z-axis by the quaternion

        # Compute the dot product between the rotated z-axis and the world z-axis
        dot_product = v_rot[:, 2]  # Since the z-axis is [0, 0, 1], this is the z component of the rotated axis

        # Check if the dot product is greater than or equal to the cosine of the threshold angle
        if dot_product >= cos_threshold:
            valid_quaternions.append(q.squeeze(0))

    return torch.stack(valid_quaternions)