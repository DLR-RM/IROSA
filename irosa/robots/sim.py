################################################################################
# Copyright (c) 2025. Markus Knauer, Joao Silverio                            #
# Licensed under the MIT License. See LICENSE file for details.                 #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 2025                                                                   #
# Author: Markus Knauer                                                        #
# E-mail: markus.knauer@dlr.de                                                 #
# Website: https://github.com/DLR-RM/IROSA                                    #
################################################################################

"""PyBullet-based robot simulator for IROSA."""

from __future__ import annotations

import time

import numpy as np
import pybullet as p
import pybullet_data as pd

from irosa.robots.robot import Robot, RobotType


class Simulator(Robot):
    """PyBullet simulator with KUKA IIWA robot and trajectory visualization."""

    def __init__(self) -> None:
        super().__init__()
        self.robot: int | None = None
        self.robot_dofs: int | None = None
        self.robot_ee_index: int | None = None
        self.ll: list[float] | None = None
        self.ul: list[float] | None = None
        self.jr: list[float] | None = None
        self.rp: list[float] | None = None
        self.flags: int | None = None
        self.time_step: float | None = None
        self.trj_id: list[list[int]] = []
        self.pt_id: dict[str, list] = dict(id=[], output=[])

    def get_type(self) -> RobotType:
        return RobotType.SIMULATION

    def on_setup(self, test_mode: bool = False) -> None:
        if not test_mode:
            self.setup_scenario()

    def on_trajectory_updated(self, trajectory: np.ndarray, mean: np.ndarray, delta_t: float = 0.05) -> None:
        super().on_trajectory_updated(trajectory, mean, delta_t)
        self.delete_drawings(self.trj_id)
        self.trj_id = [self._draw_prediction(mean)]

    def on_viapoints_added(self, positions_xyz: list[list]) -> None:
        ids = self.draw_via_points(positions_xyz, color=[1, 0, 0])
        self.pt_id["id"] += ids
        self.pt_id["output"] += positions_xyz

    def on_viapoints_removed(self, positions_xyz: list[list]) -> None:
        ids = []
        for entry in positions_xyz:
            idx = [i for i, x in enumerate(self.pt_id["output"]) if x == entry]
            for item in sorted(idx, reverse=True):
                ids.append(self.pt_id["id"].pop(item))
                self.pt_id["output"].pop(item)
        self.delete_drawings([ids])

    def init_robot_configuration(self, jp: list[float]) -> None:
        """Initialize robot joint positions."""
        index = 0
        for j in range(p.getNumJoints(self.robot)):
            p.changeDynamics(self.robot, j, linearDamping=0, angularDamping=0)
            info = p.getJointInfo(self.robot, j)
            joint_type = info[2]
            if joint_type in (p.JOINT_PRISMATIC, p.JOINT_REVOLUTE):
                p.resetJointState(self.robot, j, jp[index])
                index += 1
        self.robot_dofs = index

    def setup_scenario(
        self,
        initial_joint_positions: list | None = None,
        camera_position: list[float] | None = None,
    ) -> None:
        """Set up the robot simulation scenario.

        :param initial_joint_positions: Initial joint configuration (None uses default)
        :param camera_position: Camera position [dist, yaw, pitch, target] (None uses default)
        """
        if p.isConnected():
            p.disconnect()
        p.connect(p.GUI)

        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
        if camera_position:
            p.resetDebugVisualizerCamera(
                cameraDistance=camera_position[0],
                cameraYaw=camera_position[1],
                cameraPitch=camera_position[2],
                cameraTargetPosition=camera_position[3],
            )
        else:
            p.resetDebugVisualizerCamera(
                cameraDistance=0.5,
                cameraYaw=90,
                cameraPitch=-45,
                cameraTargetPosition=(1.081, -0.133, +0.728),
            )

        p.setAdditionalSearchPath(pd.getDataPath())
        p.setRealTimeSimulation(True)
        self.time_step = 1.0 / 60.0
        p.setTimeStep(self.time_step)
        p.setGravity(0, 0, -9.8)

        self.robot_ee_index = 6
        self.robot_dofs = 7
        self.ll = [-2.97, -2.09, -2.97, -2.09, -2.97, -2.09, -3.05]
        self.ul = [2.97, 2.09, 2.97, 2.09, 2.97, 2.09, 3.05]
        self.jr = [7] * self.robot_dofs
        if initial_joint_positions is None:
            initial_joint_positions = [2.0, 0.458, 0.31, -2.24, -0.30, 0.4, 1.32]
        self.rp = initial_joint_positions
        self.flags = p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES

        self.floor = p.loadURDF("plane.urdf", basePosition=[0.0, 0.0, -0.65], useFixedBase=True)
        self.table = p.loadURDF(
            "table/table.urdf",
            [0.5, 0.35, -0.625],
            p.getQuaternionFromEuler([0.0, 0.0, 0]),
            useFixedBase=True,
            flags=self.flags,
        )

        orn = p.getQuaternionFromEuler([0.0, 0.0, -np.pi / 2.0])
        self.robot = p.loadURDF(
            "kuka_iiwa/model.urdf",
            np.array([0.0, 0.0, 0.0]),
            orn,
            useFixedBase=True,
            flags=self.flags,
        )
        self.init_robot_configuration(initial_joint_positions)

    def move_robot(self, x_h: np.ndarray) -> None:
        """Move robot to target position using inverse kinematics."""
        joint_poses = p.calculateInverseKinematics(
            self.robot,
            self.robot_ee_index,
            x_h,
            lowerLimits=self.ll,
            upperLimits=self.ul,
            jointRanges=self.jr,
            restPoses=self.rp,
            maxNumIterations=5,
        )
        assert self.robot_dofs is not None
        for i in range(self.robot_dofs):
            start_time = time.time()
            p.setJointMotorControl2(self.robot, i, p.POSITION_CONTROL, joint_poses[i], force=5 * 240.0)
            duration = time.time() - start_time
            if i < len(self.predicting_frequency):
                time.sleep(max(0, self.predicting_frequency[i] - duration))

    def _draw_prediction(self, prediction: np.ndarray) -> list[int]:
        """Draw predicted trajectory as lines."""
        color = [0.5, 0.5, 0.5]
        ids = []
        for i in range(prediction.shape[0] - 1):
            ids.append(p.addUserDebugLine(prediction[i][0:3], prediction[i + 1][0:3], lineColorRGB=color, lineWidth=5.0))
        return ids

    def draw_via_points(self, via_points: list[list], color: list | None = None) -> list[int]:
        """Draw via-points in the simulation."""
        if color is None:
            color = [0.5, 0.5, 0.5]
        ids = []
        for entry in via_points:
            draw_to = [x + 0.01 for x in entry]
            ids.append(p.addUserDebugLine(entry, draw_to, color, lineWidth=5.0))
        return ids

    def delete_drawings(self, ids: list | None = None) -> None:
        """Delete debug drawings by ID."""
        if ids is not None:
            for id_list in ids:
                for item in id_list:
                    p.removeUserDebugItem(int(item))
        else:
            p.removeAllUserDebugItems()

    def pos_controller(self, trajectory: np.ndarray, run_in_loop: bool = False) -> None:
        """Execute trajectory on the simulated robot.

        :param trajectory: Trajectory array with 7-9 columns
        :param run_in_loop: Whether to loop continuously
        """
        if not isinstance(trajectory, np.ndarray):
            raise TypeError("trajectory must be a numpy ndarray")
        amount = trajectory.shape[1]
        if not (7 <= amount <= 9):
            raise ValueError(f"trajectory must have 7-9 columns, got {amount}")

        for traj_point in trajectory:
            with self.robot_lock:
                self.move_robot(traj_point[-7:-4].reshape((3, 1)))
            self.num_finished_loops[0] += 1
            time.sleep(0.01)

    def disconnect(self) -> None:
        """Disconnect from PyBullet."""
        p.disconnect()
