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

"""Abstract robot interface for IROSA."""

from __future__ import annotations

import abc
import enum
from threading import Lock

import numpy as np


class RobotType(enum.Enum):
    SIMULATION = 1


class Robot(abc.ABC):
    """Abstract base class for robot interfaces.

    Provides trajectory timing control, thread-safe locking, and hook methods
    for model-robot communication.
    """

    def __init__(self) -> None:
        self.robot_lock: Lock = Lock()
        self.predicting_frequency: np.ndarray = np.array([])
        self.timesteps: np.ndarray = np.array([])
        self.num_finished_loops: list[int] = [0]
        self.first_init: bool = True

    def update_tools(self, tools: list) -> list:
        return tools

    @abc.abstractmethod
    def get_type(self) -> RobotType: ...

    @abc.abstractmethod
    def pos_controller(self, trajectory: np.ndarray) -> None: ...

    def run(self, trajectory: np.ndarray, delta_t: float = 0.05) -> None:
        """Start robot trajectory execution.

        :param trajectory: Cartesian trajectory (x,y,z + quat)
        :param delta_t: Time between two points, default 0.05 for 20 Hz
        """
        if self.first_init:
            self.init_predicting_frequency(trajectory=trajectory, delta_t=delta_t)
        self.pos_controller(trajectory=trajectory)

    def init_predicting_frequency(self, trajectory: np.ndarray, delta_t: float = 0.05) -> None:
        """Initialize timing for each trajectory point.

        :param trajectory: Cartesian trajectory
        :param delta_t: Time between two points
        """
        self.predicting_frequency = np.array([delta_t for _ in range(len(trajectory))])
        self.timesteps = np.array(trajectory[:, 0])
        self.first_init = False

    def on_setup(self, test_mode: bool = False) -> None:
        """Called once during model initialization."""

    def on_trajectory_updated(self, trajectory: np.ndarray, mean: np.ndarray, delta_t: float = 0.05) -> None:
        """Called when model trajectory changes."""
        self.init_predicting_frequency(trajectory=trajectory, delta_t=delta_t)

    def on_viapoints_added(self, positions_xyz: list[list]) -> None:
        """Called when via-points are added."""

    def on_viapoints_removed(self, positions_xyz: list[list]) -> None:
        """Called when via-points are removed."""

    def change_predicting_frequency(self, percentage_factor: int, adaptation_start: float, adaptation_end: float) -> None:
        """Change execution speed for a time segment.

        Increasing delta_t slows the robot; decreasing delta_t speeds it up.
        If set to 50%, delta_t is increased by 1.5x (slowing down).
        If set to -50%, delta_t is decreased by 0.5x (speeding up).

        :param percentage_factor: Speed change in percent (positive=slower, negative=faster)
        :param adaptation_start: Start time (0.0-1.0)
        :param adaptation_end: End time (0.0-1.0)
        """
        if len(self.predicting_frequency) == 0:
            raise RuntimeError("The predicting frequency was not set, please set it first")

        factor = (abs(percentage_factor) + 100) / 100
        if percentage_factor < 0:
            factor = 1 / factor

        if not (0.0 <= adaptation_start <= 1.0):
            raise ValueError("adaptation_start has to be a value between 0.0 and 1.0")
        if not (0.0 <= adaptation_end <= 1.0):
            raise ValueError("adaptation_end has to be a value between 0.0 and 1.0")
        if adaptation_end <= adaptation_start:
            raise ValueError("adaptation_end has to be a larger value than adaptation_start")

        new_pred_frequency = self.predicting_frequency.copy()
        for idx, val_time in enumerate(self.timesteps):
            if val_time < adaptation_start:
                continue
            if val_time > adaptation_end:
                break
            new_pred_frequency[idx] = abs(self.predicting_frequency[idx] * factor)

        with self.robot_lock:
            self.predicting_frequency[:] = new_pred_frequency.copy()
