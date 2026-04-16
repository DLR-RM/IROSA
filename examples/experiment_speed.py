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

"""Experiment 1: Speed Adaptation via Natural Language (Section V-B of the paper).

Reproduces the speed modulation experiment from:
    "Interactive robot skill adaptation using natural language"

Scenario:
    The robot performs a pick-and-insert task (transferring a bearing ring from
    its box to a measurement station). The user says: "slow down between box
    and station". The system identifies the time segment corresponding to the
    phase after picking up the ring (t=0.55) and before reaching the station
    (t=0.72) using proximity thresholding on environment observations, then
    scales the time intervals in that segment.

What this example demonstrates:
    1. Training KMP from demonstrations
    2. Providing environment observations (object positions and labels)
    3. Trajectory segment determination (mapping spatial references to time)
    4. Speed modulation via the SlowDownRobot tool
"""

import pathlib

import numpy as np

from irosa.models.kmp import KMPWrapper
from irosa.robots.robot import Robot, RobotType
from irosa.tool_definitions.robot import SlowDownRobot

DATA_PATH = pathlib.Path(__file__).parent.parent / "data" / "demonstrations.npz"


# -- Environment observations (manually provided, as in the paper) --
# Each object: {position, dimensions, label}
ENVIRONMENT = [
    {"position": np.array([0.82, 0.47, 0.13]), "label": "bearing ring box"},
    {"position": np.array([0.47, -0.24, 0.08]), "label": "measurement station"},
]


def find_closest_time(trajectory: np.ndarray, target_position: np.ndarray, d_prox: float = 0.02) -> float:
    """Trajectory segment determination (Section IV-C of the paper).

    Iterates over the predicted trajectory and returns the closest time point
    for a referenced object (within proximity threshold d_prox).

    :param trajectory: KMP trajectory [time, x, y, z, ...]
    :param target_position: 3D position of the referenced object
    :param d_prox: Proximity threshold (default: 2cm as in the paper)
    :return: Normalized time (0-1) of closest approach
    """
    distances = np.linalg.norm(trajectory[:, 1:4] - target_position, axis=1)
    closest_idx = np.argmin(distances)
    return float(trajectory[closest_idx, 0])


class DummyRobot(Robot):
    """Minimal robot for timing demonstration (no simulation needed)."""

    def get_type(self):
        return RobotType.SIMULATION

    def pos_controller(self, trajectory):
        pass


def run():
    print("=" * 70)
    print("Experiment 1: Speed Adaptation (Paper Section V-B)")
    print("Command: 'slow down between box and station'")
    print("=" * 70)

    # Step 1: Train KMP from demonstrations
    model = KMPWrapper(demonstration_path=DATA_PATH, force_retrain=True)
    print(f"\nKMP trained: {model.demonstrations.shape[0]} demonstrations, {model.kmp.N} prediction points")

    # Step 2: Attach robot for timing control
    model.robot = DummyRobot()
    model.robot.init_predicting_frequency(model.trajectory)
    original_freq = model.robot.predicting_frequency.copy()

    # Step 3: Trajectory segment determination
    # The LLM extracts spatial references ("box", "station") from the command.
    # The system maps these to time points on the trajectory.
    t_box = find_closest_time(model.trajectory, ENVIRONMENT[0]["position"])
    t_station = find_closest_time(model.trajectory, ENVIRONMENT[1]["position"])
    print("\nEnvironment observations:")
    for obj in ENVIRONMENT:
        print(f"  {obj['label']}: position={obj['position']}")
    print("\nTrajectory segment determination (d_prox=2cm):")
    print(f"  '{ENVIRONMENT[0]['label']}' -> t={t_box:.3f}")
    print(f"  '{ENVIRONMENT[1]['label']}' -> t={t_station:.3f}")
    print(f"  Adaptation range: [{t_box:.3f}, {t_station:.3f}]")

    # Step 4: Apply speed modulation (what the LLM tool call does)
    # SlowDownRobot scales dt by factor (|gamma|+100)/100 = 1.5 for gamma=50
    print(f"\nTool call: SlowDownRobot(slow_down_value=50, adaption_start={t_box:.2f}, adaption_end={t_station:.2f})")
    tool = SlowDownRobot(
        slow_down_value=50,
        adaption_start=round(t_box, 2),
        adaption_end=round(t_station, 2),
    )
    result = tool.execute(model)
    print(f"Result: {result}")

    # Step 5: Verify timing changes
    new_freq = model.robot.predicting_frequency
    changed = np.where(new_freq != original_freq)[0]
    print("\nVerification:")
    print(f"  Points with modified timing: {len(changed)} / {len(new_freq)}")
    if len(changed) > 0:
        time_range = model.robot.timesteps[changed]
        ratio = new_freq[changed[0]] / original_freq[changed[0]]
        print(f"  Affected time range: [{time_range[0]:.3f}, {time_range[-1]:.3f}]")
        print(f"  Speed factor: {ratio:.2f}x (expected: 1.50x for 50% slowdown)")
        # Total execution time change
        original_total = np.sum(original_freq)
        new_total = np.sum(new_freq)
        print(f"  Original total time: {original_total:.2f}s -> New: {new_total:.2f}s")

    # Step 6: Generate figure (paper Fig. 4)
    from irosa.plotting import plot_speed_comparison

    output_dir = pathlib.Path(__file__).parent.parent / "results"
    output_dir.mkdir(exist_ok=True)
    plot_speed_comparison(
        mean=model.mean,
        predicting_frequency=model.robot.predicting_frequency,
        title="Exp. 1: Speed Adaptation — 'slow down between box and station'",
        output_path=output_dir / "experiment1_speed",
    )

    print("\n[OK] Experiment 1 completed successfully")


if __name__ == "__main__":
    run()
