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

"""Experiment 2: Trajectory Correction via Via-point Insertion (Section V-C of the paper).

Reproduces the trajectory correction experiment from:
    "Interactive robot skill adaptation using natural language"

Scenario:
    After the baseline demonstrations, a new object (camera) is introduced to
    the workspace, positioned to the left of the robot. The user says:
    "Check the ring with the camera on the left". The LLM identifies the
    camera from environment observations, determines the pickup time using
    trajectory segment determination, and inserts a via-point at the camera
    position between the pickup and the measurement station.

What this example demonstrates:
    1. Training KMP from demonstrations
    2. Introducing new objects to the environment
    3. Via-point insertion using AddViaPointsAtTime tool
    4. Trajectory modification while preserving learned skill structure
"""

import pathlib

import numpy as np

from irosa.models.kmp import KMPWrapper
from irosa.tool_definitions.kmp import AddViaPointsAtTime, GetViaPoints

DATA_PATH = pathlib.Path(__file__).parent.parent / "data" / "demonstrations.npz"


# -- Environment observations (manually provided, as in the paper) --
ENVIRONMENT = [
    {"position": np.array([0.82, 0.47, 0.13]), "label": "bearing ring box"},
    {"position": np.array([0.47, -0.24, 0.08]), "label": "measurement station"},
    {"position": np.array([0.60, 0.30, 0.25]), "label": "camera"},  # new object
]


def find_closest_time(trajectory: np.ndarray, target_position: np.ndarray) -> float:
    """Map spatial reference to time on trajectory (Section IV-C)."""
    distances = np.linalg.norm(trajectory[:, 1:4] - target_position, axis=1)
    return float(trajectory[np.argmin(distances), 0])


def run():
    print("=" * 70)
    print("Experiment 2: Trajectory Correction via Via-point (Paper Section V-C)")
    print("Command: 'Check the ring with the camera on the left'")
    print("=" * 70)

    # Step 1: Train KMP from demonstrations
    model = KMPWrapper(demonstration_path=DATA_PATH, force_retrain=True)
    original_mean = model.mean.copy()
    print(f"\nKMP trained: {model.demonstrations.shape[0]} demonstrations, {model.kmp.N} prediction points")

    # Step 2: Show environment with new camera object
    print("\nEnvironment observations:")
    for obj in ENVIRONMENT:
        print(f"  {obj['label']}: position={obj['position']}")

    # Step 3: Trajectory segment determination
    # The LLM identifies "camera" from the command and looks it up in E
    camera = ENVIRONMENT[2]
    t_box = find_closest_time(model.trajectory, ENVIRONMENT[0]["position"])
    t_station = find_closest_time(model.trajectory, ENVIRONMENT[1]["position"])

    # Via-point time: between pickup (box) and insertion (station)
    t_camera = (t_box + t_station) / 2.0
    print("\nTrajectory segment determination:")
    print(f"  Ring pickup (box): t={t_box:.3f}")
    print(f"  Measurement station: t={t_station:.3f}")
    print(f"  Camera via-point time: t={t_camera:.3f} (between pickup and station)")

    # Step 4: Insert via-point at camera position
    # The LLM calls AddViaPointsAtTime with the camera position
    camera_pos = camera["position"]
    # Use the orientation from the current trajectory at that time
    t_idx = np.argmin(np.abs(model.x_in - t_camera))
    current_orientation = original_mean[t_idx, 3:7]  # qw, qx, qy, qz
    via_output = np.concatenate([camera_pos, current_orientation]).tolist()

    print("\nTool call: AddViaPointsAtTime(")
    print(f"  input_values=[{t_camera:.3f}],")
    print(f"  output_values=[[{camera_pos[0]:.3f}, {camera_pos[1]:.3f}, {camera_pos[2]:.3f}, ...]])")

    tool = AddViaPointsAtTime(
        input_values=[round(t_camera, 3)],
        output_values=[via_output],
    )
    result = tool.execute(model)
    print(f"Result: {result}")

    # Step 5: Verify trajectory modification
    new_mean = model.mean
    displacement = np.linalg.norm(new_mean[:, :3] - original_mean[:, :3], axis=1)
    max_displacement = np.max(displacement)
    max_disp_idx = np.argmax(displacement)
    max_disp_time = model.x_in[max_disp_idx]

    print("\nVerification:")
    print(f"  Maximum trajectory displacement: {max_displacement * 100:.1f} cm at t={max_disp_time:.3f}")

    # Check via-points
    vp_tool = GetViaPoints()
    print(f"  Via-points: {vp_tool.execute(model)}")

    # Show trajectory stays close at start and end (preserves skill structure)
    start_disp = np.linalg.norm(new_mean[0, :3] - original_mean[0, :3])
    end_disp = np.linalg.norm(new_mean[-1, :3] - original_mean[-1, :3])
    print(f"  Start point displacement: {start_disp * 100:.2f} cm (should be ~0)")
    print(f"  End point displacement: {end_disp * 100:.2f} cm (should be ~0)")

    # Step 6: Generate figure (paper Fig. 5)
    from irosa.plotting import plot_trajectory

    output_dir = pathlib.Path(__file__).parent.parent / "results"
    output_dir.mkdir(exist_ok=True)
    plot_trajectory(
        mean=model.mean,
        x_in=model.x_in,
        cov=model.cov,
        demonstrations=model.demonstrations,
        via_points=model.kmp.via_points,
        title="Exp. 2: Via-point Insertion — 'Check the ring with the camera'",
        output_path=output_dir / "experiment2_viapoint",
    )

    print("\n[OK] Experiment 2 completed successfully")


if __name__ == "__main__":
    run()
