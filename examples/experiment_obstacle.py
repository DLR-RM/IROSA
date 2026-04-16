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

"""Experiment 3: Obstacle Avoidance via Repulsion Points (Section V-D of the paper).

Reproduces the obstacle avoidance experiment from:
    "Interactive robot skill adaptation using natural language"

Scenario:
    A blue industrial box (15 x 22.5 x 23.5 cm) is placed next to the bearing
    ring box, creating a collision hazard with the learned trajectory. The user
    says: "Please avoid the blue box". The LLM identifies the obstacle from
    environment observations, extracts its position and dimensions, and calls
    the AddRepulsionPoint tool. The tool constructs a signed distance field
    (SDF) from the box dimensions, evaluates the trajectory for collisions,
    and inserts via-points at corrected positions to generate a collision-free
    path.

What this example demonstrates:
    1. Training KMP from demonstrations
    2. OBB (oriented bounding box) obstacle definition from dimensions
    3. SDF-based trajectory correction
    4. Automatic via-point insertion for collision avoidance
    5. Safety margin enforcement (delta_safe = 2cm as in the paper)
"""

import pathlib

import numpy as np

from irosa.models.kmp import KMPWrapper

DATA_PATH = pathlib.Path(__file__).parent.parent / "data" / "demonstrations.npz"


# -- Environment observations (manually provided, as in the paper) --
# The blue box dimensions are from the paper: 15 x 22.5 x 23.5 cm
ENVIRONMENT = [
    {"position": np.array([0.82, 0.47, 0.13]), "dimensions": None, "label": "bearing ring box"},
    {"position": np.array([0.47, -0.24, 0.08]), "dimensions": None, "label": "measurement station"},
    {
        "position": np.array([0.75, 0.35, 0.23]),  # next to bearing ring box
        "dimensions": [0.150, 0.225, 0.235],  # width, length, height in meters
        "label": "blue box",
    },
]

# Safety margin from the paper (Section V-D): delta_safe = 2cm
SAFETY_MARGIN = 0.02


def run():
    print("=" * 70)
    print("Experiment 3: Obstacle Avoidance (Paper Section V-D)")
    print("Command: 'Please avoid the blue box'")
    print("=" * 70)

    # Step 1: Train KMP from demonstrations
    model = KMPWrapper(demonstration_path=DATA_PATH, force_retrain=True)
    original_mean = model.mean.copy()
    print(f"\nKMP trained: {model.demonstrations.shape[0]} demonstrations, {model.kmp.N} prediction points")

    # Step 2: Show environment
    print("\nEnvironment observations:")
    for obj in ENVIRONMENT:
        dims_str = f", dimensions={obj['dimensions']}" if obj["dimensions"] else ""
        print(f"  {obj['label']}: position={obj['position']}{dims_str}")

    # Step 3: Identify the obstacle (LLM extracts "blue box" from command)
    obstacle = ENVIRONMENT[2]
    obstacle_pos = obstacle["position"]
    obstacle_dims = obstacle["dimensions"]

    # Check initial clearance
    distances = np.linalg.norm(original_mean[:, :3] - obstacle_pos, axis=1)
    min_dist_before = np.min(distances)
    min_dist_idx = np.argmin(distances)
    min_dist_time = model.x_in[min_dist_idx]
    print("\nPre-correction analysis:")
    print(f"  Obstacle: '{obstacle['label']}' at {obstacle_pos}")
    print(f"  Dimensions: {obstacle_dims[0] * 100:.1f} x {obstacle_dims[1] * 100:.1f} x {obstacle_dims[2] * 100:.1f} cm")
    print(f"  Safety margin: {SAFETY_MARGIN * 100:.0f} cm")
    print(f"  Closest approach: {min_dist_before * 100:.1f} cm at t={min_dist_time:.3f}")

    # Step 4: Apply obstacle avoidance
    # The LLM calls AddRepulsionPoint with obstacle position and dimensions.
    # Internally, this constructs an OBB SDF, corrects the trajectory, and
    # inserts via-points at corrected positions.
    print("\nTool call: AddRepulsionPoint(")
    print(f"  position={obstacle_pos.tolist()},")
    print(f"  radius={max(obstacle_dims) / 2:.3f})")
    print(f"  (with safety_margin={SAFETY_MARGIN})")

    # Use the add_repulsion_point method directly with OBB dimensions
    model.add_repulsion_point(
        position=obstacle_pos,
        dimensions=obstacle_dims,
        safety_margin=SAFETY_MARGIN,
    )

    # Step 5: Verify collision-free trajectory
    new_distances = np.linalg.norm(model.mean[:, :3] - obstacle_pos, axis=1)
    min_dist_after = np.min(new_distances)
    n_viapoints = len(model.kmp.via_points["input"])

    displacement = np.linalg.norm(model.mean[:, :3] - original_mean[:, :3], axis=1)
    max_deviation = np.max(displacement)
    max_dev_idx = np.argmax(displacement)

    print("\nPost-correction results:")
    print(f"  Minimum distance to obstacle: {min_dist_before * 100:.1f} cm -> {min_dist_after * 100:.1f} cm")
    print(f"  Distance improvement: +{(min_dist_after - min_dist_before) * 100:.1f} cm")
    print(f"  Via-points inserted: {n_viapoints}")
    print(f"  Maximum trajectory deviation: {max_deviation * 100:.1f} cm at t={model.x_in[max_dev_idx]:.3f}")

    # Verify start/end preserved
    start_disp = np.linalg.norm(model.mean[0, :3] - original_mean[0, :3])
    end_disp = np.linalg.norm(model.mean[-1, :3] - original_mean[-1, :3])
    print(f"  Start point preserved: {start_disp * 100:.2f} cm deviation")
    print(f"  End point preserved: {end_disp * 100:.2f} cm deviation")

    # Step 6: Generate figure (paper Fig. 6)
    from irosa.plotting import plot_trajectory

    output_dir = pathlib.Path(__file__).parent.parent / "results"
    output_dir.mkdir(exist_ok=True)
    plot_trajectory(
        mean=model.mean,
        x_in=model.x_in,
        cov=model.cov,
        demonstrations=model.demonstrations,
        via_points=model.kmp.via_points,
        title="Exp. 3: Obstacle Avoidance — 'Please avoid the blue box'",
        output_path=output_dir / "experiment3_obstacle",
    )

    print("\n[OK] Experiment 3 completed successfully")


if __name__ == "__main__":
    run()
