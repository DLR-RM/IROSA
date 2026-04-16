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

"""Plotting utilities for IROSA experiments.

Generates publication-quality figures matching the paper's style:
- 3-row subplots (x, y, z) over normalized time
- Demonstration scatter, KMP mean line, uncertainty band
- Via-point markers, obstacle visualization
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = {
    "prediction": "#1f77b4",
    "demonstration": "#ff7f0e",
    "uncertainty": "#1f77b4",
    "via_points": "#d62728",
}

SAVE_KW = {"dpi": 300, "bbox_inches": "tight", "facecolor": "white", "edgecolor": "none"}


def _apply_style():
    plt.style.use("default")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 14,
            "axes.linewidth": 1.2,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linewidth": 0.8,
            "grid.linestyle": "--",
        }
    )


def _setup_ax(ax):
    ax.tick_params(axis="both", which="major", labelsize=12, length=5, width=1.2, color="#333333")
    ax.set_facecolor("#fafafa")
    ax.spines["left"].set_color("#333333")
    ax.spines["bottom"].set_color("#333333")


def _compute_ylim(values_list: list[np.ndarray], margin_factor: float = 0.1) -> tuple[float, float]:
    """Compute y-axis limits from data with margin, ignoring the uncertainty band."""
    all_vals = np.concatenate(values_list)
    vmin, vmax = float(np.min(all_vals)), float(np.max(all_vals))
    margin = (vmax - vmin) * margin_factor
    return vmin - margin, vmax + margin


def plot_trajectory(
    mean: np.ndarray,
    x_in: np.ndarray,
    cov: np.ndarray | None = None,
    demonstrations: np.ndarray | None = None,
    via_points: dict | None = None,
    title: str = "KMP Trajectory Predictions",
    output_path: str | Path = "trajectory_plot",
    scale_std: float = 10.0,
):
    """Plot x/y/z trajectory over normalized time (paper Figs. 4, 5, 6).

    :param mean: KMP mean prediction (N, >=3)
    :param x_in: Time points (N,)
    :param cov: Full covariance matrix (N*D, N*D) or None
    :param demonstrations: Raw demos (n_demos, n_points, n_vars) or None
    :param via_points: Dict with 'input' and 'output' lists
    :param title: Figure title
    :param output_path: Save path (without extension)
    :param scale_std: Multiplier for uncertainty band width
    """
    _apply_style()

    fig, axes = plt.subplots(3, 1, figsize=(12, 5), facecolor="white", constrained_layout=True)
    fig.suptitle(title, fontsize=16, fontweight="bold")

    time_steps = x_in
    axis_labels = ["X Position [m]", "Y Position [m]", "Z Position [m]"]

    # Extract std from covariance diagonal
    std = None
    if cov is not None:
        cov_diag = np.diagonal(cov)
        std = np.sqrt(np.abs(cov_diag)).reshape(mean.shape)

    for i in range(3):
        ax = axes[i]
        _setup_ax(ax)

        # Collect data values for y-axis limits (mean + demos, NOT uncertainty)
        ylim_data = [mean[:, i]]

        # Demonstrations
        if demonstrations is not None:
            if demonstrations.ndim == 3:
                for k in range(demonstrations.shape[0]):
                    ax.scatter(
                        demonstrations[k, :, 0],
                        demonstrations[k, :, i + 1],
                        color=COLORS["demonstration"],
                        alpha=0.3,
                        s=1.0,
                        zorder=1,
                        label="Demonstrations" if k == 0 and i == 0 else None,
                    )
                    ylim_data.append(demonstrations[k, :, i + 1])

        # Uncertainty band (clipped to y-axis limits later)
        if std is not None:
            ax.fill_between(
                time_steps,
                mean[:, i] - std[:, i] * scale_std,
                mean[:, i] + std[:, i] * scale_std,
                color=COLORS["uncertainty"],
                alpha=0.3,
                zorder=2,
                label="Model Uncertainty" if i == 0 else None,
            )

        # Mean prediction
        ax.plot(
            time_steps,
            mean[:, i],
            color=COLORS["prediction"],
            linewidth=3.0,
            zorder=4,
            label="KMP Prediction" if i == 0 else None,
        )

        # Via-points
        if via_points is not None and "input" in via_points and len(via_points["input"]) > 0:
            vp_times = np.array(via_points["input"]).flatten()
            vp_outputs = np.array(via_points["output"])
            if vp_outputs.ndim == 1:
                vp_outputs = vp_outputs.reshape(1, -1)
            # Downsample for readability
            if len(vp_times) > 50:
                step = max(1, len(vp_times) // 50)
                vp_times = vp_times[::step]
                vp_outputs = vp_outputs[::step]
            if vp_outputs.shape[1] > i:
                ax.scatter(
                    vp_times,
                    vp_outputs[:, i],
                    edgecolors=COLORS["via_points"],
                    facecolors="none",
                    s=80,
                    linewidth=2.0,
                    marker="o",
                    zorder=5,
                    label="Via-points" if i == 0 else None,
                )
                ylim_data.append(vp_outputs[:, i])

        # Set y-axis limits based on data only (not the uncertainty band)
        ymin, ymax = _compute_ylim(ylim_data, margin_factor=0.15)
        ax.set_ylim(ymin, ymax)

        ax.set_xlim(0, 1)
        ax.set_ylabel(axis_labels[i], fontsize=14, color="#333333")
        if i < 2:
            ax.tick_params(labelbottom=False)

    axes[2].set_xlabel("Normalized Time", fontsize=14, color="#333333")
    axes[0].legend(loc="upper right", fontsize=10, framealpha=0.9)

    output_path = Path(output_path)
    plt.savefig(str(output_path) + ".svg", **SAVE_KW)
    plt.savefig(str(output_path) + ".png", **SAVE_KW)
    plt.close(fig)
    print(f"Plot saved: {output_path}.svg / .png")


def plot_speed_comparison(
    mean: np.ndarray,
    predicting_frequency: np.ndarray,
    title: str = "Speed Adaptation",
    output_path: str | Path = "speed_plot",
):
    """Plot trajectory with dot spacing showing speed changes (paper Fig. 4).

    Points are plotted at cumulative-time positions; wider spacing = slower.

    :param mean: KMP mean (N, >=3)
    :param predicting_frequency: dt array (N,)
    :param title: Figure title
    :param output_path: Save path (without extension)
    """
    _apply_style()

    fig, axes = plt.subplots(3, 1, figsize=(12, 5), facecolor="white", constrained_layout=True)
    fig.suptitle(title, fontsize=16, fontweight="bold")

    # Convert dt to cumulative time, then normalize
    cum_time = np.concatenate([[0], np.cumsum(predicting_frequency)[:-1]])
    time_steps = cum_time / cum_time[-1]

    axis_labels = ["X Position [m]", "Y Position [m]", "Z Position [m]"]
    step = 10  # sample every 10th point

    for i in range(3):
        ax = axes[i]
        _setup_ax(ax)

        # Background line
        ax.plot(time_steps, mean[:, i], color=COLORS["prediction"], linewidth=2.5, alpha=0.3, zorder=2)

        # Dots showing time intervals
        ax.scatter(
            time_steps[::step],
            mean[::step, i],
            color=COLORS["prediction"],
            s=40,
            zorder=4,
            alpha=0.8,
            label="KMP Prediction" if i == 0 else None,
        )

        # Set y-axis limits from data
        ymin, ymax = _compute_ylim([mean[:, i]], margin_factor=0.15)
        ax.set_ylim(ymin, ymax)
        ax.set_xlim(0, 1)

        ax.set_ylabel(axis_labels[i], fontsize=14, color="#333333")
        if i < 2:
            ax.tick_params(labelbottom=False)

    axes[2].set_xlabel("Normalized Time", fontsize=14, color="#333333")
    axes[0].legend(loc="upper right", fontsize=10, framealpha=0.9)

    output_path = Path(output_path)
    plt.savefig(str(output_path) + ".svg", **SAVE_KW)
    plt.savefig(str(output_path) + ".png", **SAVE_KW)
    plt.close(fig)
    print(f"Plot saved: {output_path}.svg / .png")
