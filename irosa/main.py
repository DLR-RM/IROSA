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

"""Main entry point for the IROSA interactive robot skill adaptation system.

Usage:
    python -m irosa.main --demo data/demonstrations.npz --llm-model qwen2.5:72b

See README.md for full documentation.
"""

from __future__ import annotations

import argparse
import logging
import pathlib

from irosa.core.llm_client import LLMClient, LLMConfig
from irosa.models.kmp import KMPWrapper
from irosa.tool_definitions import AVAILABLE_TOOLS_GENERAL, AVAILABLE_TOOLS_KMP

logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    'You are "IROSA", a robot assistant which helps to interact with a robot and modulate robot skills. '
    "You will act as a robot and stay in character. "
    "Your task is to {task_description}. "
    "Respond to the following query using your tool calls ONLY. "
    "You may call multiple tools per one query."
)


def main(
    demonstration_path: str | pathlib.Path,
    llm_model: str = "qwen2.5:72b",
    llm_base_url: str = "http://localhost:11434/v1",
    llm_api_key: str = "ollama",
    task_description: str = "adapt a trajectory of a robot, which starts at time 0.0 and ends at time 1.0",
    use_simulation: bool = False,
) -> None:
    """Run the IROSA interactive loop.

    :param demonstration_path: Path to demonstration data file (.npz or .pickle)
    :param llm_model: LLM model name
    :param llm_base_url: LLM API base URL
    :param llm_api_key: LLM API key
    :param task_description: Task description for the LLM
    :param use_simulation: Whether to use PyBullet simulation
    """
    demonstration_path = pathlib.Path(demonstration_path)
    if not demonstration_path.exists():
        raise FileNotFoundError(f"Demonstration file not found: {demonstration_path}")

    # Initialize robot
    robot = None
    if use_simulation:
        from irosa.robots.sim import Simulator

        robot = Simulator()

    # Initialize KMP model
    available_tools = AVAILABLE_TOOLS_GENERAL.copy() + AVAILABLE_TOOLS_KMP.copy()
    model = KMPWrapper(robot=robot, demonstration_path=demonstration_path)
    model.train_skill()

    if robot is not None:
        model.run()

    # Initialize LLM
    llm_config = LLMConfig(model=llm_model, base_url=llm_base_url, api_key=llm_api_key)
    system_prompt = SYSTEM_PROMPT.format(task_description=task_description)
    llm_client = LLMClient(config=llm_config, tools=available_tools, system_prompt=system_prompt)

    print(f"\nIROSA initialized with model: {llm_model}")
    print(f"Demonstration: {demonstration_path}")
    print(f"Available tools: {[t.get_name() for t in available_tools]}")
    print("\nType your commands to adapt the robot skill. Type 'quit' to exit.\n")

    # Interactive loop (Algorithm 1 from the paper)
    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            break

        if query.lower() in ("quit", "exit", "q"):
            break

        if not query:
            continue

        # Query the LLM
        print("Processing...")
        response = llm_client.query(query, model)
        print(f"\n{response}\n")

        # Execute updated trajectory on robot
        if robot is not None:
            model.run()


def main_without_llm(
    demonstration_path: str | pathlib.Path,
    use_simulation: bool = False,
) -> KMPWrapper:
    """Initialize IROSA without LLM connection (for programmatic use).

    :param demonstration_path: Path to demonstration data file (.npz or .pickle)
    :param use_simulation: Whether to use PyBullet simulation
    :return: Trained KMPWrapper instance
    """
    demonstration_path = pathlib.Path(demonstration_path)
    if not demonstration_path.exists():
        raise FileNotFoundError(f"Demonstration file not found: {demonstration_path}")

    robot = None
    if use_simulation:
        from irosa.robots.sim import Simulator

        robot = Simulator()

    model = KMPWrapper(robot=robot, demonstration_path=demonstration_path)
    model.train_skill()

    if robot is not None:
        model.run()

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IROSA: Interactive Robot Skill Adaptation")
    parser.add_argument("--demo", required=True, help="Path to demonstration data file (.npz or .pickle)")
    parser.add_argument("--llm-model", default="qwen2.5:72b", help="LLM model name (default: qwen2.5:72b)")
    parser.add_argument("--llm-url", default="http://localhost:11434/v1", help="LLM API base URL")
    parser.add_argument("--llm-key", default="ollama", help="LLM API key")
    parser.add_argument("--task", default="", help="Task description for the LLM")
    parser.add_argument("--sim", action="store_true", help="Use PyBullet simulation")

    args = parser.parse_args()

    main(
        demonstration_path=args.demo,
        llm_model=args.llm_model,
        llm_base_url=args.llm_url,
        llm_api_key=args.llm_key,
        task_description=args.task,
        use_simulation=args.sim,
    )
