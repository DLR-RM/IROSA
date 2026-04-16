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

"""OpenAI-compatible LLM client for tool-based interaction.

Supports any OpenAI-compatible API endpoint (OpenAI, Ollama, vLLM, LM Studio, etc.).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from openai import OpenAI

from irosa.core.tool import Tool, Toolkit

logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    """Configuration for the LLM connection."""

    model: str = "qwen2.5:72b"
    base_url: str = "http://localhost:11434/v1"
    api_key: str = "ollama"
    temperature: float = 0.0
    max_retries: int = 3

    @classmethod
    def ollama(cls, model: str = "qwen2.5:72b", host: str = "localhost", port: int = 11434) -> LLMConfig:
        """Create config for Ollama backend."""
        return cls(model=model, base_url=f"http://{host}:{port}/v1", api_key="ollama")

    @classmethod
    def vllm(cls, model: str = "Qwen/Qwen2.5-72B-Instruct", host: str = "localhost", port: int = 8000) -> LLMConfig:
        """Create config for vLLM backend."""
        return cls(model=model, base_url=f"http://{host}:{port}/v1", api_key="vllm")

    @classmethod
    def openai(cls, model: str = "gpt-4o", api_key: str = "") -> LLMConfig:
        """Create config for OpenAI API."""
        return cls(model=model, base_url="https://api.openai.com/v1", api_key=api_key)


@dataclass
class ChatMessage:
    """A message in the chat history."""

    role: str  # "system", "user", "assistant", "tool"
    content: str
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None


class LLMClient:
    """Client for LLM interaction with tool calling support.

    :param config: LLM connection configuration
    :param tools: List of Tool classes available for the LLM
    :param system_prompt: System prompt for the LLM
    """

    def __init__(
        self,
        config: LLMConfig,
        tools: list[type[Tool]],
        system_prompt: str = "",
    ) -> None:
        self.config = config
        self.toolkit = Toolkit(tools)
        self.client = OpenAI(base_url=config.base_url, api_key=config.api_key)
        self.history: list[dict[str, Any]] = []

        if system_prompt:
            self.history.append({"role": "system", "content": system_prompt})

    def query(self, user_message: str, model: Any) -> str:
        """Send a query to the LLM and execute any tool calls.

        :param user_message: The user's natural language message
        :param model: The model instance to pass to tool execution
        :return: Final response string
        """
        self.history.append({"role": "user", "content": user_message})

        openai_tools = self.toolkit.get_openai_tools()

        for attempt in range(self.config.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.config.model,
                    messages=self.history,  # type: ignore[arg-type]
                    tools=openai_tools if openai_tools else None,  # type: ignore[arg-type]
                    temperature=self.config.temperature,
                )
            except Exception as e:
                logger.error("LLM API error: %s", e)
                return f"Error communicating with LLM: {e}"

            message = response.choices[0].message

            if message.tool_calls:
                # Add assistant message with tool calls to history
                self.history.append(message.model_dump())

                tool_calls_parsed = []
                for tc in message.tool_calls:
                    tool_calls_parsed.append(
                        {
                            "id": tc.id,
                            "function": {
                                "name": tc.function.name,  # type: ignore[union-attr]
                                "arguments": tc.function.arguments,  # type: ignore[union-attr]
                            },
                        }
                    )

                # Execute tools
                results = self.toolkit.parse_and_execute(tool_calls_parsed, model)

                # Add tool results to history
                for tc, result in zip(message.tool_calls, results):
                    self.history.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": result,
                        }
                    )

                logger.info("Tool calls executed: %s", [tc.function.name for tc in message.tool_calls])  # type: ignore[union-attr]

                # If this was the last retry, return what we have
                if attempt == self.config.max_retries:
                    return "\n".join(results)

                # Continue the loop to get the final response after tool execution
                continue

            # No tool calls - return the text response
            content = message.content or ""
            self.history.append({"role": "assistant", "content": content})
            return content

        return "Error: Max retries exceeded"

    def reset_history(self) -> None:
        """Clear chat history, keeping the system prompt."""
        system_msgs = [m for m in self.history if m.get("role") == "system"]
        self.history = system_msgs
