"""Optional Microsoft AutoGen implementation team backed by local Qwen.

This is deliberately separate from the deterministic trading loop.  It can be
used for offline code review/planning but can never place or modify MT5 orders.
Run a local OpenAI-compatible server containing Qwen2.5-Coder-7B-Instruct first.
"""

from __future__ import annotations

from typing import Any

from config import Settings


def build_implementation_team(settings: Settings) -> tuple[Any, Any]:
    from autogen_agentchat.agents import AssistantAgent
    from autogen_agentchat.teams import RoundRobinGroupChat
    from autogen_agentchat.conditions import MaxMessageTermination
    from autogen_ext.models.openai import OpenAIChatCompletionClient

    client = OpenAIChatCompletionClient(
        model=settings.qwen_model,
        base_url=settings.qwen_base_url,
        api_key="local-not-used",
        temperature=0,
        parallel_tool_calls=False,
        model_info={
            "vision": False,
            "function_calling": False,
            "json_output": False,
            "family": "unknown",
            "structured_output": False,
        },
    )
    implementer = AssistantAgent(
        "system_implementer",
        model_client=client,
        system_message="Design deterministic, typed Python components. Never issue trading instructions or orders.",
    )
    reviewer = AssistantAgent(
        "risk_reviewer",
        model_client=client,
        system_message="Review code for fail-closed behavior, risk caps, offline isolation, and test gaps.",
    )
    team = RoundRobinGroupChat([implementer, reviewer], termination_condition=MaxMessageTermination(4))
    return team, client


async def review_task(task: str, settings: Settings) -> str:
    team, client = build_implementation_team(settings)
    try:
        result = await team.run(task=task)
        return "\n".join(str(message.content) for message in result.messages if hasattr(message, "content"))
    finally:
        await client.close()

