from .base import BaseTool, ToolExecutionContext, ToolOutcome, ToolRegistry
from .clarification import (
    ClarificationOption,
    ClarificationRequest,
    ConductorClarificationHandler,
    InteractiveClarificationHandler,
    RequestClarificationTool,
)


def default_tool_registry(model_registry, *, ask_user: bool = False) -> ToolRegistry:
    """Route clarification to GPT-6 Astra unless user input is explicitly enabled."""
    handler = (
        InteractiveClarificationHandler()
        if ask_user
        else ConductorClarificationHandler(model_registry)
    )
    return ToolRegistry([
        RequestClarificationTool(handler),
    ])


__all__ = [
    "BaseTool", "ToolExecutionContext", "ToolOutcome", "ToolRegistry",
    "ClarificationOption", "ClarificationRequest",
    "InteractiveClarificationHandler", "ConductorClarificationHandler",
    "RequestClarificationTool", "default_tool_registry",
]
