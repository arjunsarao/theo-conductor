from .base import BaseTool, ToolExecutionContext, ToolOutcome, ToolRegistry
from .clarification import (
    ClarificationOption,
    ClarificationRequest,
    ConductorClarificationHandler,
    InteractiveClarificationHandler,
    RequestClarificationTool,
)


def default_tool_registry(model_registry, *, ask_user: bool = False, conductor_model: str | None = None) -> ToolRegistry:
    """Route clarification to the active conductor unless user input is enabled."""
    handler = (
        InteractiveClarificationHandler()
        if ask_user
        else ConductorClarificationHandler(
            model_registry, model_id=conductor_model
        )
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
