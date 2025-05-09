from typing import Protocol

from inspect_ai.solver._task_state import TaskState
from inspect_ai.tool._tool_call import ToolCall, ToolCallView

from ._approval import Approval


class Approver(Protocol):
    """Protocol for approvers."""

    async def __call__(
        self,
        message: str,
        call: ToolCall,
        view: ToolCallView,
        state: TaskState | None = None,
    ) -> Approval:
        """
        Approve or reject a tool call.

        Args:
            message: Message genreated by the model along with the tool call.
            call: The tool call to be approved.
            view: Custom rendering of tool context and call.
            state: The current task state, if available.

        Returns:
            Approval: An Approval object containing the decision and explanation.
        """
        ...
