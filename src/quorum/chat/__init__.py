"""Talking to a project: naming its meetings, then asking and telling.

Everything the agent knows was already reachable through eight separate
commands. What was missing was a way to *refer* to a meeting in a sentence, and
somewhere to have a conversation about it that remembers the last thing you
said.
"""

from quorum.chat.agent import ChatAgent, ChatTurn, Conversation
from quorum.chat.answer import Coverage, GroundedAnswer, answer_question
from quorum.chat.naming import (
    MeetingRef,
    auto_handle,
    list_meetings,
    resolve_meeting,
    set_handle,
)
from quorum.chat.tools import TOOLS, ToolContext, ToolRequest, ToolResult, run_tool

__all__ = [
    "ChatAgent",
    "ChatTurn",
    "Conversation",
    "Coverage",
    "GroundedAnswer",
    "answer_question",
    "MeetingRef",
    "auto_handle",
    "list_meetings",
    "resolve_meeting",
    "set_handle",
    "TOOLS",
    "ToolContext",
    "ToolRequest",
    "ToolResult",
    "run_tool",
]
