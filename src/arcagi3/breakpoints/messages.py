from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TypedDict, Union


class AgentBreakpoint(TypedDict, total=False):
    request_id: str
    point_id: str
    phase: str
    payload: Dict[str, Any]


class GlobalStateSnapshot(TypedDict):
    paused: bool
    breakpoints: Dict[str, bool]


class AgentStateSnapshot(TypedDict, total=False):
    agent_id: str
    config: Optional[str]
    card_id: Optional[str]
    game_id: Optional[str]
    status: str
    score: int
    last_step: Optional[str]
    current_breakpoint: Optional[AgentBreakpoint]
    breakpoints: Dict[str, bool]
    play_num: Optional[int]
    play_action_counter: Optional[int]
    action_counter: Optional[int]
    schema: Dict[str, Any]


ServerStateSnapshotMessage = TypedDict(
    "ServerStateSnapshotMessage",
    {
        "type": str,
        "global": GlobalStateSnapshot,  # "global" is a keyword but needed for JSON
        "agents": List[AgentStateSnapshot],
    },
)


class BreakpointPendingMessage(TypedDict):
    type: str
    agent: AgentStateSnapshot
    point_id: str
    payload: Dict[str, Any]
    request_id: str


class BreakpointResolvedMessage(TypedDict):
    type: str
    agent_id: str
    point_id: str
    request_id: str


class AgentUpdatedMessage(TypedDict):
    type: str
    agent: AgentStateSnapshot


GlobalStateUpdatedMessage = TypedDict(
    "GlobalStateUpdatedMessage",
    {
        "type": str,
        "global": GlobalStateSnapshot,  # "global" is a keyword but needed for JSON
    },
)


ServerMessage = Union[
    ServerStateSnapshotMessage,
    BreakpointPendingMessage,
    BreakpointResolvedMessage,
    AgentUpdatedMessage,
    GlobalStateUpdatedMessage,
]


@dataclass(frozen=True)
class AgentHelloPayload:
    agent_id: str
    config: Optional[str]
    card_id: Optional[str]
    game_id: Optional[str]
    schema: Dict[str, Any]
    breakpoints: Dict[str, bool]
