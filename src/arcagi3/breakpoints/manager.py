from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from arcagi3.breakpoints.client import BreakpointClient
from arcagi3.breakpoints.spec import BreakpointSpec

logger = logging.getLogger(__name__)

ApplyOverridesFn = Callable[[Dict[str, Any], Dict[str, Any], Any], Dict[str, Any]]


@dataclass(frozen=True)
class BreakpointHook:
    point_id: str
    apply_overrides: Optional[ApplyOverridesFn] = None


class BreakpointManager:
    def __init__(
        self,
        *,
        enabled: bool,
        ws_url: str,
        spec: Optional[BreakpointSpec] = None,
        hooks: Optional[Dict[str, BreakpointHook]] = None,
        agent_id: Optional[str] = None,
    ) -> None:
        self.enabled = enabled
        self._spec = spec or BreakpointSpec()
        self._hooks = hooks or {}
        self._client = BreakpointClient(
            breakpoint_ws_url=ws_url,
            enable_breakpoints=enabled,
            agent_id=agent_id,
        )
        self._client.set_schema(self._spec)

    def update_spec(self, spec: BreakpointSpec) -> None:
        self._spec = spec
        self._client.set_schema(spec)

    def update_hooks(self, hooks: Dict[str, BreakpointHook]) -> None:
        self._hooks = hooks

    def update_identity(
        self,
        *,
        config: Optional[str] = None,
        card_id: Optional[str] = None,
        game_id: Optional[str] = None,
    ) -> None:
        self._client.set_identity(config=config, card_id=card_id, game_id=game_id)

    def has_point(self, point_id: str) -> bool:
        return point_id in self._spec.point_ids()

    def pause(
        self,
        point_id: str,
        payload: Dict[str, Any],
        *,
        context: Optional[Any] = None,
        status_before: Optional[str] = "PAUSED",
        status_after: Optional[str] = "RUNNING",
        step_name: Optional[str] = None,
        score: Optional[int] = None,
    ) -> Dict[str, Any]:
        if not self.enabled or not self.has_point(point_id):
            return payload

        # Extract play_num and play_action_counter from context if available
        play_num = None
        play_action_counter = None
        if context is not None and hasattr(context, "game"):
            game = context.game
            if hasattr(game, "play_num"):
                play_num = game.play_num
            if hasattr(game, "play_action_counter"):
                play_action_counter = game.play_action_counter

        update_payload: Dict[str, Any] = {"type": "agent_update", "agent_id": self._client.agent_id}
        if status_before:
            update_payload["status"] = status_before
        if step_name:
            update_payload["step_name"] = step_name
        if score is not None:
            update_payload["score"] = score
        if play_num is not None:
            update_payload["play_num"] = play_num
        if play_action_counter is not None:
            update_payload["play_action_counter"] = play_action_counter
        self._client.send_agent_update(update_payload)

        overrides = self._client.await_breakpoint(point_id, payload)

        update_payload = {"type": "agent_update", "agent_id": self._client.agent_id}
        if status_after:
            update_payload["status"] = status_after
        if step_name:
            update_payload["step_name"] = step_name
        if score is not None:
            update_payload["score"] = score
        if play_num is not None:
            update_payload["play_num"] = play_num
        if play_action_counter is not None:
            update_payload["play_action_counter"] = play_action_counter
        self._client.send_agent_update(update_payload)

        hook = self._hooks.get(point_id)
        return apply_breakpoint_overrides(payload, overrides, hook, context)


def apply_breakpoint_overrides(
    payload: Dict[str, Any],
    overrides: Any,
    hook: Optional[BreakpointHook],
    context: Optional[Any],
) -> Dict[str, Any]:
    if not isinstance(overrides, dict):
        return payload
    if hook and hook.apply_overrides:
        try:
            return hook.apply_overrides(payload, overrides, context)
        except Exception:
            logger.debug(
                "Failed to apply breakpoint overrides for %s", hook.point_id, exc_info=True
            )
            return payload
    return overrides
