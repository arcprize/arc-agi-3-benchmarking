from __future__ import annotations

from typing import Any, Dict, List, Optional

from arcagi3.agent import HUMAN_ACTIONS, HUMAN_ACTIONS_LIST, MultimodalAgent
from arcagi3.prompts import PromptManager
from arcagi3.schemas import GameStep
from arcagi3.utils.context import SessionContext, GameProgress
from arcagi3.utils.formatting import grid_to_text_matrix
from arcagi3.utils.image import image_to_base64, make_image_block
from arcagi3.utils.parsing import extract_json_from_response


class SwarmAgent(MultimodalAgent):
    """
    Swarm agent that coordinates multiple games in sync.

    Use `play_swarm(game_ids, max_rounds=...)` to run a synchronized loop.
    """

    def __init__(
        self,
        *args,
        use_vision: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.prompt_manager = PromptManager()
        self.use_vision = use_vision

    def _available_action_descriptions(self, context: SessionContext) -> List[str]:
        if context.game.available_actions:
            indices = [int(str(a)) for a in context.game.available_actions]
            return [
                HUMAN_ACTIONS[HUMAN_ACTIONS_LIST[i - 1]]
                for i in indices
                if 1 <= i <= len(HUMAN_ACTIONS_LIST)
            ]
        return list(HUMAN_ACTIONS.values())

    def _validate_action(self, context: SessionContext, action_name: str) -> bool:
        if not action_name or not action_name.startswith("ACTION"):
            return False
        if not context.game.available_actions:
            return True
        action_num = action_name.replace("ACTION", "")
        normalized_available = {str(a) for a in context.game.available_actions}
        return action_num in normalized_available

    def _build_game_summary(
        self,
        context: SessionContext,
        include_images: bool,
    ) -> List[Dict[str, Any]]:
        blocks: List[Dict[str, Any]] = []
        actions_list = self._available_action_descriptions(context)
        actions_text = "\n".join(f"- {a}" for a in actions_list)
        memory = context.datastore.get("memory_prompt", "")
        summary = (
            f"Game: {context.game.game_id}\n"
            f"State: {context.game.current_state}\n"
            f"Score: {context.game.current_score}\n"
            f"Available actions:\n{actions_text}\n"
        )
        if memory:
            summary += f"Memory:\n{memory}\n"

        blocks.append({"type": "text", "text": summary})

        if include_images:
            for img in context.frame_images:
                blocks.append(make_image_block(image_to_base64(img)))
        else:
            for i, grid in enumerate(context.frames.frame_grids):
                blocks.append(
                    {"type": "text", "text": f"Frame {i}:\n{grid_to_text_matrix(grid)}"}
                )

        return blocks

    def coordinate_actions(
        self,
        coordinator_context: SessionContext,
        game_contexts: Dict[str, SessionContext],
    ) -> Dict[str, Any]:
        want_vision = self.use_vision and bool(
            getattr(self.provider.model_config, "is_multimodal", False)
        )
        content: List[Dict[str, Any]] = []
        summaries: List[str] = []

        for game_id, ctx in game_contexts.items():
            summaries.append(game_id)
            content.extend(self._build_game_summary(ctx, include_images=want_vision))

        prompt_text = self.prompt_manager.render(
            "swarm_instruct",
            {"game_ids": ", ".join(summaries)},
        )
        content.append({"type": "text", "text": prompt_text})

        messages = [
            {"role": "system", "content": self.prompt_manager.render("system")},
            {"role": "user", "content": content},
        ]

        response = self.provider.call_with_tracking(coordinator_context, messages)
        message = self.provider.extract_content(response)
        return extract_json_from_response(message)

    def step(self, context: SessionContext) -> GameStep:
        coordinator_context = SessionContext(
            game=GameProgress(game_id=f"swarm-{context.game.game_id}")
        )
        plan = self.coordinate_actions(coordinator_context, {context.game.game_id: context})
        actions = plan.get("actions", []) if isinstance(plan, dict) else []
        action_item = actions[0] if actions else {}

        action_name = action_item.get("action")
        if not action_name:
            raise ValueError("No action in swarm response")
        if not self._validate_action(context, str(action_name)):
            raise ValueError(
                f"Invalid action '{action_name}' for available_actions={context.game.available_actions}"
            )

        action_payload: Dict[str, Any] = {"action": action_name}
        action_data = action_item.get("data")
        if isinstance(action_data, dict) and action_data:
            action_payload["data"] = dict(action_data)
        elif "x" in action_item or "y" in action_item:
            action_payload["x"] = action_item.get("x", 0)
            action_payload["y"] = action_item.get("y", 0)

        return GameStep(action=action_payload, reasoning={"swarm_plan": plan})

    def play_swarm(
        self,
        game_ids: List[str],
        max_rounds: int = 10,
    ) -> Dict[str, Dict[str, Any]]:
        coordinator_context = SessionContext(game=GameProgress(game_id="swarm"))
        game_contexts: Dict[str, SessionContext] = {}
        game_states: Dict[str, Dict[str, Any]] = {}

        for game_id in game_ids:
            state = self.game_client.reset_game(self.card_id, game_id, guid=None)
            guid = state.get("guid")
            ctx = SessionContext(game=GameProgress(game_id=game_id, play_num=1))
            ctx.set_game_identity(game_id=game_id, guid=guid)
            ctx.set_available_actions(
                state.get(
                    "available_actions",
                    list(HUMAN_ACTIONS.keys()),
                )
            )
            ctx.update(
                frame_grids=state.get("frame", []),
                current_score=state.get("score", 0),
                current_state=state.get("state", "IN_PROGRESS"),
                guid=guid,
            )
            game_contexts[game_id] = ctx
            game_states[game_id] = state

        for _ in range(max_rounds):
            plan = self.coordinate_actions(coordinator_context, game_contexts)
            actions = plan.get("actions", []) if isinstance(plan, dict) else []

            if not actions:
                break

            for item in actions:
                game_id = item.get("game_id")
                if not game_id or game_id not in game_contexts:
                    continue

                ctx = game_contexts[game_id]
                action_name = item.get("action")
                if not action_name or not self._validate_action(ctx, str(action_name)):
                    continue

                action_data: Optional[Dict[str, Any]] = None
                if isinstance(item.get("data"), dict):
                    action_data = dict(item.get("data") or {})
                elif "x" in item or "y" in item:
                    action_data = {"x": item.get("x", 0), "y": item.get("y", 0)}

                state = self._execute_game_action(
                    str(action_name),
                    action_data,
                    game_id,
                    ctx.game.guid,
                    reasoning={"swarm_plan": item},
                )
                game_states[game_id] = state

                ctx.update(
                    frame_grids=state.get("frame", []),
                    current_score=state.get("score", 0),
                    current_state=state.get("state", "IN_PROGRESS"),
                    guid=state.get("guid", ctx.game.guid),
                )
                ctx.set_game_identity(guid=state.get("guid", ctx.game.guid))
                ctx.set_counters(
                    action_counter=ctx.game.action_counter + 1,
                    play_action_counter=ctx.game.play_action_counter + 1,
                )

            if all(
                state.get("state") in ("WIN", "GAME_OVER") for state in game_states.values()
            ):
                break

        results: Dict[str, Dict[str, Any]] = {}
        for game_id, state in game_states.items():
            results[game_id] = {
                "state": state.get("state", "IN_PROGRESS"),
                "score": state.get("score", 0),
                "actions_taken": game_contexts[game_id].game.action_counter,
            }

        return results


__all__ = ["SwarmAgent"]

