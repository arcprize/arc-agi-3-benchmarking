from __future__ import annotations

from typing import Any, Dict

from arcagi3.breakpoints.manager import BreakpointHook
from arcagi3.breakpoints.spec import (
    BreakpointFieldSpec,
    BreakpointPointSpec,
    BreakpointSectionSpec,
    BreakpointSpec,
)
from arcagi3.utils.context import SessionContext


def get_adcr_breakpoint_spec() -> BreakpointSpec:
    """Get the breakpoint specification for the ADCR agent."""
    return BreakpointSpec(
        sections=[
            BreakpointSectionSpec(
                section_id="analyze",
                label="Analyze",
                points=[
                    BreakpointPointSpec(
                        point_id="analyze.post",
                        label="Analyze (post)",
                        phase="post",
                        fields=[
                            BreakpointFieldSpec(
                                key="analysis",
                                label="Analysis",
                                path="analysis",
                                editor="textarea",
                            ),
                            BreakpointFieldSpec(
                                key="memory_prompt",
                                label="Memory prompt",
                                path="memory_prompt",
                                editor="textarea",
                            ),
                            BreakpointFieldSpec(
                                key="memory_word_limit",
                                label="Memory word limit",
                                path="memory_word_limit",
                                editor="number",
                            ),
                        ],
                    )
                ],
            ),
            BreakpointSectionSpec(
                section_id="decide",
                label="Decide",
                points=[
                    BreakpointPointSpec(
                        point_id="decide.post",
                        label="Decide (post)",
                        phase="post",
                        fields=[
                            BreakpointFieldSpec(
                                key="result",
                                label="Human action JSON",
                                path="result",
                                editor="json",
                            ),
                            BreakpointFieldSpec(
                                key="memory_prompt",
                                label="Memory prompt",
                                path="memory_prompt",
                                editor="textarea",
                            ),
                            BreakpointFieldSpec(
                                key="memory_word_limit",
                                label="Memory word limit",
                                path="memory_word_limit",
                                editor="number",
                            ),
                        ],
                    )
                ],
            ),
            BreakpointSectionSpec(
                section_id="convert",
                label="Convert",
                points=[
                    BreakpointPointSpec(
                        point_id="convert.post",
                        label="Convert (post)",
                        phase="post",
                        fields=[
                            BreakpointFieldSpec(
                                key="result",
                                label="Game action JSON",
                                path="result",
                                editor="json",
                            ),
                            BreakpointFieldSpec(
                                key="memory_prompt",
                                label="Memory prompt",
                                path="memory_prompt",
                                editor="textarea",
                            ),
                            BreakpointFieldSpec(
                                key="memory_word_limit",
                                label="Memory word limit",
                                path="memory_word_limit",
                                editor="number",
                            ),
                        ],
                    )
                ],
            ),
            BreakpointSectionSpec(
                section_id="execute",
                label="Execute Action",
                points=[
                    BreakpointPointSpec(
                        point_id="execute_action.pre",
                        label="Execute action (pre)",
                        phase="pre",
                        fields=[
                            BreakpointFieldSpec(
                                key="action",
                                label="Action",
                                path="action",
                                editor="text",
                            ),
                            BreakpointFieldSpec(
                                key="action_data",
                                label="Action data",
                                path="action_data",
                                editor="json",
                            ),
                            BreakpointFieldSpec(
                                key="reasoning",
                                label="Reasoning",
                                path="reasoning",
                                editor="json",
                            ),
                        ],
                    ),
                    BreakpointPointSpec(
                        point_id="execute_action.post",
                        label="Execute action (post)",
                        phase="post",
                        fields=[
                            BreakpointFieldSpec(
                                key="result",
                                label="Action result",
                                path="result",
                                editor="json",
                            )
                        ],
                    ),
                ],
            ),
        ]
    )


def get_adcr_breakpoint_hooks(agent) -> Dict[str, BreakpointHook]:
    """Get the breakpoint hooks for the ADCR agent.

    Args:
        agent: The ADCRAgent instance (needed to access memory_word_limit)
    """

    def _apply_memory(
        payload: Dict[str, Any], overrides: Dict[str, Any], context: SessionContext
    ) -> Dict[str, Any]:
        if not isinstance(context, SessionContext):
            return payload
        if "memory_prompt" in overrides and isinstance(overrides.get("memory_prompt"), str):
            context.datastore["memory_prompt"] = overrides["memory_prompt"]
        if "memory_word_limit" in overrides:
            try:
                agent.memory_word_limit = int(overrides["memory_word_limit"])
            except Exception:
                pass
        return overrides

    return {
        "analyze.post": BreakpointHook(point_id="analyze.post", apply_overrides=_apply_memory),
        "decide.post": BreakpointHook(point_id="decide.post", apply_overrides=_apply_memory),
        "convert.post": BreakpointHook(point_id="convert.post", apply_overrides=_apply_memory),
    }
