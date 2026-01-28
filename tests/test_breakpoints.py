from arcagi3.breakpoints.manager import BreakpointHook, apply_breakpoint_overrides
from arcagi3.breakpoints.spec import (
    BreakpointFieldSpec,
    BreakpointPointSpec,
    BreakpointSectionSpec,
    BreakpointSpec,
    merge_breakpoint_specs,
)


def test_merge_breakpoint_specs():
    base = BreakpointSpec(
        sections=[
            BreakpointSectionSpec(
                section_id="analyze",
                label="Analyze",
                points=[
                    BreakpointPointSpec(
                        point_id="analyze.post",
                        label="Analyze (post)",
                        fields=[
                            BreakpointFieldSpec(key="analysis", label="Analysis", path="analysis")
                        ],
                    )
                ],
            )
        ]
    )
    override = BreakpointSpec(
        sections=[
            BreakpointSectionSpec(
                section_id="analyze",
                label="Analyze Override",
                points=[
                    BreakpointPointSpec(
                        point_id="analyze.post",
                        label="Analyze (override)",
                        fields=[
                            BreakpointFieldSpec(key="memory", label="Memory", path="memory_prompt")
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
                    )
                ],
            ),
        ]
    )

    merged = merge_breakpoint_specs(base, override)
    assert len(merged.sections) == 2
    analyze = next(section for section in merged.sections if section.section_id == "analyze")
    assert analyze.label == "Analyze Override"
    assert analyze.points[0].label == "Analyze (override)"
    assert analyze.points[0].fields[0].key == "memory"


def test_apply_breakpoint_overrides_with_hook():
    payload = {"value": 1}
    overrides = {"value": 2}
    context = {"touched": False}

    def apply_override(_payload, _overrides, ctx):
        ctx["touched"] = True
        return {"value": _overrides["value"] + 1}

    hook = BreakpointHook(point_id="test.point", apply_overrides=apply_override)
    updated = apply_breakpoint_overrides(payload, overrides, hook, context)
    assert updated["value"] == 3
    assert context["touched"] is True
