from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


@dataclass(frozen=True)
class BreakpointFieldSpec:
    """
    Field spec describes a payload field that can be edited in the UI.

    `path` is a dotted path into the payload (e.g. "memory.text").
    """

    key: str
    label: str
    path: str
    editor: str = "json"
    description: Optional[str] = None
    read_only: bool = False

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "key": self.key,
            "label": self.label,
            "path": self.path,
            "editor": self.editor,
            "read_only": self.read_only,
        }
        if self.description:
            payload["description"] = self.description
        return payload


@dataclass(frozen=True)
class BreakpointPointSpec:
    """
    A single breakpoint point within a section.
    """

    point_id: str
    label: str
    phase: str = "post"
    fields: List[BreakpointFieldSpec] = field(default_factory=list)
    description: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "point_id": self.point_id,
            "label": self.label,
            "phase": self.phase,
            "fields": [f.to_dict() for f in self.fields],
        }
        if self.description:
            payload["description"] = self.description
        return payload


@dataclass(frozen=True)
class BreakpointSectionSpec:
    section_id: str
    label: str
    points: List[BreakpointPointSpec] = field(default_factory=list)
    description: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "section_id": self.section_id,
            "label": self.label,
            "points": [p.to_dict() for p in self.points],
        }
        if self.description:
            payload["description"] = self.description
        return payload


@dataclass(frozen=True)
class BreakpointSpec:
    """
    Root breakpoint spec; contains ordered sections.
    """

    sections: List[BreakpointSectionSpec] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"sections": [s.to_dict() for s in self.sections]}

    def point_ids(self) -> List[str]:
        ids: List[str] = []
        for section in self.sections:
            for point in section.points:
                ids.append(point.point_id)
        return ids


def _index_sections(sections: Iterable[BreakpointSectionSpec]) -> Dict[str, BreakpointSectionSpec]:
    return {section.section_id: section for section in sections}


def _index_points(points: Iterable[BreakpointPointSpec]) -> Dict[str, BreakpointPointSpec]:
    return {point.point_id: point for point in points}


def merge_breakpoint_specs(
    base: Optional[BreakpointSpec], override: Optional[BreakpointSpec]
) -> BreakpointSpec:
    """
    Merge two specs. Sections and points are merged by ID; override wins.
    """

    if not base and not override:
        return BreakpointSpec()
    if not base:
        return override or BreakpointSpec()
    if not override:
        return base

    base_sections = _index_sections(base.sections)
    override_sections = _index_sections(override.sections)
    merged_sections: List[BreakpointSectionSpec] = []

    section_ids = list(base_sections.keys())
    for section_id in override_sections.keys():
        if section_id not in base_sections:
            section_ids.append(section_id)

    for section_id in section_ids:
        base_section = base_sections.get(section_id)
        override_section = override_sections.get(section_id)
        if base_section and not override_section:
            merged_sections.append(base_section)
            continue
        if override_section and not base_section:
            merged_sections.append(override_section)
            continue

        assert base_section and override_section
        base_points = _index_points(base_section.points)
        override_points = _index_points(override_section.points)
        point_ids = list(base_points.keys())
        for point_id in override_points.keys():
            if point_id not in base_points:
                point_ids.append(point_id)
        merged_points: List[BreakpointPointSpec] = []
        for point_id in point_ids:
            base_point = base_points.get(point_id)
            override_point = override_points.get(point_id)
            if base_point and not override_point:
                merged_points.append(base_point)
                continue
            if override_point and not base_point:
                merged_points.append(override_point)
                continue
            assert base_point and override_point
            merged_points.append(
                BreakpointPointSpec(
                    point_id=override_point.point_id,
                    label=override_point.label or base_point.label,
                    phase=override_point.phase or base_point.phase,
                    fields=override_point.fields or base_point.fields,
                    description=override_point.description or base_point.description,
                )
            )

        merged_sections.append(
            BreakpointSectionSpec(
                section_id=override_section.section_id,
                label=override_section.label or base_section.label,
                points=merged_points,
                description=override_section.description or base_section.description,
            )
        )

    return BreakpointSpec(sections=merged_sections)


def load_breakpoint_spec(path: Optional[str]) -> Optional[BreakpointSpec]:
    if not path:
        return None
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    sections: List[BreakpointSectionSpec] = []
    for section_payload in data.get("sections", []):
        points: List[BreakpointPointSpec] = []
        for point_payload in section_payload.get("points", []):
            fields = [
                BreakpointFieldSpec(
                    key=str(field_payload.get("key", "")),
                    label=str(field_payload.get("label", "")),
                    path=str(field_payload.get("path", "")),
                    editor=str(field_payload.get("editor", "json")),
                    description=field_payload.get("description"),
                    read_only=bool(field_payload.get("read_only", False)),
                )
                for field_payload in point_payload.get("fields", [])
            ]
            points.append(
                BreakpointPointSpec(
                    point_id=str(point_payload.get("point_id", "")),
                    label=str(point_payload.get("label", "")),
                    phase=str(point_payload.get("phase", "post")),
                    fields=fields,
                    description=point_payload.get("description"),
                )
            )
        sections.append(
            BreakpointSectionSpec(
                section_id=str(section_payload.get("section_id", "")),
                label=str(section_payload.get("label", "")),
                points=points,
                description=section_payload.get("description"),
            )
        )
    return BreakpointSpec(sections=sections)
