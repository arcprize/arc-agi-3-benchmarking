from __future__ import annotations

from typing import Any

from arcagi3.adcr_agent import ADCRAgent


def get_kwargs(args: Any) -> dict[str, Any]:
    return {
        "use_vision": getattr(args, "use_vision", True),
        "show_images": getattr(args, "show_images", False),
        "memory_word_limit": getattr(args, "memory_limit", None),
    }


definition = {
    "name": "adcr",
    "description": "Analyze -> Decide -> Convert -> Review reference agent",
    "agent_class": ADCRAgent,
    "get_kwargs": get_kwargs,
}

agents = [definition]

__all__ = ["definition", "agents", "get_kwargs"]
