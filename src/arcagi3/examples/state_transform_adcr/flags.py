from __future__ import annotations

import argparse

from arcagi3.examples.state_transform_adcr import (
    StateTransformADCRAgent,
    StateTransformPayload,
)
from arcagi3.utils.context import SessionContext
from arcagi3.utils.formatting import grid_to_text_matrix
from arcagi3.utils.image import grid_to_image


def add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--state-transform",
        default="basic",
        choices=("basic", "text-only"),
        help="State transform preset for StateTransformADCRAgent",
    )


def _basic_state_transform(context: SessionContext) -> StateTransformPayload:
    text_parts = []
    for idx, grid in enumerate(context.frames.frame_grids):
        text_parts.append(f"Frame {idx}:\n{grid_to_text_matrix(grid)}")
    text = "\n\n".join(text_parts) if text_parts else "No frames available."
    images = [grid_to_image(grid) for grid in context.frames.frame_grids] or None
    return StateTransformPayload(text=text, images=images)


def _text_only_state_transform(context: SessionContext) -> StateTransformPayload:
    text_parts = []
    for idx, grid in enumerate(context.frames.frame_grids):
        text_parts.append(f"Frame {idx}:\n{grid_to_text_matrix(grid)}")
    text = "\n\n".join(text_parts) if text_parts else "No frames available."
    return StateTransformPayload(text=text, images=None)


def get_kwargs(args):
    if args.state_transform == "text-only":
        transform = _text_only_state_transform
    else:
        transform = _basic_state_transform
    return {"state_transform": transform}


flags = {
    "name": "state-transform",
    "description": "ADCR with a state transform pre-processing step",
    "agent_class": StateTransformADCRAgent,
    "get_kwargs": get_kwargs,
    "add_args": add_args,
}

