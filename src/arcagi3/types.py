"""
Shared type aliases used across ARC-AGI-3 harness modules.

Keep these in one place so we don't redefine them in multiple files and so downstream
agents/testers can import a single canonical name for typing.
"""

from typing import List

from PIL import Image

# A single 2D grid (e.g. 64x64) of integer color ids.
FrameGrid = List[List[int]]

# A sequence of grids representing the frames returned by the API.
FrameGridSequence = List[FrameGrid]

# A sequence of PIL images corresponding to frames.
FrameImageSequence = List[Image.Image]
