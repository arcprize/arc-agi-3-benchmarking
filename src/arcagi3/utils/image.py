"""
Image processing utilities for ARC-AGI-3 games.

Handles conversion of grid data to images and visual diff generation.
"""
import base64
import io
from typing import List, Sequence, Tuple

import numpy as np
from PIL import Image

# 16-color palette (RGBA format)
_PALETTE: List[Tuple[int, int, int, int]] = [
    (0xFF, 0xFF, 0xFF, 0xFF),  # 0 White
    (0xCC, 0xCC, 0xCC, 0xFF),  # 1 Off-white
    (0x99, 0x99, 0x99, 0xFF),  # 2 Neutral light
    (0x66, 0x66, 0x66, 0xFF),  # 3 Neutral
    (0x33, 0x33, 0x33, 0xFF),  # 4 Off-black
    (0x00, 0x00, 0x00, 0xFF),  # 5 Black
    (0xE5, 0x3A, 0xA3, 0xFF),  # 6 Magenta
    (0xFF, 0x7B, 0xCC, 0xFF),  # 7 Magenta light
    (0xF9, 0x3C, 0x31, 0xFF),  # 8 Red
    (0x1E, 0x93, 0xFF, 0xFF),  # 9 Blue
    (0x88, 0xD8, 0xF1, 0xFF),  # 10 Blue light
    (0xFF, 0xDC, 0x00, 0xFF),  # 11 Yellow
    (0xFF, 0x85, 0x1B, 0xFF),  # 12 Orange
    (0x92, 0x12, 0x31, 0xFF),  # 13 Maroon
    (0x4F, 0xCC, 0x30, 0xFF),  # 14 Green
    (0xA3, 0x56, 0xD6, 0xFF),  # 15 Purple
]

_SCALE = 2  # Scale factor for upscaling (64px -> 128px)
_TARGET_SIZE = 64 * _SCALE


def _validate_grid(grid: Sequence[Sequence[int]]) -> None:
    """Validate that grid is 64x64 with values 0-15"""
    if len(grid) != 64 or any(len(row) != 64 for row in grid):
        raise ValueError("Grid must be 64×64.")
    if any(cell not in range(16) for row in grid for cell in row):
        raise ValueError("Grid values must be integers 0–15.")


def grid_to_image(grid: Sequence[Sequence[int]]) -> Image.Image:
    """
    Convert a 64×64 int grid to a 128×128 RGBA Pillow Image.
    
    Args:
        grid: 64x64 grid of integers (0-15) representing colors
        
    Returns:
        PIL Image (128x128 RGBA)
    """
    _validate_grid(grid)
    
    # Flatten grid into raw bytes (R, G, B, A per pixel)
    raw = bytearray()
    for row in grid:
        for idx in row:
            raw.extend(_PALETTE[idx])
    
    img = Image.frombytes("RGBA", (64, 64), bytes(raw))
    # Nearest-neighbor upscale keeps crisp pixel art
    img = img.resize((_TARGET_SIZE, _TARGET_SIZE), Image.NEAREST)
    return img


def image_to_base64(img: Image.Image) -> str:
    """
    Return a base-64 encoded PNG (no data-URL prefix).
    
    Args:
        img: PIL Image
        
    Returns:
        Base64 encoded string
    """
    buffer = io.BytesIO()
    img.save(buffer, format="PNG", optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def make_image_block(b64_string: str) -> dict:
    """
    Return the JSON block expected for an inline base-64 image.
    
    Args:
        b64_string: Base64 encoded image string
        
    Returns:
        Dict with image_url structure for API
    """
    return {
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{b64_string}"},
    }


def display_image_in_terminal(img: Image.Image, width: int = 32) -> None:
    """
    Display an image in the terminal using Unicode blocks and ANSI colors.

    Args:
        img: PIL Image to display
        width: Display width in characters (default: 32)
    """
    # Calculate height maintaining aspect ratio
    aspect_ratio = img.height / img.width
    height = int(width * aspect_ratio / 2)  # /2 because terminal chars are taller

    # Resize image to fit terminal
    img_resized = img.resize((width, height * 2), Image.NEAREST)
    pixels = np.array(img_resized.convert("RGB"))

    # Use half blocks: top half and bottom half
    for y in range(0, height * 2, 2):
        line = ""
        for x in range(width):
            # Get top and bottom pixel colors
            top_pixel = pixels[y, x] if y < pixels.shape[0] else [0, 0, 0]
            bottom_pixel = pixels[y + 1, x] if y + 1 < pixels.shape[0] else [0, 0, 0]

            # Convert RGB to ANSI color codes
            top_r, top_g, top_b = top_pixel
            bottom_r, bottom_g, bottom_b = bottom_pixel

            # Use lower half block (▄) with top color as background, bottom as foreground
            line += f"\033[38;2;{bottom_r};{bottom_g};{bottom_b}m\033[48;2;{top_r};{top_g};{top_b}m▄\033[0m"

        print(line)
    print("\033[0m")  # Reset colors


def image_diff(
    img_a: Image.Image,
    img_b: Image.Image,
    highlight_rgb: Tuple[int, int, int] = (255, 0, 0),  # red
) -> Image.Image:
    """
    Compare img_a vs img_b and create a visual diff.
    
    If images are identical, returns pure black image.
    If they differ, only changed pixels are highlighted on black background.
    
    Args:
        img_a: First image
        img_b: Second image
        highlight_rgb: RGB color for highlighting differences
        
    Returns:
        PIL Image showing differences
    """
    a = np.asarray(img_a.convert("RGB"))
    b = np.asarray(img_b.convert("RGB"))
    
    if a.shape != b.shape:
        raise ValueError(
            f"Images must have the same dimensions; got {a.shape} vs {b.shape}"
        )
    
    # Boolean mask: True where any channel differs
    diff_mask = np.any(a != b, axis=-1)
    
    # Fast equality check
    if not diff_mask.any():
        # Identical – return black image
        return Image.new("RGB", (a.shape[1], a.shape[0]), (0, 0, 0))
    
    # Start with black canvas, paint the differing pixels
    diff_img = np.zeros_like(a)
    diff_img[diff_mask] = highlight_rgb
    
    return Image.fromarray(diff_img)

