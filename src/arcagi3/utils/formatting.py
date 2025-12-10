"""
Utility functions for formatting text and grids.
"""
import json
from typing import Dict, List


def get_human_inputs_text(available_actions: List[str], human_actions_map: Dict[str, str]) -> str:
    """
    Convert available actions to human-readable text.
    
    Args:
        available_actions: List of available action codes (e.g., ["ACTION1", "ACTION2"])
        human_actions_map: Dictionary mapping action codes to descriptions
        
    Returns:
        Formatted text listing available actions
    """
    text = "\n"
    for action in available_actions:
        if action in human_actions_map:
            text += f"{human_actions_map[action]}\n"
    return text


def grid_to_text_matrix(grid: List[List[int]]) -> str:
    """
    Convert a grid matrix to a readable text representation.
    
    Args:
        grid: 64x64 grid of integers (0-15) representing colors
        
    Returns:
        Formatted text representation of the grid (JSON format)
    """
    # Format as JSON for clarity and compactness
    return json.dumps(grid, separators=(',', ','))
