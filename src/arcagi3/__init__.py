"""ARC-AGI-3 Benchmarking Framework"""

from .agent import MultimodalAgent
from .game_client import GameClient
from .schemas import (  # ARC-AGI-3 Game Schemas; Provider Schemas; Adapter compatibility schemas
    ARCPair, ARCTaskOutput, Attempt, Cost, GameAction, GameResult, GameState,
    ModelConfig, Usage)

__version__ = "0.1.0"

__all__ = [
    # Game components
    "MultimodalAgent",
    "GameClient",
    # Schemas
    "GameAction",
    "GameState", 
    "GameResult",
    "Cost",
    "Usage",
    "ModelConfig",
    "Attempt",
    "ARCTaskOutput",
    "ARCPair",
]

