"""ARC-AGI-3 Benchmarking Framework"""

from arcagi3.agent import MultimodalAgent
from arcagi3.game_client import GameClient
from arcagi3.schemas import (  # ARC-AGI-3 Game Schemas; Provider Schemas; Adapter compatibility schemas
    ARCPair,
    ARCTaskOutput,
    Attempt,
    Cost,
    GameAction,
    GameResult,
    GameState,
    ModelConfig,
    Usage,
)

__version__ = "0.9.0"

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
