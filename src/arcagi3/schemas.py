import json
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, model_validator

# ============================================================================
# Static ARC Task Schemas (for adapter compatibility)
# ============================================================================

class ARCTaskOutput(BaseModel):
    """Output for static ARC tasks (not used in ARC-AGI-3 games)"""
    output: List[List[int]]


class ARCPair(BaseModel):
    """Input/output pair for static ARC tasks (not used in ARC-AGI-3 games)"""
    input: List[List[int]]
    output: Optional[List[List[int]]] = None


# ============================================================================
# ARC-AGI-3 Game Schemas
# ============================================================================

class GameAction(Enum):
    """Available actions in ARC-AGI-3 games"""
    RESET = "RESET"
    ACTION1 = "ACTION1"  # Move Up
    ACTION2 = "ACTION2"  # Move Down
    ACTION3 = "ACTION3"  # Move Left
    ACTION4 = "ACTION4"  # Move Right
    ACTION5 = "ACTION5"  # Perform Action
    ACTION6 = "ACTION6"  # Click object on screen (requires x, y)
    ACTION7 = "ACTION7"  # Undo


class GameState(Enum):
    """Possible game states"""
    NOT_PLAYED = "NOT_PLAYED"
    IN_PROGRESS = "IN_PROGRESS"
    WIN = "WIN"
    GAME_OVER = "GAME_OVER"


class ActionData(BaseModel):
    """Data associated with a game action"""
    x: Optional[int] = None
    y: Optional[int] = None
    reasoning: Optional[Dict[str, Any]] = None


class GameActionRecord(BaseModel):
    """Record of a single action taken during a game"""
    action_num: int
    action: str
    action_data: Optional[ActionData] = None
    reasoning: Optional[Dict[str, Any]] = None
    result_score: int
    result_state: str
    cost: Optional["Cost"] = None


class GameResult(BaseModel):
    """Complete result of playing a single game"""
    game_id: str
    config: str
    final_score: int
    final_state: str  # GameState as string
    actions_taken: int
    duration_seconds: float
    total_cost: "Cost"
    usage: "Usage"
    actions: List[GameActionRecord]
    final_memory: Optional[str] = None
    timestamp: datetime = None
    scorecard_url: Optional[str] = None
    card_id: Optional[str] = None
    
    model_config = {
        'json_encoders': {
            datetime: lambda v: v.isoformat() if v else None
        }
    }
    
    @model_validator(mode='before')
    @classmethod
    def set_timestamp(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        if isinstance(values, dict) and 'timestamp' not in values:
            values['timestamp'] = datetime.utcnow()
        return values


# ============================================================================
# Provider/Model Schemas (from old repo)
# ============================================================================

class APIType:
    """Enum for the different API types that can be used with the OpenAI API."""
    CHAT_COMPLETIONS = "chat_completions"
    RESPONSES = "responses"


class Message(BaseModel):
    role: str
    content: str


class Choice(BaseModel):
    index: int
    message: Message


class CompletionTokensDetails(BaseModel):
    reasoning_tokens: int = 0
    accepted_prediction_tokens: int = 0
    rejected_prediction_tokens: int = 0


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    completion_tokens_details: Optional[CompletionTokensDetails] = None


class Cost(BaseModel):
    prompt_cost: float
    completion_cost: float  # Cost of completion_tokens * output_cost_per_token
    reasoning_cost: Optional[float] = None
    total_cost: float  # Sum of prompt_cost, completion_cost, and reasoning_cost


class AttemptMetadata(BaseModel):
    model: str
    provider: str
    start_timestamp: datetime
    end_timestamp: datetime
    choices: List[Choice]
    reasoning_summary: Optional[str] = None
    kwargs: Dict[str, Any]
    usage: Usage
    cost: Cost
    task_id: Optional[str] = None
    pair_index: Optional[int] = 0
    test_id: Optional[str] = None
    
    model_config = {
        'json_encoders': {
            datetime: lambda v: v.isoformat()
        }
    }

    def __str__(self):
        return json.dumps(self.model_dump(), indent=2, default=str)
    
    __repr__ = __str__


class Attempt(BaseModel):
    answer: Union[str, Dict[str, Any]]
    metadata: AttemptMetadata
    correct: Optional[bool] = None

    @model_validator(mode="before")
    @classmethod
    def check_answer_present(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure the required 'answer' field exists."""
        if isinstance(values, dict) and "answer" not in values:
            raise KeyError("answer")
        return values


class ModelPricing(BaseModel):
    date: str
    input: float  # Cost per 1M tokens
    output: float  # Cost per 1M tokens


class ModelConfig(BaseModel):
    """
    A model configuration used to populate a model's kwargs and calculate pricing metadata.
    Points to models.yml
    """
    name: str  # Config name
    model_name: str  # The actual model name to use with the provider's API
    provider: str
    is_multimodal: bool = False
    pricing: ModelPricing
    api_type: Optional[str] = APIType.CHAT_COMPLETIONS
    kwargs: Dict[str, Any] = {}
    
    model_config = {
        'protected_namespaces': (),
        'extra': 'allow'
    }
    
    @model_validator(mode='before')
    @classmethod
    def extract_kwargs(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Extract all extra fields into kwargs"""
        if not isinstance(values, dict):
            return values
            
        kwargs = {}
        known_fields = {'name', 'provider', 'pricing', 'kwargs', 'model_name', 'api_type', 'is_multimodal'}
        
        for field_name, value in values.items():
            if field_name not in known_fields:
                kwargs[field_name] = value
                
        # Update the kwargs field with our extracted values
        if kwargs:
            values['kwargs'] = {**kwargs, **values.get('kwargs', {})}
            
            # Remove the extracted fields from the top level
            for field_name in kwargs:
                if field_name in values:
                    del values[field_name]
        
        # Ensure capability flags are not kept in kwargs accidentally
        if 'kwargs' in values and isinstance(values['kwargs'], dict):
            values['kwargs'].pop('is_multimodal', None)
                    
        return values


# ============================================================================
# Stream Response
# ============================================================================

@dataclass
class StreamResponse:
    """Wrapper for consumed stream responses"""
    content: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
