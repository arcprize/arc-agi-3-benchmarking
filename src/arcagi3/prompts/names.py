from enum import Enum


class PromptName(str, Enum):
    """Names for all supported agent prompts."""

    SYSTEM = "system"
    ACTION_INSTRUCT = "action_instruct"
    ANALYZE_INSTRUCT = "analyze_instruct"
    FIND_ACTION_INSTRUCT = "find_action_instruct"
    COMPRESS_MEMORY = "compress_memory"


__all__ = ["PromptName"]


