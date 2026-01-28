from __future__ import annotations

from typing import Iterable, List, Optional

from arcagi3.adapters.provider import ProviderAdapter
from arcagi3.prompts import PromptManager
from arcagi3.utils.context import SessionContext


def truncate_memory(
    memory_text: str,
    *,
    max_words: int,
    provider: ProviderAdapter,
    context: Optional[SessionContext] = None,
    rules: Optional[Iterable[str] | str] = None,
) -> str:
    """
    Truncate memory text using the model.

    Args:
        memory_text: The memory body to truncate.
        max_words: Required word limit.
        provider: Provider adapter used to call the model.
        context: Optional session context for usage tracking.
        rules: Optional rules/keywords to emphasize during compression.
    """
    rules_list: List[str] = []
    if isinstance(rules, str):
        rules_list = [line.strip() for line in rules.splitlines() if line.strip()]
    elif rules:
        rules_list = [str(rule).strip() for rule in rules if str(rule).strip()]

    rules_block = "\n".join(f"- {rule}" for rule in rules_list) if rules_list else "None"

    prompt_text = PromptManager().render(
        "truncate_memory",
        {
            "memory_limit": max_words,
            "memory_text": memory_text,
            "rules_block": rules_block,
        },
    )

    messages = [
        {"role": "system", "content": "You are a concise summarizer for game memory."},
        {"role": "user", "content": prompt_text},
    ]

    if context is not None:
        response = provider.call_with_tracking(context, messages, step_name="truncate_memory")
    else:
        response = provider.call_provider(messages)

    compressed = provider.extract_content(response).strip()
    return compressed or memory_text


__all__ = ["truncate_memory"]
