"""
Enhanced scorecard output formatting for game results.

Provides rich, visually appealing output for game completion summaries.
"""

from datetime import datetime
from typing import Optional

from arcagi3.schemas import GameResult


def format_timestamp(timestamp: Optional[datetime]) -> str:
    """Format timestamp for display."""
    if timestamp is None:
        return "N/A"
    if timestamp.tzinfo:
        return timestamp.strftime("%Y-%m-%d %H:%M:%S %Z")
    return timestamp.strftime("%Y-%m-%d %H:%M:%S UTC")


def get_status_icon(final_state: str) -> str:
    """Get status icon based on final state."""
    if final_state == "WIN":
        return "✅"
    elif final_state == "NOT_FINISHED":
        return "❌"
    elif final_state == "GAME_OVER":
        return "💀"
    else:
        return "⚠️"


def format_percentage(value: float, total: float) -> str:
    """Format percentage with 1 decimal place."""
    if total == 0:
        return "0.0%"
    return f"{(value / total * 100):.1f}%"


def format_action_summary(actions: list, max_show: int = 3) -> list[str]:
    """Format action summary showing first and last actions."""
    if not actions:
        return ["No actions taken"]

    if len(actions) <= max_show * 2:
        # Show all actions if we have few enough
        return [
            f"Action #{a.action_num}: {a.action}  →  Score: {a.result_score}  ({a.result_state})"
            for a in actions
        ]

    # Show first few, ellipsis, last few
    lines = []
    for a in actions[:max_show]:
        lines.append(
            f"Action #{a.action_num}: {a.action}  →  Score: {a.result_score}  ({a.result_state})"
        )
    lines.append("...")
    for a in actions[-max_show:]:
        lines.append(
            f"Action #{a.action_num}: {a.action}  →  Score: {a.result_score}  ({a.result_state})"
        )
    return lines


def format_box_line(content: str, width: int = 74) -> str:
    """Format a line inside a box with proper padding."""
    # Box width is 78, content area is 74 (78 - 4 for borders)
    if len(content) > width:
        content = content[: width - 3] + "..."
    return f"│ {content:<{width}} │"


def print_result(result: GameResult) -> None:
    """
    Print enhanced game result summary with rich formatting.

    Args:
        result: GameResult object containing all game execution data
    """
    # Calculate metrics
    actions_taken = result.actions_taken
    duration = result.duration_seconds
    total_cost = result.total_cost.total_cost
    total_tokens = result.usage.total_tokens

    # Cost breakdown
    prompt_cost = result.total_cost.prompt_cost
    completion_cost = result.total_cost.completion_cost
    reasoning_cost = result.total_cost.reasoning_cost or 0.0

    # Token breakdown
    prompt_tokens = result.usage.prompt_tokens
    completion_tokens = result.usage.completion_tokens

    # Performance metrics
    actions_per_sec = actions_taken / duration if duration > 0 else 0.0
    avg_time_per_action = duration / actions_taken if actions_taken > 0 else 0.0
    tokens_per_sec = total_tokens / duration if duration > 0 else 0.0
    cost_per_action = total_cost / actions_taken if actions_taken > 0 else 0.0
    tokens_per_action = total_tokens / actions_taken if actions_taken > 0 else 0.0

    # Status icon
    status_icon = get_status_icon(result.final_state)

    # Format timestamp
    timestamp_str = format_timestamp(result.timestamp)

    # Action summary
    action_lines = format_action_summary(result.actions)

    # Build output
    output_lines = []

    # Header
    output_lines.append("")
    output_lines.append(
        "╔══════════════════════════════════════════════════════════════════════════╗"
    )
    output_lines.append(
        "║                          🎮 GAME RESULT SUMMARY                           ║"
    )
    output_lines.append(
        "╚══════════════════════════════════════════════════════════════════════════╝"
    )
    output_lines.append("")

    # Game info
    output_lines.append(f"Game ID:     {result.game_id}")
    output_lines.append(f"Model Config: {result.config}")
    output_lines.append(f"Timestamp:   {timestamp_str}")
    output_lines.append("")

    # Outcome section
    output_lines.append(
        "┌──────────────────────────────────────────────────────────────────────────┐"
    )
    output_lines.append(
        "│ 📊 OUTCOME                                                                │"
    )
    output_lines.append(
        "├──────────────────────────────────────────────────────────────────────────┤"
    )
    output_lines.append(format_box_line(f"Status:     {status_icon} {result.final_state}"))
    output_lines.append(format_box_line(f"Score:      {result.final_score}"))
    output_lines.append(format_box_line(f"Actions:    {actions_taken}"))
    output_lines.append(format_box_line(f"Duration:   {duration:.2f}s"))
    output_lines.append(
        "└──────────────────────────────────────────────────────────────────────────┘"
    )
    output_lines.append("")

    # Cost breakdown
    output_lines.append(
        "┌──────────────────────────────────────────────────────────────────────────┐"
    )
    output_lines.append(
        "│ 💰 COST BREAKDOWN                                                         │"
    )
    output_lines.append(
        "├──────────────────────────────────────────────────────────────────────────┤"
    )
    if total_cost > 0:
        prompt_pct = format_percentage(prompt_cost, total_cost)
        completion_pct = format_percentage(completion_cost, total_cost)
        reasoning_pct = (
            format_percentage(reasoning_cost, total_cost) if reasoning_cost > 0 else "0.0%"
        )

        output_lines.append(format_box_line(f"Prompt:     ${prompt_cost:.4f}  ({prompt_pct})"))
        output_lines.append(
            format_box_line(f"Completion: ${completion_cost:.4f}  ({completion_pct})")
        )
        if reasoning_cost > 0:
            output_lines.append(
                format_box_line(f"Reasoning:  ${reasoning_cost:.4f}  ({reasoning_pct})")
            )
        else:
            output_lines.append(format_box_line("Reasoning:  $0.0000  (0.0%)"))
    else:
        output_lines.append(format_box_line("Prompt:     $0.0000  (0.0%)"))
        output_lines.append(format_box_line("Completion: $0.0000  (0.0%)"))
        output_lines.append(format_box_line("Reasoning:  $0.0000  (0.0%)"))
    output_lines.append(
        "│ ──────────────────────────────────────────────────────────────────────── │"
    )
    output_lines.append(format_box_line(f"Total:      ${total_cost:.4f}"))
    output_lines.append(format_box_line(f"Per Action: ${cost_per_action:.4f}"))
    output_lines.append(
        "└──────────────────────────────────────────────────────────────────────────┘"
    )
    output_lines.append("")

    # Token usage
    output_lines.append(
        "┌──────────────────────────────────────────────────────────────────────────┐"
    )
    output_lines.append(
        "│ 🔢 TOKEN USAGE                                                            │"
    )
    output_lines.append(
        "├──────────────────────────────────────────────────────────────────────────┤"
    )
    if total_tokens > 0:
        prompt_tokens_pct = format_percentage(prompt_tokens, total_tokens)
        completion_tokens_pct = format_percentage(completion_tokens, total_tokens)

        output_lines.append(
            format_box_line(f"Prompt:     {prompt_tokens:,} tokens  ({prompt_tokens_pct})")
        )
        output_lines.append(
            format_box_line(f"Completion: {completion_tokens:,} tokens  ({completion_tokens_pct})")
        )
    else:
        output_lines.append(format_box_line("Prompt:     0 tokens  (0.0%)"))
        output_lines.append(format_box_line("Completion: 0 tokens  (0.0%)"))
    output_lines.append(
        "│ ──────────────────────────────────────────────────────────────────────── │"
    )
    output_lines.append(format_box_line(f"Total:      {total_tokens:,} tokens"))
    output_lines.append(format_box_line(f"Per Action: {tokens_per_action:.0f} tokens"))
    output_lines.append(
        "└──────────────────────────────────────────────────────────────────────────┘"
    )
    output_lines.append("")

    # Performance metrics
    output_lines.append(
        "┌──────────────────────────────────────────────────────────────────────────┐"
    )
    output_lines.append(
        "│ ⚡ PERFORMANCE METRICS                                                    │"
    )
    output_lines.append(
        "├──────────────────────────────────────────────────────────────────────────┤"
    )
    output_lines.append(format_box_line(f"Actions/sec:    {actions_per_sec:.2f}"))
    output_lines.append(format_box_line(f"Avg Time/Action: {avg_time_per_action:.2f}s"))
    output_lines.append(format_box_line(f"Tokens/sec:     {tokens_per_sec:.1f}"))
    output_lines.append(
        "└──────────────────────────────────────────────────────────────────────────┘"
    )
    output_lines.append("")

    # Action summary
    output_lines.append(
        "┌──────────────────────────────────────────────────────────────────────────┐"
    )
    output_lines.append(
        "│ 🎯 ACTION SUMMARY                                                         │"
    )
    output_lines.append(
        "├──────────────────────────────────────────────────────────────────────────┤"
    )
    for line in action_lines:
        output_lines.append(format_box_line(line))
    output_lines.append(
        "└──────────────────────────────────────────────────────────────────────────┘"
    )
    output_lines.append("")

    # Scorecard link
    if result.scorecard_url:
        output_lines.append(
            "┌──────────────────────────────────────────────────────────────────────────┐"
        )
        output_lines.append(
            "│ 🔗 SCORECARD                                                              │"
        )
        output_lines.append(
            "├──────────────────────────────────────────────────────────────────────────┤"
        )
        output_lines.append(format_box_line("View detailed scorecard online:"))
        # Split long URLs across lines if needed
        url = result.scorecard_url
        if len(url) > 74:
            # Try to break at a reasonable point
            parts = url.split("/")
            if len(parts) > 3:
                base = "/".join(parts[:3])
                path = "/" + "/".join(parts[3:])
                output_lines.append(format_box_line(base))
                output_lines.append(format_box_line(path))
            else:
                output_lines.append(format_box_line(url))
        else:
            output_lines.append(format_box_line(url))
        if result.card_id:
            output_lines.append(format_box_line(""))
            output_lines.append(format_box_line(f"Card ID: {result.card_id}"))
        output_lines.append(
            "└──────────────────────────────────────────────────────────────────────────┘"
        )
        output_lines.append("")
        # Add plain URL line for easy clicking/copying
        output_lines.append(f"link: {result.scorecard_url}")
        output_lines.append("")

    # Footer
    output_lines.append(
        "╔══════════════════════════════════════════════════════════════════════════╗"
    )
    output_lines.append(
        "║                          End of Result Summary                            ║"
    )
    output_lines.append(
        "╚══════════════════════════════════════════════════════════════════════════╝"
    )
    output_lines.append("")

    # Print all lines directly (without logging prefix)
    for line in output_lines:
        print(line)
