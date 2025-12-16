"""
Helper module for generating multirun reports.
"""
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from arcagi3.game_client import GameClient


logger = logging.getLogger(__name__)


def generate_multirun_report(manifest: Dict[str, Any], manifest_path: Path) -> None:
    """
    Generate a markdown report summarizing the multirun results.
    Organized by game -> model -> other settings.
    """
    jobs = manifest.get("jobs", [])
    if not jobs:
        logger.warning("No jobs found in manifest, skipping report generation")
        return
    
    # Get base URL from GameClient
    base_url = "https://three.arcprize.org"
    game_info_map: Dict[str, Dict[str, Any]] = {}
    
    try:
        client = GameClient()
        base_url = client.ROOT_URL
        
        # Fetch game info to get titles/descriptions
        try:
            games_list = client.list_games()
            for game in games_list or []:
                game_id = game.get("game_id")
                if game_id:
                    game_info_map[game_id] = game
        except Exception as e:
            logger.warning(f"Failed to fetch game info: {e}")
    except Exception as e:
        logger.warning(f"Failed to initialize GameClient: {e}, using default base URL")
    
    # Organize jobs by game -> model
    games_dict: Dict[str, Dict[str, list[Dict[str, Any]]]] = {}
    
    for job in jobs:
        game_id = job.get("game_id", "unknown")
        config = job.get("config", "unknown")
        
        if game_id not in games_dict:
            games_dict[game_id] = {}
        if config not in games_dict[game_id]:
            games_dict[game_id][config] = []
        
        games_dict[game_id][config].append(job)
    
    # Generate markdown
    lines = []
    
    # Header
    run_name = manifest.get("run_name", "unknown")
    lines.append(f"# Multirun Report: {run_name}\n")
    lines.append(f"\n**Generated:** {datetime.now(timezone.utc).isoformat()}\n")
    lines.append("\n---\n\n")
    
    # Sweep configuration summary
    sweep = manifest.get("sweep", {})
    lines.append("## Sweep Configuration\n\n")
    
    # Show what was varied
    varied_params = []
    if len(sweep.get("configs", [])) > 1:
        varied_params.append("Models")
    if len(sweep.get("games", [])) > 1:
        varied_params.append("Games")
    if len(sweep.get("max_actions", [])) > 1:
        varied_params.append("Max Actions")
    if len(sweep.get("num_plays", [])) > 1:
        varied_params.append("Number of Plays")
    if len(sweep.get("max_episode_actions", [])) > 1:
        varied_params.append("Max Episode Actions")
    if len(sweep.get("memory_limits", [])) > 1:
        varied_params.append("Memory Limits")
    if len(sweep.get("checkpoint_frequencies", [])) > 1:
        varied_params.append("Checkpoint Frequencies")
    if len(sweep.get("use_vision", [])) > 1:
        varied_params.append("Vision Mode")
    if len(sweep.get("show_helper_images", [])) > 1:
        varied_params.append("Helper Images")
    
    if varied_params:
        lines.append(f"**Varied Parameters:** {', '.join(varied_params)}\n\n")
    
    # Fixed parameters table
    lines.append("### Fixed Parameters\n\n")
    lines.append("| Parameter | Value |\n")
    lines.append("|-----------|-------|\n")
    
    if len(sweep.get("max_actions", [])) == 1:
        lines.append(f"| Max Actions | {sweep.get('max_actions', [None])[0]} |\n")
    if len(sweep.get("num_plays", [])) == 1:
        lines.append(f"| Number of Plays | {sweep.get('num_plays', [None])[0]} |\n")
    if len(sweep.get("max_episode_actions", [])) == 1:
        max_ep_actions = sweep.get("max_episode_actions", [None])[0]
        lines.append(f"| Max Episode Actions | {max_ep_actions if max_ep_actions != 0 else 'Unlimited'} |\n")
    if len(sweep.get("memory_limits", [])) == 1:
        mem_limit = sweep.get("memory_limits", [None])[0]
        lines.append(f"| Memory Limit | {mem_limit if mem_limit is not None else 'Unlimited'} |\n")
    if len(sweep.get("checkpoint_frequencies", [])) == 1:
        lines.append(f"| Checkpoint Frequency | {sweep.get('checkpoint_frequencies', [None])[0]} |\n")
    if len(sweep.get("use_vision", [])) == 1:
        lines.append(f"| Vision Mode | {sweep.get('use_vision', [None])[0]} |\n")
    if len(sweep.get("show_helper_images", [])) == 1:
        lines.append(f"| Helper Images | {sweep.get('show_helper_images', [None])[0]} |\n")
    
    lines.append("\n---\n\n")
    
    # Games list with links
    lines.append("## Games Played\n\n")
    for game_id in sorted(games_dict.keys()):
        game_info = game_info_map.get(game_id, {})
        game_title = game_info.get("title", game_id)
        game_url = f"{base_url}/games/{game_id}"
        lines.append(f"* {game_title} - [{game_id.upper()}]({game_url})\n")
    lines.append("\n---\n\n")
    
    # Detailed breakdown by game
    for game_id in sorted(games_dict.keys()):
        game_info = game_info_map.get(game_id, {})
        game_title = game_info.get("title", game_id)
        game_display = game_id.upper()
        
        lines.append(f"### Game: {game_display}\n\n")
        if game_title != game_id:
            lines.append(f"*{game_title}*\n\n")
        
        # For each model
        model_num = 1
        for config in sorted(games_dict[game_id].keys()):
            jobs_for_model = games_dict[game_id][config]
            
            # If multiple jobs for same game+model (e.g., from sweep_repeats), show all
            if len(jobs_for_model) > 1:
                lines.append(f"{model_num}. **{config}** ({len(jobs_for_model)} runs)\n\n")
            else:
                lines.append(f"{model_num}. **{config}**\n\n")
            
            # Show each job
            for job_idx, job in enumerate(jobs_for_model):
                if len(jobs_for_model) > 1:
                    lines.append(f"   **Run {job_idx + 1}:**\n\n")
                
                checkpoint_id = job.get("checkpoint_id")
                scorecard_url = job.get("scorecard_url")
                final_score = job.get("final_score")
                final_state = job.get("final_state")
                status = job.get("status", "unknown")
                
                # Scorecard link
                if scorecard_url:
                    lines.append(f"   [Scorecard]({scorecard_url})\n\n")
                elif checkpoint_id:
                    # Construct scorecard URL from checkpoint ID
                    scorecard_url = f"{base_url}/scorecards/{checkpoint_id}"
                    lines.append(f"   [Scorecard]({scorecard_url})\n\n")
                
                # Checkpoint ID
                if checkpoint_id:
                    lines.append(f"   - Checkpoint ID: `{checkpoint_id}`\n")
                
                # Final score and state
                if final_score is not None:
                    lines.append(f"   - Final Score: {final_score}\n")
                if final_state:
                    lines.append(f"   - State: {final_state}\n")
                
                # Status
                if status != "completed":
                    lines.append(f"   - Status: {status}\n")
                
                # Other settings if they vary
                other_settings = []
                if len(sweep.get("max_actions", [])) > 1:
                    other_settings.append(f"Max Actions: {job.get('max_actions')}")
                if len(sweep.get("num_plays", [])) > 1:
                    other_settings.append(f"Num Plays: {job.get('num_plays')}")
                if len(sweep.get("max_episode_actions", [])) > 1:
                    max_ep_actions = job.get("max_episode_actions", 0)
                    other_settings.append(f"Max Episode Actions: {max_ep_actions if max_ep_actions != 0 else 'Unlimited'}")
                if len(sweep.get("memory_limits", [])) > 1:
                    mem_limit = job.get("memory_limit")
                    other_settings.append(f"Memory Limit: {mem_limit if mem_limit is not None else 'Unlimited'}")
                if len(sweep.get("checkpoint_frequencies", [])) > 1:
                    other_settings.append(f"Checkpoint Freq: {job.get('checkpoint_frequency')}")
                if len(sweep.get("use_vision", [])) > 1:
                    other_settings.append(f"Vision: {job.get('use_vision')}")
                if len(sweep.get("show_helper_images", [])) > 1:
                    other_settings.append(f"Helper Images: {job.get('show_helper_image')}")
                
                if other_settings:
                    lines.append(f"   - {' | '.join(other_settings)}\n")
                
                if len(jobs_for_model) > 1 and job_idx < len(jobs_for_model) - 1:
                    lines.append("\n")
            
            lines.append("\n")
            model_num += 1
        
        lines.append("\n")
    
    # Write to file
    report_path = manifest_path.with_suffix(".md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("".join(lines))
    
    logger.info(f"Multirun report written to: {report_path}")

