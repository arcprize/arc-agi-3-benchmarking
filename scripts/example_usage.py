"""
Example usage of the ARC-AGI-3 benchmarking framework.

See README.md for complete documentation and usage examples.
"""
import os
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import the main components
from src.arcagi3.agent import MultimodalAgent
from src.arcagi3.game_client import GameClient
from src.arcagi3.utils import read_models_config, generate_scorecard_tags


def example_single_game():
    """Example: Run a single game programmatically"""
    logger.info("Example 1: Running a single game")
    logger.info("=" * 60)
    
    # Initialize game client
    game_client = GameClient()
    
    # Get list of available games
    games = game_client.list_games()
    logger.info(f"Available games: {len(games)}")
    for game in games[:3]:  # Show first 3
        logger.info(f"  - {game['game_id']}: {game['title']}")
    
    if not games:
        logger.warning("No games available. Check your ARC_API_KEY.")
        return
    
    # Select first game
    game_id = games[0]['game_id']
    logger.info(f"\nPlaying game: {game_id}")
    
    # Configure model
    config = "gpt-4o-mini-2024-07-18"  # Use a cheaper model for testing
    logger.info(f"Using config: {config}")
    
    # Open scorecard (tags are optional, but recommended for tracking)
    # When using ARC3Tester, tags are automatically generated from model config
    scorecard_response = game_client.open_scorecard([game_id])
    card_id = scorecard_response.get("card_id")
    logger.info(f"Scorecard created with card_id: {card_id}")
    
    try:
        # Create agent
        agent = MultimodalAgent(
            config=config,
            game_client=game_client,
            card_id=card_id,
            max_actions=10,  # Limit for example
            retry_attempts=2,
            max_episode_actions=0,
        )
        
        # Play game
        result = agent.play_game(game_id)
        
        # Print results
        logger.info(f"\n{'=' * 60}")
        logger.info("Game Results:")
        logger.info(f"{'=' * 60}")
        logger.info(f"Final State: {result.final_state}")
        logger.info(f"Final Score: {result.final_score}")
        logger.info(f"Actions Taken: {result.actions_taken}")
        logger.info(f"Duration: {result.duration_seconds:.2f}s")
        logger.info(f"Total Cost: ${result.total_cost.total_cost:.4f}")
        logger.info(f"Total Tokens: {result.usage.total_tokens}")
        logger.info(f"\nView your scorecard online: {result.scorecard_url}")
        logger.info(f"{'=' * 60}")
        
    finally:
        # Clean up
        game_client.close_scorecard(card_id)
        game_client.close()


def example_list_games():
    """Example: List all available games"""
    logger.info("\nExample 2: Listing all games")
    logger.info("=" * 60)
    
    game_client = GameClient()
    games = game_client.list_games()
    
    logger.info(f"Total games available: {len(games)}\n")
    for game in games:
        logger.info(f"  {game['game_id']:<30} {game['title']}")
    
    game_client.close()


def example_model_configs():
    """Example: List available model configurations"""
    logger.info("\nExample 3: Available model configurations")
    logger.info("=" * 60)
    
    # Read models.yml to show available configs
    import yaml
    models_file = "src/arcagi3/models.yml"
    
    with open(models_file, 'r') as f:
        config_data = yaml.safe_load(f)
    
    logger.info(f"Total model configs: {len(config_data['models'])}\n")
    logger.info("Sample configs:")
    for model in config_data['models'][:10]:  # Show first 10
        logger.info(f"  {model['name']:<40} (Provider: {model['provider']})")


def example_custom_tags():
    """Example: Using custom tags with scorecards"""
    logger.info("\nExample 4: Custom scorecard tags")
    logger.info("=" * 60)
    
    config_name = "gpt-4o-mini-2024-07-18"
    model_config = read_models_config(config_name)
    
    # Generate automatic tags from model config
    auto_tags = generate_scorecard_tags(model_config)
    
    logger.info(f"\nModel config: {config_name}")
    logger.info("\nAutomatically generated tags:")
    for tag in auto_tags[:5]:  # Show first 5
        logger.info(f"  - {tag}")
    if len(auto_tags) > 5:
        logger.info(f"  ... and {len(auto_tags) - 5} more")
    
    # You can also add custom tags for your experiments
    custom_tags = auto_tags + ["experiment:baseline", "version:1.0"]
    
    logger.info(f"\nWith custom experiment tags:")
    for tag in custom_tags[-3:]:  # Show last 3
        logger.info(f"  - {tag}")
    
    logger.info("\nNote: When using ARC3Tester or main.py, tags are automatically added.")
    logger.info("For manual control, use game_client.open_scorecard(game_ids, tags=custom_tags)")


if __name__ == "__main__":
    logger.info("ARC-AGI-3 Benchmarking Framework - Examples")
    logger.info("=" * 60)
    logger.info("")
    
    # Check if API key is set
    if not os.getenv("ARC_API_KEY"):
        logger.error("ERROR: ARC_API_KEY not found in environment.")
        logger.error("Please set it in your .env file or environment.")
        exit(1)
    
    # Run examples
    try:
        example_model_configs()
        example_list_games()
        example_custom_tags()
        
        # Uncomment to actually run a game (will cost tokens)
        #example_single_game()
        
    except Exception as e:
        logger.error(f"\nError running examples: {e}", exc_info=True)



