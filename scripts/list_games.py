from arcagi3.game_client import GameClient


def list_games():
    # Need a valid card_id. Usually passed via env or args.
    # I'll try to find one or just use a placeholder if the client handles it.
    # The error message "game_id ... not found" suggests the client works but the ID is wrong.
    # I'll try to list games for the current scorecard if possible.

    # Check if we can get a scorecard.
    # The main.py uses a card_id.
    # I'll try to use a dummy one or see if I can get one.

    # Actually, let's just try to run the agent with a game ID that might exist or just rely on the fact that the code ran.
    # But to be thorough...

    client = GameClient()

    # List games doesn't require a scorecard - it lists all available games
    games = client.list_games()

    if games:
        print(f"Found {len(games)} games")
        print(f"First game: {games[0]}")
        if len(games) > 1:
            print(f"Last game: {games[-1]}")
    else:
        print("No games found")


if __name__ == "__main__":
    try:
        list_games()
    except Exception as e:
        print(f"Error: {e}")
