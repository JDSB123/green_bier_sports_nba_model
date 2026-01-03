import asyncio
from src.ingestion.the_odds import fetch_events
from src.utils.secrets import read_secret_strict

async def test_api():
    try:
        api_key = read_secret_strict('THE_ODDS_API_KEY')
        print(f'API key found, length: {len(api_key)}')

        games = await fetch_events('basketball_nba')
        print(f'Successfully fetched {len(games) if games else 0} games')

        if games:
            first_game = games[0]
            print(f'First game: {first_game.get("home_team", "?")} vs {first_game.get("away_team", "?")}')

    except Exception as e:
        print(f'API test failed: {e}')

if __name__ == "__main__":
    asyncio.run(test_api())