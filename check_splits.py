import asyncio
from src.ingestion.betting_splits import fetch_public_betting_splits
from src.ingestion.the_odds import fetch_events
from src.utils.logging import get_logger

logger = get_logger(__name__)

async def check_splits():
    try:
        # Try to fetch games first
        games = await fetch_events('basketball_nba')
        if not games:
            print('No games found from The Odds API')
            return

        print(f'Found {len(games)} games')

        # Try to get splits
        splits_dict = await fetch_public_betting_splits(games, source='auto')
        print(f'Got betting splits for {len(splits_dict)} games')

        if splits_dict:
            game_key = list(splits_dict.keys())[0]
            splits = splits_dict[game_key]
            print(f'Sample splits for {game_key}:')
            print(f'  Source: {splits.source}')
            print(f'  Has real data: {splits.source not in ("mock", "synthetic")}')
            print(f'  Spread public home: {splits.spread_home_ticket_pct}%')
        else:
            print('No betting splits data available')

    except Exception as e:
        print(f'Error: {e}')

if __name__ == "__main__":
    asyncio.run(check_splits())