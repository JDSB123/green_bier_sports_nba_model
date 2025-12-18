"""
Azure Function: Generate NBA Picks
Triggers: HTTP (GET/POST) or Teams Bot command
"""
import logging
import json
import os
import asyncio
from datetime import datetime, date
from typing import Dict, Any
import azure.functions as func
import sys
from pathlib import Path

# Add project root to path for imports
# In Azure Functions, the structure is: /home/site/wwwroot/
# From generate_picks/__init__.py, going up 2 levels gets us to wwwroot
FUNCTION_ROOT = Path(__file__).parent.parent  # This is /home/site/wwwroot in Azure
sys.path.insert(0, str(FUNCTION_ROOT))

# Import our NBA prediction modules
from src.config import settings
from src.prediction import UnifiedPredictionEngine
from scripts.build_rich_features import RichFeatureBuilder
from src.ingestion import the_odds
from src.utils.slate_analysis import get_target_date, fetch_todays_games
from src.utils.comprehensive_edge import calculate_comprehensive_edge
from src.modeling.edge_thresholds import get_edge_thresholds_for_game
from src.ingestion.betting_splits import fetch_public_betting_splits

logger = logging.getLogger(__name__)


async def _generate_picks_async(date_param: str, matchup_filter: str = None, output_format: str = 'json') -> Dict[str, Any]:
    """Async helper to generate picks."""
    # Resolve target date
    try:
        target_date = get_target_date(date_param)
    except ValueError as e:
        raise ValueError(f"Invalid date format: {e}")
    
    # Initialize prediction engine
    # In Azure Functions, models should be in the function app directory
    FUNCTION_ROOT = Path(__file__).parent.parent
    models_dir = FUNCTION_ROOT / "data" / "processed" / "models"
    if not models_dir.exists():
        # Fallback to settings path
        models_dir = Path(settings.data_processed_dir) / "models"
    
    engine = UnifiedPredictionEngine(models_dir=models_dir)
    feature_builder = RichFeatureBuilder(season=settings.current_season)
    
    # Get edge thresholds
    edge_thresholds = get_edge_thresholds_for_game(
        game_date=target_date,
        bet_types=["spread", "total", "moneyline", "1h_spread", "1h_total"]
    )
    
    # Fetch games
    games = await fetch_todays_games(target_date)
    if not games:
        return {
            "date": str(target_date),
            "message": "No games found for this date",
            "predictions": [],
            "total_plays": 0,
            "games": 0
        }
    
    # Filter by matchup if specified
    if matchup_filter:
        matchup_lower = matchup_filter.lower()
        games = [
            g for g in games
            if matchup_lower in g.get("home_team", "").lower() or 
               matchup_lower in g.get("away_team", "").lower()
        ]
        if not games:
            return {
                "date": str(target_date),
                "message": f"No games found matching '{matchup_filter}'",
                "predictions": [],
                "total_plays": 0,
                "games": 0
            }
    
    # Fetch betting splits
    try:
        splits_dict = await fetch_public_betting_splits(games, source="auto")
    except Exception as e:
        logger.warning(f"Could not fetch betting splits: {e}")
        splits_dict = {}
    
    # Process each game and generate picks
    predictions = []
    total_plays = 0
    
    for game in games:
        home_team = game.get("home_team")
        away_team = game.get("away_team")
        
        try:
            # Build features
            game_key = f"{away_team}@{home_team}"
            features = await feature_builder.build_game_features(
                home_team, away_team, betting_splits=splits_dict.get(game_key)
            )
            
            fh_features = features.copy()  # Simplified - in production use proper 1H features
            
            # Extract odds
            from src.utils.slate_analysis import extract_consensus_odds
            odds = extract_consensus_odds(game)
            
            # Calculate comprehensive edge
            comprehensive_edge = calculate_comprehensive_edge(
                features=features,
                fh_features=fh_features,
                odds=odds,
                game=game,
                betting_splits=splits_dict.get(game_key),
                edge_thresholds=edge_thresholds
            )
            
            # Count plays
            game_plays = 0
            plays = []
            
            for period in ["full_game", "first_half"]:
                period_label = "FG" if period == "full_game" else "1H"
                period_data = comprehensive_edge.get(period, {})
                
                for market in ["spread", "total", "moneyline"]:
                    market_data = period_data.get(market, {})
                    if market_data.get("pick"):
                        game_plays += 1
                        plays.append({
                            "period": period_label,
                            "market": market.upper(),
                            "pick": market_data.get("pick"),
                            "edge": market_data.get("edge", 0),
                            "confidence": market_data.get("confidence", 0),
                            "line": market_data.get("market_line"),
                            "odds": market_data.get("market_odds")
                        })
            
            total_plays += game_plays
            
            predictions.append({
                "matchup": f"{away_team} @ {home_team}",
                "home_team": home_team,
                "away_team": away_team,
                "commence_time": game.get("commence_time"),
                "plays": plays,
                "play_count": game_plays
            })
            
        except Exception as e:
            logger.error(f"Error processing {home_team} vs {away_team}: {e}")
            continue
    
    return {
        "date": str(target_date),
        "total_plays": total_plays,
        "games": len(predictions),
        "predictions": predictions,
        "generated_at": datetime.now().isoformat()
    }


def main(req: func.HttpRequest) -> func.HttpResponse:
    """
    Generate picks for NBA games.
    
    Query parameters:
    - date: 'today', 'tomorrow', or YYYY-MM-DD format
    - matchup: Optional filter for specific team (e.g., 'Lakers')
    - format: 'json' (default) or 'teams' (Teams Adaptive Card format)
    
    Example:
        GET /api/generate_picks?date=today
        GET /api/generate_picks?date=today&matchup=Lakers
        POST /api/generate_picks (with JSON body: {"date": "today", "matchup": "Lakers"})
    """
    try:
        # Get parameters from query string or JSON body
        if req.method == 'POST':
            try:
                req_body = req.get_json()
                date_param = req_body.get('date', 'today')
                matchup_filter = req_body.get('matchup')
                output_format = req_body.get('format', 'json')
            except ValueError:
                date_param = req.params.get('date', 'today')
                matchup_filter = req.params.get('matchup')
                output_format = req.params.get('format', 'json')
        else:
            date_param = req.params.get('date', 'today')
            matchup_filter = req.params.get('matchup')
            output_format = req.params.get('format', 'json')
        
        logger.info(f"Generating picks for date={date_param}, matchup={matchup_filter}")
        
        # Run async function
        try:
            result = asyncio.run(_generate_picks_async(date_param, matchup_filter, output_format))
        except ValueError as e:
            return func.HttpResponse(
                json.dumps({"error": str(e)}),
                status_code=400,
                mimetype="application/json"
            )
        
        # Optionally post to Teams channel if channel_id is provided
        channel_id = req.params.get('channel_id') or (req_body.get('channel_id') if req.method == 'POST' and req_body else None)
        if channel_id:
            try:
                from teams_message_service import get_teams_service
                teams_service = get_teams_service()
                asyncio.run(teams_service.post_picks_to_channel(channel_id, result))
            except Exception as e:
                logger.warning(f"Failed to post to Teams channel: {e}")
        
        # Return Teams Adaptive Card format if requested
        if output_format == 'teams':
            teams_card = _format_for_teams(result)
            return func.HttpResponse(
                json.dumps(teams_card),
                mimetype="application/json"
            )
        
        return func.HttpResponse(
            json.dumps(result),
            mimetype="application/json"
        )
        
    except Exception as e:
        logger.error(f"Error generating picks: {e}", exc_info=True)
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            status_code=500,
            mimetype="application/json"
        )


def _format_for_teams(result: Dict[str, Any]) -> Dict[str, Any]:
    """Format picks as Teams Adaptive Card."""
    facts = []
    
    for pred in result["predictions"]:
        matchup = pred["matchup"]
        for play in pred["plays"]:
            facts.append({
                "title": f"{matchup} - {play['period']} {play['market']}",
                "value": f"{play['pick']} | Edge: {play['edge']:.1f} | Confidence: {play['confidence']:.1%}"
            })
    
    return {
        "type": "message",
        "attachments": [{
            "contentType": "application/vnd.microsoft.card.adaptive",
            "content": {
                "type": "AdaptiveCard",
                "version": "1.4",
                "body": [
                    {
                        "type": "TextBlock",
                        "size": "Large",
                        "weight": "Bolder",
                        "text": f"🏀 NBA Picks - {result['date']}"
                    },
                    {
                        "type": "TextBlock",
                        "text": f"Total Plays: {result['total_plays']} across {result['games']} games",
                        "spacing": "Small"
                    },
                    {
                        "type": "FactSet",
                        "facts": facts
                    }
                ]
            }
        }]
    }