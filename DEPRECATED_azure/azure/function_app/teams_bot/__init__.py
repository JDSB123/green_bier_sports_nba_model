"""
Microsoft Teams Bot Integration for NBA Picks
Handles Teams bot commands and sends responses
"""
import logging
import json
import os
import asyncio
import azure.functions as func
from typing import Dict, Any
import sys
from pathlib import Path

# Import the generate picks function
FUNCTION_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(FUNCTION_ROOT))
from generate_picks import _generate_picks_async, _format_for_teams

logger = logging.getLogger(__name__)


def main(req: func.HttpRequest) -> func.HttpResponse:
    """
    Microsoft Teams Bot endpoint.
    
    Handles:
    - Bot framework messages
    - Command parsing ("run picks for today", "run picks for Lakers vs Celtics", etc.)
    - Sending Adaptive Cards to Teams
    
    Expected Teams commands:
    - "run picks for today"
    - "run picks for tomorrow"
    - "run picks for Lakers"
    - "run picks for Lakers vs Celtics"
    - "track picks" (returns live tracker HTML)
    """
    try:
        # Get request body
        try:
            req_body = req.get_json()
        except ValueError:
            return func.HttpResponse(
                json.dumps({"error": "Invalid JSON body"}),
                status_code=400,
                mimetype="application/json"
            )
        
        # Handle Teams Bot Framework activity
        activity_type = req_body.get("type")
        
        if activity_type == "message":
            # Extract command from message text
            message_text = req_body.get("text", "").strip().lower()
            conversation = req_body.get("conversation", {})
            
            # Parse command
            date_param = "today"
            matchup_filter = None
            
            if "tomorrow" in message_text:
                date_param = "tomorrow"
            elif "today" in message_text or "slate" in message_text:
                date_param = "today"
            
            # Extract matchup (e.g., "Lakers", "Lakers vs Celtics")
            # Simple parsing - can be enhanced
            if "vs" in message_text or "@" in message_text:
                # Try to extract team names
                parts = message_text.split("vs") if "vs" in message_text else message_text.split("@")
                if len(parts) >= 2:
                    # Take first team name
                    matchup_filter = parts[0].split()[-1]  # Simple extraction
            
            # Check for single team mention
            common_teams = [
                "lakers", "celtics", "warriors", "nets", "knicks", "heat",
                "bucks", "76ers", "suns", "nuggets", "mavericks", "clippers"
            ]
            for team in common_teams:
                if team in message_text and matchup_filter is None:
                    matchup_filter = team
                    break
            
            # Generate picks
            try:
                result = asyncio.run(_generate_picks_async(date_param, matchup_filter, 'teams'))
                
                # Format as Teams Adaptive Card
                teams_response = _format_for_teams(result)
                
                # Return Teams Bot Framework response
                return func.HttpResponse(
                    json.dumps(teams_response),
                    mimetype="application/json"
                )
                
            except Exception as e:
                logger.error(f"Error generating picks: {e}", exc_info=True)
                error_response = {
                    "type": "message",
                    "text": f"Error generating picks: {str(e)}"
                }
                return func.HttpResponse(
                    json.dumps(error_response),
                    mimetype="application/json"
                )
        
        elif activity_type == "invoke" and req_body.get("name") == "adaptiveCard/action":
            # Handle Adaptive Card actions (e.g., "Refresh", "View Details")
            action_type = req_body.get("value", {}).get("action")
            
            if action_type == "refresh":
                # Re-generate picks
                date_param = req_body.get("value", {}).get("date", "today")
                matchup_filter = req_body.get("value", {}).get("matchup")
                
                try:
                    result = asyncio.run(_generate_picks_async(date_param, matchup_filter, 'teams'))
                    teams_response = _format_for_teams(result)
                    return func.HttpResponse(
                        json.dumps(teams_response),
                        mimetype="application/json"
                    )
                except Exception as e:
                    error_response = {
                        "type": "message",
                        "text": f"Error refreshing picks: {str(e)}"
                    }
                    return func.HttpResponse(
                        json.dumps(error_response),
                        mimetype="application/json"
                    )
        
        # Default: echo back for testing
        return func.HttpResponse(
            json.dumps({
                "type": "message",
                "text": "NBA Picks Bot is ready. Try: 'run picks for today'"
            }),
            mimetype="application/json"
        )
        
    except Exception as e:
        logger.error(f"Error in Teams bot: {e}", exc_info=True)
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            status_code=500,
            mimetype="application/json"
        )