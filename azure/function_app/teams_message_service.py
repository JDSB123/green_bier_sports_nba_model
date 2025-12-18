"""
Microsoft Teams Message Service
Sends messages and Adaptive Cards to Teams channels
"""
import logging
import json
import os
import httpx
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class TeamsMessageService:
    """Service for sending messages to Microsoft Teams."""
    
    def __init__(self):
        self.webhook_url = os.getenv("TEAMS_WEBHOOK_URL", "")
        self.bot_id = os.getenv("TEAMS_BOT_ID", "")
        self.bot_password = os.getenv("TEAMS_BOT_PASSWORD", "")
        self.tenant_id = os.getenv("TEAMS_TENANT_ID", "")
    
    async def send_message_to_channel(
        self,
        channel_id: str,
        message: str,
        adaptive_card: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Send a message to a Teams channel.
        
        Args:
            channel_id: Teams channel ID
            message: Plain text message
            adaptive_card: Optional Adaptive Card JSON
        
        Returns:
            True if successful, False otherwise
        """
        if not self.bot_id or not self.bot_password:
            logger.warning("Teams bot credentials not configured")
            return False
        
        try:
            # Use Microsoft Graph API to send message
            # This requires proper authentication with Azure AD
            # For now, we'll use webhook if available, otherwise log
            if self.webhook_url:
                return await self._send_via_webhook(message, adaptive_card)
            else:
                logger.warning("Teams webhook URL not configured. Message not sent.")
                return False
                
        except Exception as e:
            logger.error(f"Error sending Teams message: {e}", exc_info=True)
            return False
    
    async def _send_via_webhook(
        self,
        message: str,
        adaptive_card: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Send message via Teams webhook (Incoming Webhook connector)."""
        try:
            payload = {
                "@type": "MessageCard",
                "@context": "https://schema.org/extensions",
                "summary": message[:200],
                "themeColor": "0078D4",
                "text": message
            }
            
            if adaptive_card:
                # Convert Adaptive Card to MessageCard format (simplified)
                payload.update({
                    "title": adaptive_card.get("title", "NBA Picks"),
                    "sections": self._adaptive_card_to_sections(adaptive_card)
                })
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.webhook_url,
                    json=payload,
                    timeout=10.0
                )
                response.raise_for_status()
                return True
                
        except Exception as e:
            logger.error(f"Error sending webhook: {e}")
            return False
    
    def _adaptive_card_to_sections(self, card: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert Adaptive Card to MessageCard sections format."""
        sections = []
        
        # Extract facts from Adaptive Card
        if "body" in card:
            for item in card["body"]:
                if item.get("type") == "FactSet":
                    facts = item.get("facts", [])
                    if facts:
                        section = {
                            "facts": [
                                {
                                    "name": fact.get("title", ""),
                                    "value": fact.get("value", "")
                                }
                                for fact in facts
                            ]
                        }
                        sections.append(section)
        
        return sections
    
    async def post_picks_to_channel(
        self,
        channel_id: str,
        picks_result: Dict[str, Any]
    ) -> bool:
        """
        Post picks result to Teams channel as formatted message.
        
        Args:
            channel_id: Teams channel ID
            picks_result: Result from generate_picks function
        
        Returns:
            True if successful
        """
        date_str = picks_result.get("date", "Unknown")
        total_plays = picks_result.get("total_plays", 0)
        games_count = picks_result.get("games", 0)
        predictions = picks_result.get("predictions", [])
        
        # Format message
        message_lines = [
            f"🏀 **NBA Picks - {date_str}**",
            "",
            f"**Total Plays:** {total_plays} across {games_count} games",
            ""
        ]
        
        for pred in predictions:
            matchup = pred.get("matchup", "Unknown")
            plays = pred.get("plays", [])
            
            message_lines.append(f"### {matchup}")
            for play in plays:
                period = play.get("period", "")
                market = play.get("market", "")
                pick = play.get("pick", "")
                edge = play.get("edge", 0)
                confidence = play.get("confidence", 0)
                
                fire_count = self._calculate_fire_count(confidence, abs(edge))
                fire_emoji = "🔥" * fire_count
                
                message_lines.append(
                    f"- **{period} {market}:** {pick} | Edge: {edge:+.1f} | Confidence: {confidence:.1%} {fire_emoji}"
                )
            message_lines.append("")
        
        message_lines.append(f"_Generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_")
        
        message = "\n".join(message_lines)
        
        # Create Adaptive Card for better formatting
        adaptive_card = self._create_picks_adaptive_card(picks_result)
        
        return await self.send_message_to_channel(
            channel_id,
            message,
            adaptive_card
        )
    
    def _create_picks_adaptive_card(self, picks_result: Dict[str, Any]) -> Dict[str, Any]:
        """Create Adaptive Card from picks result."""
        date_str = picks_result.get("date", "Unknown")
        total_plays = picks_result.get("total_plays", 0)
        games_count = picks_result.get("games", 0)
        predictions = picks_result.get("predictions", [])
        
        facts = []
        for pred in picks_result["predictions"]:
            matchup = pred.get("matchup", "Unknown")
            for play in pred.get("plays", []):
                period = play.get("period", "")
                market = play.get("market", "")
                pick = play.get("pick", "")
                edge = play.get("edge", 0)
                confidence = play.get("confidence", 0)
                
                fire_count = self._calculate_fire_count(confidence, abs(edge))
                fire_emoji = "🔥" * fire_count
                
                facts.append({
                    "title": f"{matchup} - {period} {market}",
                    "value": f"{pick} | Edge: {edge:+.1f} | Conf: {confidence:.1%} {fire_emoji}"
                })
        
        return {
            "type": "AdaptiveCard",
            "version": "1.4",
            "body": [
                {
                    "type": "TextBlock",
                    "size": "Large",
                    "weight": "Bolder",
                    "text": f"🏀 NBA Picks - {date_str}"
                },
                {
                    "type": "TextBlock",
                    "text": f"Total Plays: {total_plays} across {games_count} games",
                    "spacing": "Small"
                },
                {
                    "type": "FactSet",
                    "facts": facts[:20]  # Limit to 20 facts (Teams limit)
                }
            ]
        }
    
    def _calculate_fire_count(self, confidence: float, edge: float) -> int:
        """Calculate fire rating (1-5 fires)."""
        edge_norm = min(edge / 10.0, 1.0)
        combined = (confidence * 0.6) + (edge_norm * 0.4)
        
        if combined >= 0.85:
            return 5
        elif combined >= 0.70:
            return 4
        elif combined >= 0.60:
            return 3
        elif combined >= 0.52:
            return 2
        else:
            return 1


# Global instance
_teams_service: Optional[TeamsMessageService] = None


def get_teams_service() -> TeamsMessageService:
    """Get global Teams message service instance."""
    global _teams_service
    if _teams_service is None:
        _teams_service = TeamsMessageService()
    return _teams_service