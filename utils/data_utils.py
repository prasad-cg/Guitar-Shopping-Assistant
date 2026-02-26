"""
Data utility functions for the multi-agent system
"""
from typing import Dict, List, Any
from datetime import datetime


def format_agent_response(agent_name: str, content: str, metadata: Dict = None) -> Dict:
    """Format agent response with metadata"""
    return {
        "agent": agent_name,
        "timestamp": datetime.now().isoformat(),
        "content": content,
        "metadata": metadata or {}
    }


def parse_customer_preferences(customer_input: str) -> Dict[str, Any]:
    """Parse and structure customer preferences from input"""
    return {
        "raw_input": customer_input,
        "timestamp": datetime.now().isoformat()
    }


def aggregate_agent_responses(responses: List[Dict]) -> Dict:
    """Aggregate responses from multiple agents"""
    return {
        "total_agents": len(responses),
        "responses": responses,
        "aggregation_time": datetime.now().isoformat()
    }


def get_guitar_categories() -> List[str]:
    """Get available guitar categories"""
    return [
        "Acoustic Guitars",
        "Electric Guitars",
        "Bass Guitars",
        "Classical Guitars",
        "12-String Guitars"
    ]


def get_price_ranges() -> Dict[str, tuple]:
    """Get available price ranges"""
    return {
        "Budget": (0, 500),
        "Mid-Range": (500, 1500),
        "Premium": (1500, 5000),
        "Ultra-Premium": (5000, float('inf'))
    }


def get_playing_styles() -> List[str]:
    """Get available playing styles"""
    return [
        "Rock",
        "Blues",
        "Jazz",
        "Classical",
        "Folk",
        "Country",
        "Metal",
        "Pop",
        "Fingerstyle",
        "Strumming"
    ]


def format_conversation_history(history: List[Dict]) -> str:
    """Format conversation history for display"""
    formatted = []
    for msg in history:
        if msg.get("role") == "user":
            formatted.append(f"👤 Customer: {msg.get('content')}")
        elif msg.get("role") == "agent":
            agent_name = msg.get("agent", "Agent")
            formatted.append(f"🤖 {agent_name}: {msg.get('content')}")
        else:
            formatted.append(f"ℹ️ {msg.get('content')}")
    
    return "\n".join(formatted)
