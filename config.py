"""
Configuration file for Guitar Shopping Multi-Agent System
"""
import os
from dotenv import load_dotenv

load_dotenv()

# LLM Configuration
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
AZURE_OPENAI_API_VERSION = os.getenv("API_VERSION", "2025-01-01-preview")
# Optional: embedding deployment (must be an embedding-capable model deployed in Azure)
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1")
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.5))

# RAG Configuration
DATA_SOURCE_PATH = os.path.join(os.path.dirname(__file__), "data", "guitar_catalog.xlsx")
RAG_PDF_PATH = DATA_SOURCE_PATH  # backward compat alias
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K_RESULTS = 5

# Agent Configuration
AGENT_MAX_ITERATIONS = 10
AGENT_TIMEOUT = 60

# Streamlit Configuration
STREAMLIT_PAGE_TITLE = "🎸 AI Guitar Shopping Assistant - Multi-Agent Demo"
STREAMLIT_LAYOUT = "wide"

# Agent Roles
AGENTS = {
    "information": {
        "name": "Information Agent",
        "description": "Provides comprehensive information about guitar types, features, and specifications"
    },
    "recommendation": {
        "name": "Recommendation Agent",
        "description": "Analyzes customer preferences and recommends the best guitar options"
    },
    "negotiator": {
        "name": "Price Negotiator Agent",
        "description": "Negotiates pricing, discounts, and handles customer concerns about pricing"
    }
}

# Application Theme
APP_THEME = {
    "title": "🎸 Guitar Shopping Assistant",
    "subtitle": "Multi-Agent AI System Powered by LangGraph",
    "description": """
    This demo showcases a **multi-agentic AI system**.
    Multiple specialized agents collaborate to provide an exceptional customer experience.
    """
}
