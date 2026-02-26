"""
LLM Setup and Configuration
"""
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from config import AZURE_OPENAI_DEPLOYMENT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, MODEL_NAME, TEMPERATURE


def get_llm():
    """Initialize and return the LLM instance"""
    return AzureChatOpenAI(
        api_key=AZURE_OPENAI_API_KEY,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        deployment_name=AZURE_OPENAI_DEPLOYMENT,
        api_version="2025-01-01-preview",
        temperature=TEMPERATURE,
        max_tokens=4096
    )


def get_llm_with_custom_temp(temperature):
    """Initialize LLM with custom temperature"""
    return AzureChatOpenAI(
        api_key=AZURE_OPENAI_API_KEY,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        deployment_name=AZURE_OPENAI_DEPLOYMENT,
        api_version="2025-01-01-preview",
        temperature=temperature,
        max_tokens=4096
    )
