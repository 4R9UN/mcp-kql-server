"""Async Azure OpenAI client for NL2KQL generation."""

import logging

from .constants import (
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_DEPLOYMENT,
)

logger = logging.getLogger(__name__)

_client = None


def _get_client():
    """Lazy singleton — returns None if env vars not configured."""
    global _client
    if _client is not None:
        return _client
    if not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_API_KEY or not AZURE_OPENAI_DEPLOYMENT:
        return None
    from openai import AsyncAzureOpenAI

    _client = AsyncAzureOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
    )
    logger.info("Azure OpenAI client initialized (deployment: %s)", AZURE_OPENAI_DEPLOYMENT)
    return _client


async def generate_kql(system_prompt: str, user_prompt: str) -> str | None:
    """Call Azure OpenAI to generate a KQL query from natural language.

    Returns the raw LLM response string, or None if the client is not
    configured or the call fails (caller should fall back to schema-only).
    """
    client = _get_client()
    if not client:
        return None
    try:
        response = await client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=500,
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error("Azure OpenAI call failed: %s", e)
        return None
