"""Azure OpenAI client for NL2KQL generation and embeddings."""

import logging

from .constants import (
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_DEPLOYMENT,
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
)

logger = logging.getLogger(__name__)

_async_client = None
_sync_client = None


def _get_async_client():
    """Lazy async singleton — returns None if env vars not configured."""
    global _async_client
    if _async_client is not None:
        return _async_client
    if not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_API_KEY or not AZURE_OPENAI_DEPLOYMENT:
        return None
    from openai import AsyncAzureOpenAI

    _async_client = AsyncAzureOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
    )
    logger.info("Azure OpenAI async client initialized (deployment: %s)", AZURE_OPENAI_DEPLOYMENT)
    return _async_client


def _get_sync_client():
    """Lazy sync singleton for embeddings — returns None if env vars not configured."""
    global _sync_client
    if _sync_client is not None:
        return _sync_client
    if not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_API_KEY or not AZURE_OPENAI_EMBEDDING_DEPLOYMENT:
        return None
    from openai import AzureOpenAI

    _sync_client = AzureOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
    )
    logger.info("Azure OpenAI sync client initialized (embedding deployment: %s)", AZURE_OPENAI_EMBEDDING_DEPLOYMENT)
    return _sync_client


async def generate_kql(system_prompt: str, user_prompt: str) -> str | None:
    """Call Azure OpenAI to generate a KQL query from natural language.

    Returns the raw LLM response string, or None if the client is not
    configured or the call fails (caller should fall back to schema-only).
    """
    client = _get_async_client()
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
        logger.error("Azure OpenAI chat call failed: %s", e)
        return None


def generate_embedding(text: str) -> list[float] | None:
    """Generate embedding using Azure OpenAI (sync, for use in memory.py).

    Returns a list of floats (1536-dim for text-embedding-3-small),
    or None if not configured or the call fails.
    """
    client = _get_sync_client()
    if not client:
        return None
    try:
        response = client.embeddings.create(
            model=AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
            input=text,
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error("Azure OpenAI embedding call failed: %s", e)
        return None
