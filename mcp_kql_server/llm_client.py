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
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error("Azure OpenAI chat call failed: %s", e)
        return None


async def generate_descriptions(tables: list[dict]) -> dict[str, str]:
    """Generate business descriptions for a batch of tables.

    Args:
        tables: list of {"table": name, "columns": {col: type, ...}}

    Returns:
        dict mapping table_name -> description string
    """
    client = _get_async_client()
    if not client:
        return {}

    # Build batch prompt
    table_lines = []
    for t in tables:
        col_names = ", ".join(list(t["columns"].keys())[:15])
        extra = f" (+{len(t['columns']) - 15} more)" if len(t["columns"]) > 15 else ""
        table_lines.append(f"- {t['table']}: {col_names}{extra}")

    tables_text = "\n".join(table_lines)

    system_prompt = (
        "You are an integration analyst expert. You are well familiar with IDP integrations like entra, google workspace, AWS and more. "
        "You also know TPRM and Procurement systems like Pivot, Panorays, Evisort, ZenGRC and more. "
        "For each source below, write a concise 1-2-sentences "
        "business description of what data it likely contains and what it's used for. "
        "Output ONLY lines in the format: table_name | description\n"
        "No extra text, no numbering, no blank lines."
    )
    user_prompt = f"Tables:\n{tables_text}"

    try:
        response = await client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        # Parse "table_name | description" lines
        result = {}
        raw = response.choices[0].message.content or ""
        for line in raw.strip().split("\n"):
            if "|" in line:
                parts = line.split("|", 1)
                name = parts[0].strip().lstrip("- ")
                desc = parts[1].strip()
                if name and desc:
                    result[name] = desc
        return result
    except Exception as e:
        logger.error("generate_descriptions failed: %s", e)
        return {}


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
        embedding = response.data[0].embedding
        logger.info("generate_embedding: success, dims=%d, text_preview=%.80s",
                     len(embedding), text)
        return embedding
    except Exception as e:
        logger.error("generate_embedding failed: %s", e)
        return None
