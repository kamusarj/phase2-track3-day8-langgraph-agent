"""LLM factory helper.

Provides a simple interface to create LLM clients for use in nodes.
Supports multiple providers with automatic fallback.

Usage in nodes:
    from .llm import get_llm
    llm = get_llm()
    response = llm.invoke("Hello")
"""

from __future__ import annotations

import os
from pathlib import Path


def _load_env() -> None:
    """Load .env file from project root if python-dotenv is available."""
    try:
        from dotenv import load_dotenv
        # Walk up from this file to find .env
        env_path = Path(__file__).resolve().parent.parent.parent / ".env"
        if env_path.exists():
            load_dotenv(env_path, override=False)
    except ImportError:
        pass  # dotenv not installed, rely on environment variables


# Auto-load .env on module import
_load_env()


def get_llm(model: str | None = None, temperature: float = 0.0):
    """Create an LLM client from environment configuration.

    Checks for API keys in this order:
    1. GEMINI_API_KEY → ChatGoogleGenerativeAI
    2. DEEPSEEK_API_KEY → ChatOpenAI (OpenAI-compatible endpoint)
    3. MISTRAL_API_KEY → ChatMistralAI
    4. OPENAI_API_KEY → ChatOpenAI
    5. ANTHROPIC_API_KEY → ChatAnthropic

    Override model with the `model` parameter or LLM_MODEL env var.
    """
    if os.getenv("GEMINI_API_KEY"):
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore
        except ImportError as exc:
            raise RuntimeError("Install: pip install langchain-google-genai") from exc
        return ChatGoogleGenerativeAI(  # type: ignore
            model=model or os.getenv("LLM_MODEL", "gemini-2.0-flash"),
            google_api_key=os.getenv("GEMINI_API_KEY"),
            temperature=temperature,
            max_retries=3,
        )

    if os.getenv("DEEPSEEK_API_KEY"):
        try:
            from langchain_openai import ChatOpenAI  # type: ignore
        except ImportError as exc:
            raise RuntimeError("Install: pip install langchain-openai") from exc
        model_name = str(model or os.getenv("LLM_MODEL", "deepseek-chat"))
        api_key_str = str(os.getenv("DEEPSEEK_API_KEY") or "")
        return ChatOpenAI(  # type: ignore
            model=model_name,
            api_key=api_key_str,  # type: ignore[arg-type]
            base_url="https://api.deepseek.com",
            temperature=temperature,
        )

    if os.getenv("MISTRAL_API_KEY"):
        try:
            from langchain_mistralai import ChatMistralAI  # type: ignore
        except ImportError as exc:
            raise RuntimeError("Install: pip install langchain-mistralai") from exc
        model_name = str(model or os.getenv("LLM_MODEL", "mistral-small-latest"))
        api_key_str = str(os.getenv("MISTRAL_API_KEY") or "")
        return ChatMistralAI(  # type: ignore
            model=model_name,
            api_key=api_key_str,  # type: ignore[arg-type]
            temperature=temperature,
        )

    if os.getenv("OPENAI_API_KEY"):
        try:
            from langchain_openai import ChatOpenAI  # type: ignore
        except ImportError as exc:
            raise RuntimeError("Install: pip install langchain-openai") from exc
        model_name = str(model or os.getenv("LLM_MODEL", "gpt-4o-mini"))
        return ChatOpenAI(  # type: ignore
            model=model_name,
            temperature=temperature,
        )

    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            from langchain_anthropic import ChatAnthropic  # type: ignore
        except ImportError as exc:
            raise RuntimeError("Install: pip install langchain-anthropic") from exc
        model_name = str(model or os.getenv("LLM_MODEL", "claude-sonnet-4-20250514"))
        return ChatAnthropic(  # type: ignore
            model=model_name,
            temperature=temperature,
        )

    raise RuntimeError(
        "No LLM API key found. Set GEMINI_API_KEY, DEEPSEEK_API_KEY, MISTRAL_API_KEY, "
        "OPENAI_API_KEY, or ANTHROPIC_API_KEY in .env\n"
        "See .env.example for configuration."
    )


def get_llm_with_fallback(model: str | None = None, temperature: float = 0.0):
    """Try each available provider in priority order.

    Returns a RunnableWithFallbacks if multiple providers are configured,
    or a single LLM if only one is available.
    """
    llms = []
    providers = [
        ("GEMINI_API_KEY", None),
        ("DEEPSEEK_API_KEY", None),
        ("MISTRAL_API_KEY", None),
        ("OPENAI_API_KEY", None),
        ("ANTHROPIC_API_KEY", None),
    ]

    for key_name, _ in providers:
        if os.getenv(key_name):
            # Temporarily isolate this key to build a single-provider LLM
            try:
                saved = {}
                for other_key, _ in providers:
                    if other_key != key_name and os.getenv(other_key):
                        saved[other_key] = os.environ.pop(other_key)
                llm = get_llm(model=model, temperature=temperature)
                llms.append(llm)
            except Exception:
                pass
            finally:
                for k, v in saved.items():
                    os.environ[k] = v

    if not llms:
        raise RuntimeError(
            "No LLM API key found. Set GEMINI_API_KEY, DEEPSEEK_API_KEY, MISTRAL_API_KEY, "
            "OPENAI_API_KEY, or ANTHROPIC_API_KEY in .env"
        )

    if len(llms) == 1:
        return llms[0]

    # Use LangChain's with_fallbacks for automatic retry across providers
    primary = llms[0]
    return primary.with_fallbacks(llms[1:])
