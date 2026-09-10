from __future__ import annotations

from core.config import LLM_PROVIDER

# Retries transient errors (e.g. rate limits) at the client level, so the returned
# object stays a plain BaseChatModel (create_react_agent needs .bind_tools(), which
# a Runnable wrapper like RunnableRetry from .with_retry() does not expose).
MAX_RETRIES = 5


def make_llm():
    """Build the configured provider's chat LLM, with built-in retry on transient errors."""
    if LLM_PROVIDER == "groq":
        from langchain_groq import ChatGroq
        from core.config import GROQ_MODEL

        return ChatGroq(model=GROQ_MODEL, max_retries=MAX_RETRIES)
    elif LLM_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI
        from core.config import OPENAI_MODEL

        return ChatOpenAI(model=OPENAI_MODEL, max_retries=MAX_RETRIES)
    elif LLM_PROVIDER == "github":
        from langchain_openai import ChatOpenAI
        from core.config import GITHUB_BASE_URL, GITHUB_MODEL, GITHUB_TOKEN

        return ChatOpenAI(
            model=GITHUB_MODEL,
            base_url=GITHUB_BASE_URL,
            api_key=GITHUB_TOKEN,
            max_retries=MAX_RETRIES,
        )
    else:
        from langchain_google_genai import ChatGoogleGenerativeAI
        from core.config import GEMINI_MODEL

        return ChatGoogleGenerativeAI(model=GEMINI_MODEL, max_retries=MAX_RETRIES)

