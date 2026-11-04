import os

from langchain_core.runnables import RunnableConfig
from langfuse import get_client
from langfuse.langchain import CallbackHandler


TRACING_KEYS = ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY")


def is_tracing_enabled() -> bool:
    return all(os.environ.get(key) for key in TRACING_KEYS)


def build_run_config(debate_id: str, motion: str) -> RunnableConfig:
    if not is_tracing_enabled():
        return {}
    return {
        "callbacks": [CallbackHandler()],
        "run_name": "debate",
        "metadata": {
            "langfuse_session_id": debate_id,
            "langfuse_trace_name": f"debate: {motion}",
            "langfuse_tags": ["debate"],
        },
    }


def flush_traces() -> None:
    if is_tracing_enabled():
        get_client().flush()
