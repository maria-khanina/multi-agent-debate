from functools import partial
import time
from typing import Callable, TypeVar, cast

from configurations.config_manager import get_chat_model
from debate_settings import BACKOFF_ATTEMPTS, BACKOFF_BASE_SECONDS
from langchain_core.messages import HumanMessage

ResultType = TypeVar("ResultType")


def execute_with_backoff(api_call: Callable[[], ResultType]) -> ResultType:
    for attempt in range(BACKOFF_ATTEMPTS - 1):
        try:
            return api_call()
        except Exception:
            print(f"[Backoff] API call failed, attempt {attempt + 1}")
            time.sleep(BACKOFF_BASE_SECONDS * 2 ** attempt)
    try:
        return api_call()
    except Exception:
        print(f"[Backoff] API call failed, attempt {BACKOFF_ATTEMPTS}")
        raise


def ask_structured(
    role: str,
    schema: type[ResultType],
    prompt_text: str,
) -> ResultType:
    language_model = get_chat_model(role).with_structured_output(schema)
    answer = language_model.invoke([HumanMessage(content=prompt_text)])
    return cast(ResultType, answer)


def ask_structured_with_backoff(
    role: str,
    schema: type[ResultType],
    prompt_text: str,
) -> ResultType:
    return execute_with_backoff(partial(ask_structured, role, schema, prompt_text))
