from collections.abc import Iterator
from typing import Any, Protocol, TypeVar

from langchain_core.runnables import RunnableConfig

StateType = TypeVar("StateType", contravariant=True)


class RunnableGraph(Protocol[StateType]):
    def invoke(self, state: StateType) -> dict[str, Any]: ...

    def stream(
        self,
        state: StateType,
        config: RunnableConfig,
    ) -> Iterator[dict[str, dict[str, Any]]]: ...
