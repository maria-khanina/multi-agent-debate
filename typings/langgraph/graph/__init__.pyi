from collections.abc import Callable, Hashable, Mapping
from typing import Any, Generic, TypeVar

StateType = TypeVar("StateType")

START: str
END: str

class StateGraph(Generic[StateType]):
    def __init__(self, state_schema: type[StateType]) -> None: ...
    def add_node(
        self,
        node: str,
        action: Callable[..., Any],
    ) -> StateGraph[StateType]: ...
    def add_edge(self, start_key: str, end_key: str) -> StateGraph[StateType]: ...
    def add_conditional_edges(
        self,
        source: str,
        path: Callable[..., Hashable],
        path_map: Mapping[str, str],
    ) -> StateGraph[StateType]: ...
    def set_entry_point(self, key: str) -> StateGraph[StateType]: ...
    def compile(self) -> object: ...
