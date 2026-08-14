from abc import abstractmethod
from dataclasses import dataclass
from typing import Protocol

from rlax.typing import Array, Key, PyTree


@dataclass
class AlgorithmState:
    step: Array


class Algorithm[TAlgorithmState: AlgorithmState](Protocol):
    @abstractmethod
    def init(self, key: Key) -> TAlgorithmState: ...

    @abstractmethod
    def train(
        self, key: Key, state: TAlgorithmState, num_steps: int
    ) -> TAlgorithmState: ...

    @abstractmethod
    def evaluate(
        self, key: Key, state: TAlgorithmState, num_steps: int, num_envs: int
    ) -> dict[str, PyTree]: ...
