from abc import abstractmethod
from typing import Protocol

from rlax.typing import Array, Key, PyTree, Shape


class Buffer[TState](Protocol):
    """Experience storage, decoupled from what is stored.

    Items are arbitrary pytrees of arrays; a buffer never looks inside them.
    Like an :class:`~rlax.Agent`, a buffer is a stateless object whose state is
    a pytree threaded through :meth:`add` and :meth:`sample`, so it can live
    inside a scanned training loop.
    """

    @abstractmethod
    def init(self, item: PyTree) -> TState:
        """Create the state. ``item`` is a ``(batch, ...)`` sample, used for
        shapes and dtypes only, as in :meth:`~rlax.Agent.init`."""
        ...

    @abstractmethod
    def add(self, state: TState, item: PyTree) -> TState:
        """Store ``item``, which may carry any number of leading dims."""
        ...

    @abstractmethod
    def sample(self, key: Key, state: TState, shape: Shape) -> PyTree:
        """Draw items with leading dims ``shape``. Only valid once
        :meth:`can_sample` holds."""
        ...

    @abstractmethod
    def can_sample(self, state: TState) -> Array:
        """Scalar bool: whether the buffer holds enough for one :meth:`sample`."""
        ...
