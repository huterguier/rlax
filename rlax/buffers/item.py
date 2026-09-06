from dataclasses import dataclass

import jax
import jax.numpy as jnp

from rlax.buffers.buffer import Buffer
from rlax.typing import Array, Key, PyTree, Shape


@jax.tree_util.register_dataclass
@dataclass
class ItemBufferState:
    data: PyTree
    """``(capacity, ...)`` per leaf."""
    index: Array
    """Next slot to write."""
    size: Array
    """Slots filled so far, at most ``capacity``."""


class ItemBuffer(Buffer[ItemBufferState]):
    """A ring of single items with uniform sampling.

    :meth:`add` flattens all leading dims of ``item`` and writes the items
    contiguously; the oldest are overwritten once ``capacity`` is reached.
    :meth:`sample` returns items with leading dims ``shape``.
    """

    capacity: int

    def __init__(self, capacity: int):
        self.capacity = capacity

    def init(self, item: PyTree) -> ItemBufferState:
        return ItemBufferState(
            data=jax.tree.map(
                lambda x: jnp.zeros((self.capacity, *x.shape[1:]), x.dtype), item
            ),
            index=jnp.int32(0),
            size=jnp.int32(0),
        )

    def add(self, state: ItemBufferState, item: PyTree) -> ItemBufferState:
        def flatten(x, data):
            x = x.reshape(-1, *data.shape[1:])
            # Only the last ``capacity`` items survive a wrap-around, and a
            # scatter with repeated indices has no defined winner.
            return x[-self.capacity :]

        item = jax.tree.map(flatten, item, state.data)
        n = jax.tree.leaves(item)[0].shape[0]
        slots = (state.index + jnp.arange(n)) % self.capacity
        state.data = jax.tree.map(lambda d, x: d.at[slots].set(x), state.data, item)
        state.index = (state.index + n) % self.capacity
        state.size = jnp.minimum(state.size + n, self.capacity)
        return state

    def sample(self, key: Key, state: ItemBufferState, shape: Shape) -> PyTree:
        slots = jax.random.randint(key, tuple(shape), 0, state.size)
        return jax.tree.map(lambda d: d[slots], state.data)

    def can_sample(self, state: ItemBufferState) -> Array:
        return state.size > 0
