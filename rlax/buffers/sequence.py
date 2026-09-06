from dataclasses import dataclass

import jax
import jax.numpy as jnp

from rlax.buffers.buffer import Buffer
from rlax.typing import Array, Key, PyTree, Shape


@jax.tree_util.register_dataclass
@dataclass
class SequenceBufferState:
    data: PyTree
    """``(lanes, capacity, ...)`` per leaf."""
    index: Array
    """Next slot to write, shared by all lanes."""
    size: Array
    """Slots filled per lane, at most ``capacity``."""


class SequenceBuffer(Buffer[SequenceBufferState]):
    """A ring per lane, sampling contiguous windows of ``length`` items.

    A lane is typically one environment. :meth:`init` takes the lane count
    from the batch axis of its sample. :meth:`add` takes ``(*time, lanes, ...)``:
    the last leading axis is the lane, everything before it is flattened as
    time. :meth:`sample` returns ``(length, *shape, ...)``, time first, so
    ``shape=(batch,)`` is what :meth:`~rlax.Agent.update` expects.

    Windows may cross episode boundaries; consumers mask on ``done``.
    """

    capacity: int
    length: int

    def __init__(self, capacity: int, length: int):
        assert 1 <= length <= capacity, (
            f"length ({length}) must be between 1 and capacity ({capacity})"
        )
        self.capacity = capacity
        self.length = length

    def init(self, item: PyTree) -> SequenceBufferState:
        return SequenceBufferState(
            data=jax.tree.map(
                lambda x: jnp.zeros((x.shape[0], self.capacity, *x.shape[1:]), x.dtype),
                item,
            ),
            index=jnp.int32(0),
            size=jnp.int32(0),
        )

    def add(self, state: SequenceBufferState, item: PyTree) -> SequenceBufferState:
        def flatten(x, data):
            lanes, *item_shape = data.shape[0], *data.shape[2:]
            x = x.reshape(-1, lanes, *item_shape)
            x = x[-self.capacity :]
            return jnp.moveaxis(x, 0, 1)  # (lanes, time, ...)

        item = jax.tree.map(flatten, item, state.data)
        n = jax.tree.leaves(item)[0].shape[1]
        slots = (state.index + jnp.arange(n)) % self.capacity
        state.data = jax.tree.map(lambda d, x: d.at[:, slots].set(x), state.data, item)
        state.index = (state.index + n) % self.capacity
        state.size = jnp.minimum(state.size + n, self.capacity)
        return state

    def sample(self, key: Key, state: SequenceBufferState, shape: Shape) -> PyTree:
        shape = tuple(shape)
        lanes = jax.tree.leaves(state.data)[0].shape[0]
        key_lane, key_start = jax.random.split(key)
        lane = jax.random.randint(key_lane, shape, 0, lanes)
        # Windows must lie within the filled region and not straddle the write
        # head, so starts are offsets from the oldest slot.
        oldest = state.index - state.size
        offset = jax.random.randint(key_start, shape, 0, state.size - self.length + 1)
        start = oldest + offset
        slots = (start[..., None] + jnp.arange(self.length)) % self.capacity
        window = jax.tree.map(lambda d: d[lane[..., None], slots], state.data)
        return jax.tree.map(lambda x: jnp.moveaxis(x, len(shape), 0), window)

    def can_sample(self, state: SequenceBufferState) -> Array:
        return state.size >= self.length
