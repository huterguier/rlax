from collections.abc import Callable

import flax.linen as nn

from rlax.typing import Array


class MLP(nn.Module):
    """Stack of dense layers, each followed by ``activation``."""

    layer_sizes: tuple[int, ...]
    activation: Callable[[Array], Array] = nn.tanh
    kernel_init: nn.initializers.Initializer = nn.initializers.orthogonal(2**0.5)

    @nn.compact
    def __call__(self, x: Array) -> Array:
        for size in self.layer_sizes:
            x = self.activation(nn.Dense(size, kernel_init=self.kernel_init)(x))
        return x
