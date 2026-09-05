import flax.linen as nn
import jax.numpy as jnp

from rlax.networks.mlp import MLP
from rlax.typing import Array, Observation


class Critic(nn.Module):
    """An MLP torso feeding a scalar value output, returned with shape
    ``(batch,)``.

    ``obs_key`` selects one entry of a dict observation as the torso's input;
    ``None`` feeds the observation as is.
    """

    layer_sizes: tuple[int, ...] = (64, 64)
    obs_key: str | None = None

    @nn.compact
    def __call__(self, obs: Observation) -> Array:
        x = obs if self.obs_key is None else obs[self.obs_key]
        x = MLP(self.layer_sizes)(x)
        value = nn.Dense(1, kernel_init=nn.initializers.orthogonal(1.0))(x)
        return jnp.squeeze(value, -1)
