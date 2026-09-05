import distrax
import flax.linen as nn

from rlax.networks.mlp import MLP
from rlax.typing import Observation


class Actor(nn.Module):
    """An MLP torso feeding ``head``, which returns a distrax distribution.

    ``obs_key`` selects one entry of a dict observation as the torso's input;
    ``None`` feeds the observation as is.
    """

    head: nn.Module
    layer_sizes: tuple[int, ...] = (64, 64)
    obs_key: str | None = None

    @nn.compact
    def __call__(self, obs: Observation) -> distrax.Distribution:
        x = obs if self.obs_key is None else obs[self.obs_key]
        return self.head(MLP(self.layer_sizes)(x))
