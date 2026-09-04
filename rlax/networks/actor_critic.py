import distrax
import flax.linen as nn
import jax.numpy as jnp

from rlax.networks.mlp import MLP
from rlax.typing import Array


class ActorCritic(nn.Module):
    """Two separate MLP torsos: one feeding ``head``, one feeding a scalar
    value output. Returns ``(dist, value)`` as :class:`~rlax.PPO` expects.

    ``actor_obs_size`` gives an asymmetric actor-critic: the policy reads the
    first ``actor_obs_size`` observation entries, the value torso the whole
    vector. ``obs_mean``/``obs_std``, when given, standardize the observation
    before either torso.
    """

    head: nn.Module
    policy_layers: tuple[int, ...] = (64, 64)
    value_layers: tuple[int, ...] = (64, 64)
    actor_obs_size: int | None = None
    obs_mean: Array | None = None
    obs_std: Array | None = None

    @nn.compact
    def __call__(self, obs: Array) -> tuple[distrax.Distribution, Array]:
        if self.obs_mean is not None:
            obs = (obs - self.obs_mean) / self.obs_std

        x = obs if self.actor_obs_size is None else obs[..., : self.actor_obs_size]
        dist = self.head(MLP(self.policy_layers, name="policy")(x))

        x = MLP(self.value_layers, name="value")(obs)
        value = nn.Dense(1, kernel_init=nn.initializers.orthogonal(1.0))(x)
        return dist, jnp.squeeze(value, -1)
