"""Policy heads: modules mapping features to a distrax distribution over actions."""

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from rlax.typing import Array

_GH_NODES, _GH_WEIGHTS = np.polynomial.hermite.hermgauss(8)
_SQRT2 = 2.0**0.5
_SQRT_PI = np.pi**0.5


class SquashedNormal(distrax.MultivariateNormalDiag):
    """Diagonal Gaussian over pre-tanh actions whose entropy is that of the
    tanh-squashed action.

    ``sample`` and ``log_prob`` are those of the plain Gaussian, so an
    algorithm works in the unbounded space. Only ``entropy`` differs: it adds
    the expected tanh log-determinant, evaluated by Gauss-Hermite quadrature,
    so an entropy bonus measures spread in the space the environment sees.

    Pair with ``gxm.wrappers.SquashActions``, which applies the tanh and
    keeps the pre-tanh action in the timestep.
    """

    def entropy(self) -> Array:
        x = self.loc[..., None] + self.scale_diag[..., None] * _SQRT2 * _GH_NODES
        log_det = distrax.Tanh().forward_log_det_jacobian(x)
        correction = jnp.sum(log_det * _GH_WEIGHTS, axis=-1) / _SQRT_PI
        return super().entropy() + jnp.sum(correction, axis=-1)


class CategoricalHead(nn.Module):
    """Logits over ``num_actions`` discrete actions."""

    num_actions: int

    @nn.compact
    def __call__(self, x: Array) -> distrax.Categorical:
        logits = nn.Dense(
            self.num_actions, kernel_init=nn.initializers.orthogonal(0.01)
        )(x)
        return distrax.Categorical(logits=logits)


class GaussianHead(nn.Module):
    """Diagonal Gaussian over ``action_size`` continuous actions, with a
    state-dependent scale of ``softplus(x) + min_std``.

    With ``squash=True`` the head returns a :class:`SquashedNormal` and must be
    used with an environment wrapped in ``gxm.wrappers.SquashActions``.
    """

    action_size: int
    squash: bool = False
    min_std: float = 1e-3

    @nn.compact
    def __call__(self, x: Array) -> distrax.MultivariateNormalDiag:
        loc_scale = nn.Dense(
            2 * self.action_size, kernel_init=nn.initializers.orthogonal(0.01)
        )(x)
        loc, scale = jnp.split(loc_scale, 2, axis=-1)
        scale = jax.nn.softplus(scale) + self.min_std
        dist = SquashedNormal if self.squash else distrax.MultivariateNormalDiag
        return dist(loc, scale)
