from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp

from rlax.agent import Agent
from rlax.typing import Action, Array, Key, PyTree, Timestep, Transition


@jax.tree_util.register_dataclass
@dataclass
class NormalizeState:
    inner: PyTree
    mean: Array
    """Frozen for the current rollout; what ``act`` and ``update`` normalize with."""
    std: Array
    count: Array
    run_mean: Array
    run_m2: Array


def welford_batch(
    count: Array, mean: Array, m2: Array, x: Array
) -> tuple[Array, Array, Array]:
    """Fold a ``(batch, ...)`` sample into running Welford statistics."""
    n = x.shape[0]
    batch_mean = x.mean(axis=0)
    batch_m2 = ((x - batch_mean) ** 2).sum(axis=0)
    delta = batch_mean - mean
    total = count + n
    mean = mean + delta * n / total
    m2 = m2 + batch_m2 + delta**2 * count * n / total
    return total, mean, m2


class Normalize[TCarry](Agent[NormalizeState, TCarry]):
    """Standardize observations with running statistics.

    The wrapped agent only ever sees ``(obs - mean) / std``. Statistics are
    accumulated from every training ``act`` call, but the ``mean``/``std`` used
    for normalizing stay frozen for a whole rollout and its ``update``, so the
    data an update sees is normalized the same way it was when acted upon. The
    frozen values are refreshed at the end of each ``update``.

    Observations must be flat arrays. ``initial`` seeds the frozen ``mean`` and
    ``std`` used for the very first rollout.
    """

    agent: Agent
    eps: float
    initial: tuple[Array, Array] | None

    def __init__(
        self,
        agent: Agent,
        eps: float = 1e-3,
        initial: tuple[Array, Array] | None = None,
    ):
        self.agent = agent
        self.eps = eps
        self.initial = initial

    def normalize(self, state: NormalizeState, obs: Array) -> Array:
        return (obs - state.mean) / state.std

    def init(self, key: Key, timestep: Timestep) -> NormalizeState:
        obs = timestep.next_obs
        shape = obs.shape[1:]
        if self.initial is None:
            mean, std = jnp.zeros(shape), jnp.ones(shape)
        else:
            mean, std = (jnp.asarray(x, obs.dtype) for x in self.initial)
        state = NormalizeState(
            inner=None,
            mean=mean,
            std=std,
            count=jnp.zeros((), obs.dtype),
            run_mean=jnp.zeros(shape, obs.dtype),
            run_m2=jnp.zeros(shape, obs.dtype),
        )
        timestep = replace(timestep, next_obs=self.normalize(state, obs))
        state.inner = self.agent.init(key, timestep)
        return state

    def initialize_carry(
        self, key: Key, state: NormalizeState, timestep: Timestep
    ) -> TCarry:
        timestep = replace(timestep, next_obs=self.normalize(state, timestep.next_obs))
        return self.agent.initialize_carry(key, state.inner, timestep)

    def act(
        self,
        key: Key,
        state: NormalizeState,
        carry: TCarry,
        timestep: Timestep,
        evaluation: bool = False,
    ) -> tuple[Action, NormalizeState, TCarry, PyTree]:
        obs = timestep.next_obs
        if not evaluation:
            state.count, state.run_mean, state.run_m2 = welford_batch(
                state.count, state.run_mean, state.run_m2, obs
            )
        timestep = replace(timestep, next_obs=self.normalize(state, obs))
        action, state.inner, carry, aux = self.agent.act(
            key, state.inner, carry, timestep, evaluation
        )
        return action, state, carry, aux

    def update(
        self, key: Key, state: NormalizeState, transitions: Transition, aux: PyTree
    ) -> NormalizeState:
        transitions = replace(
            transitions,
            obs=self.normalize(state, transitions.obs),
            next_obs=self.normalize(state, transitions.next_obs),
        )
        state.inner = self.agent.update(key, state.inner, transitions, aux)
        state.mean = state.run_mean
        state.std = jnp.maximum(
            jnp.sqrt(state.run_m2 / jnp.maximum(state.count, 1)), self.eps
        )
        return state
