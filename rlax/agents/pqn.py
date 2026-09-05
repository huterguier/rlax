from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import lox
import optax

from rlax.agent import AgentBase
from rlax.typing import (
    Action,
    Array,
    Key,
    Optimizer,
    Params,
    PyTree,
    Timestep,
    Transition,
)


@jax.tree_util.register_dataclass
@dataclass
class PQNState:
    step: Array
    params: Params
    opt_state: optax.OptState


@dataclass
class PQNConfig:
    num_envs: int
    num_steps_rollout: int
    num_minibatches: int
    num_update_epochs: int
    epsilon_start: float
    epsilon_end: float
    epsilon_steps: int
    gamma: float
    lam: float
    max_grad_norm: float

    @property
    def num_steps_loop(self) -> int:
        return self.num_envs * self.num_steps_rollout

    def num_loops(self, num_steps: int) -> int:
        return int(num_steps // self.num_steps_loop)


class PQNAgent(AgentBase[PQNState]):
    """Parallelised Q-Network, as an :class:`~rlax.Agent`.

    The network maps a batch of observations to Q-values of shape
    ``(batch, num_actions)``. ``aux`` is the Q-values per step. ``state.step``
    counts the transitions consumed by :meth:`update` and drives the epsilon
    schedule.
    """

    config: PQNConfig
    network: Any
    optimizer: Optimizer

    def __init__(self, config: PQNConfig, network, optimizer):
        self.config = config
        self.network = network
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(self.config.max_grad_norm),
            optimizer,
        )

    def init(self, key: Key, timestep: Timestep) -> PQNState:
        params = self.network.init(key, timestep.next_obs)
        return PQNState(
            step=jnp.int32(0), params=params, opt_state=self.optimizer.init(params)
        )

    def epsilon(self, state: PQNState) -> Array:
        return optax.linear_schedule(
            self.config.epsilon_start,
            self.config.epsilon_end,
            self.config.epsilon_steps,
        )(state.step)

    def act(
        self,
        key: Key,
        state: PQNState,
        carry: None,
        timestep: Timestep,
        evaluation: bool = False,
    ) -> tuple[Action, PQNState, None, PyTree]:
        q = self.network.apply(state.params, timestep.next_obs)
        action_greedy = jnp.argmax(q, axis=-1)
        if evaluation:
            return action_greedy, state, carry, None
        key_explore, key_random = jax.random.split(key)
        batch, num_actions = q.shape
        action_random = jax.random.randint(key_random, (batch,), 0, num_actions)
        explore = jax.random.uniform(key_explore, (batch,)) < self.epsilon(state)
        action = jnp.where(explore, action_random, action_greedy)
        return action, state, carry, q

    def minibatches(self, key: Key, transitions: Transition, qs_next: Array) -> PyTree:
        def lambda_returns(transitions, qs_next):
            def lambda_step(carry, x):
                g_next = carry
                reward, q, done = x
                q_next = jnp.max(q, axis=-1)
                g = reward + self.config.gamma * (1 - done) * (
                    (1 - self.config.lam) * q_next + self.config.lam * g_next
                )
                return g, g

            g_next = jnp.max(qs_next[-1], axis=-1)
            _, returns = jax.lax.scan(
                lambda_step,
                g_next,
                (transitions.reward, qs_next, transitions.done),
                reverse=True,
            )
            return returns

        targets = lambda_returns(transitions, qs_next)
        batches = (transitions, targets)
        batch = jax.tree.map(lambda x: x.reshape(-1, *x.shape[2:]), batches)
        batch = jax.tree.map(lambda x: jax.random.permutation(key, x), batch)
        minibatches = jax.tree.map(
            lambda x: x.reshape(self.config.num_minibatches, -1, *x.shape[1:]),
            batch,
        )
        return minibatches

    def update(
        self, key: Key, state: PQNState, transitions: Transition, aux: PyTree
    ) -> PQNState:
        # aux[t] is Q(obs_t); Q(next_obs_t) is aux[t + 1] for all but the last
        # step. Post-reset observations are masked by ``done`` in the targets.
        qs = aux
        q_last = self.network.apply(state.params, transitions.next_obs[-1])
        qs_next = jnp.concatenate([qs[1:], q_last[None]], axis=0)
        minibatches = self.minibatches(key, transitions, qs_next)

        def loss(params, minibatch):
            batch, targets = minibatch
            qs = self.network.apply(params, batch.obs)
            qs_pred = jnp.take_along_axis(qs, batch.action[:, None], axis=-1).squeeze(
                -1
            )
            loss_value = jnp.mean((qs_pred - targets) ** 2)
            lox.log({"loss": loss_value, "q_pred": jnp.mean(qs_pred)})
            return loss_value

        def update_step(state, minibatch):
            grads = jax.grad(loss)(state.params, minibatch)
            updates, state.opt_state = self.optimizer.update(grads, state.opt_state)
            state.params = optax.apply_updates(state.params, updates)
            return state, None

        def update_epoch(state_minibatches, _):
            state, minibatches = state_minibatches
            state = jax.lax.scan(update_step, state, minibatches)[0]
            return (state, minibatches), None

        (state, _), _ = jax.lax.scan(
            update_epoch, (state, minibatches), length=self.config.num_update_epochs
        )
        num_steps, num_envs = transitions.reward.shape
        state.step += num_steps * num_envs
        return state
