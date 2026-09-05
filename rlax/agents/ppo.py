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
class PPOState:
    params: Params
    opt_state: optax.OptState


@dataclass
class PPOConfig:
    num_envs: int
    num_steps_rollout: int
    num_minibatches: int
    num_update_epochs: int
    gamma: float
    lam: float
    ratio_clip: float
    value_coefficient: float
    entropy_coefficient: float
    normalize_advantage: bool
    max_grad_norm: float
    value_clip: float | None = None

    def __post_init__(self):
        assert self.num_update_epochs >= 1, (
            f"num_update_epochs ({self.num_update_epochs}) must be >= 1"
        )
        assert self.num_steps_loop % self.num_minibatches == 0, (
            f"num_envs * num_steps_rollout ({self.num_steps_loop}) must be divisible "
            f"by num_minibatches ({self.num_minibatches})"
        )

    @property
    def num_steps_loop(self) -> int:
        return self.num_envs * self.num_steps_rollout

    def num_loops(self, num_steps: int) -> int:
        return int(num_steps // self.num_steps_loop)


class PPOAgent(AgentBase[PPOState]):
    """Proximal Policy Optimization, as an :class:`~rlax.Agent`.

    ``actor`` maps a batch of observations to a distrax distribution with batch
    shape ``(batch,)``; ``critic`` maps it to values of shape ``(batch,)``. Their
    parameters live under ``params["actor"]`` and ``params["critic"]`` and share
    one optimizer. ``aux`` is ``(log_prob, value)`` per step.
    """

    config: PPOConfig
    actor: Any
    critic: Any
    optimizer: Optimizer

    def __init__(self, config: PPOConfig, actor, critic, optimizer):
        self.config = config
        self.actor = actor
        self.critic = critic
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(self.config.max_grad_norm),
            optimizer,
        )

    def init(self, key: Key, timestep: Timestep) -> PPOState:
        key_actor, key_critic = jax.random.split(key)
        params = {
            "actor": self.actor.init(key_actor, timestep.next_obs),
            "critic": self.critic.init(key_critic, timestep.next_obs),
        }
        return PPOState(params=params, opt_state=self.optimizer.init(params))

    def act(
        self,
        key: Key,
        state: PPOState,
        carry: None,
        timestep: Timestep,
        evaluation: bool = False,
    ) -> tuple[Action, PPOState, None, PyTree]:
        dist = self.actor.apply(state.params["actor"], timestep.next_obs)
        if evaluation:
            return dist.mode(), state, carry, None
        value = self.critic.apply(state.params["critic"], timestep.next_obs)
        assert value.ndim == 1, f"expected value of shape (batch,), got {value.shape}"
        action, log_prob = dist.sample_and_log_prob(seed=key)
        return action, state, carry, (log_prob, value)

    def advantages(
        self, transitions: Transition, values: Array, value_last: Array
    ) -> Array:
        def gae_step(carry, x):
            advantage, value_next = carry
            reward, done, value = x
            delta = reward + self.config.gamma * (1 - done) * value_next - value
            advantage = delta + (
                self.config.gamma * self.config.lam * (1 - done) * advantage
            )
            return (advantage, value), advantage

        _, advantages = jax.lax.scan(
            gae_step,
            (jnp.zeros_like(value_last), value_last),
            (transitions.reward, transitions.done, values),
            reverse=True,
        )
        return advantages

    def minibatches(self, key: Key, batch: PyTree) -> PyTree:
        batch = jax.tree.map(lambda x: jax.random.permutation(key, x), batch)
        minibatches = jax.tree.map(
            lambda x: x.reshape(self.config.num_minibatches, -1, *x.shape[1:]),
            batch,
        )
        return minibatches

    def update(
        self, key: Key, state: PPOState, transitions: Transition, aux: PyTree
    ) -> PPOState:
        log_probs, values = aux
        obs_last = jax.tree.map(lambda x: x[-1], transitions.next_obs)
        value_last = self.critic.apply(state.params["critic"], obs_last)
        advantages = self.advantages(transitions, values, value_last)
        returns = advantages + values
        batch = (transitions, log_probs, values, advantages, returns)
        batch = jax.tree.map(lambda x: x.reshape(-1, *x.shape[2:]), batch)

        def loss(params, minibatch):
            transitions, log_probs_old, values_old, advantages, returns = minibatch
            if self.config.normalize_advantage:
                advantages = (advantages - advantages.mean()) / (
                    advantages.std() + 1e-8
                )

            dist = self.actor.apply(params["actor"], transitions.obs)
            values = self.critic.apply(params["critic"], transitions.obs)
            log_probs = dist.log_prob(transitions.action)
            entropy = dist.entropy().mean()
            ratio = jnp.exp(log_probs - log_probs_old)
            approximate_kl = jnp.mean(log_probs_old - log_probs)
            clip_fraction = jnp.mean(
                (jnp.abs(ratio - 1.0) > self.config.ratio_clip).astype(jnp.float32)
            )

            loss_actor = -jnp.minimum(
                ratio * advantages,
                jnp.clip(
                    ratio,
                    1.0 - self.config.ratio_clip,
                    1.0 + self.config.ratio_clip,
                )
                * advantages,
            ).mean()

            loss_critic = 0.5 * (values - returns) ** 2
            if self.config.value_clip is not None:
                values_clipped = values_old + jnp.clip(
                    values - values_old,
                    -self.config.value_clip,
                    self.config.value_clip,
                )
                loss_critic = jnp.maximum(
                    loss_critic, 0.5 * (values_clipped - returns) ** 2
                )
            loss_critic = loss_critic.mean()

            loss_value = (
                loss_actor
                - self.config.entropy_coefficient * entropy
                + self.config.value_coefficient * loss_critic
            )
            explained_variance = 1 - jnp.var(returns - values) / (
                jnp.var(returns) + 1e-8
            )
            lox.log(
                {
                    "actor/loss": loss_actor,
                    "actor/entropy": entropy,
                    "actor/approximate_kl": approximate_kl,
                    "actor/clip_fraction": clip_fraction,
                    "critic/loss": loss_critic,
                    "critic/explained_variance": explained_variance,
                    "critic/value": values.mean(),
                }
            )
            return loss_value

        def update_step(state, minibatch):
            grads = jax.grad(loss)(state.params, minibatch)
            updates, state.opt_state = self.optimizer.update(grads, state.opt_state)
            state.params = optax.apply_updates(state.params, updates)
            return state, None

        def update_epoch(state, key):
            minibatches = self.minibatches(key, batch)
            state = jax.lax.scan(update_step, state, minibatches)[0]
            return state, None

        keys = jax.random.split(key, self.config.num_update_epochs)
        state, _ = jax.lax.scan(update_epoch, state, keys)
        return state
