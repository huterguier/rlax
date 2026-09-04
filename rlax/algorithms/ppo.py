from dataclasses import dataclass
from typing import Any

import gxm
import jax
import jax.numpy as jnp
import lox
import optax
from gxm.wrappers import RecordEpisodeStatistics

from rlax.algorithms.algorithm import Algorithm, AlgorithmState
from rlax.evaluation import evaluate_steps
from rlax.typing import (
    Array,
    Environment,
    Key,
    Optimizer,
    Params,
    PyTree,
    Transition,
)


@jax.tree_util.register_dataclass
@dataclass
class PPOState(AlgorithmState):
    params: Params
    opt_state: optax.OptState
    env_state: gxm.EnvironmentState
    timestep: gxm.Timestep


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


class PPO(Algorithm[PPOState]):
    """Proximal Policy Optimization.

    The network maps a batch of observations to ``(dist, value)``, where ``dist``
    is a distrax distribution with batch shape ``(batch,)`` and ``value`` is an
    array of shape ``(batch,)``.
    """

    config: PPOConfig
    network: Any
    optimizer: Optimizer
    env: Environment

    def __init__(
        self,
        config: PPOConfig,
        env,
        network,
        optimizer,
    ):
        self.config = config
        if not env.has_wrapper(RecordEpisodeStatistics):
            env = RecordEpisodeStatistics(env)
        self.env = env
        self.network = network
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(self.config.max_grad_norm),
            optimizer,
        )

    def init(self, key: Key) -> PPOState:
        key_init, key_network = jax.random.split(key, 2)
        keys_init = jax.random.split(key_init, self.config.num_envs)
        env_state, timestep = jax.vmap(self.env.init)(keys_init)
        params = self.network.init(key_network, timestep.next_obs)
        opt_state = self.optimizer.init(params)

        return PPOState(
            step=jnp.int32(0),
            params=params,
            opt_state=opt_state,
            env_state=env_state,
            timestep=timestep,
        )

    def rollout(
        self, key: Key, state: PPOState
    ) -> tuple[PPOState, Transition, Array, Array, Array]:

        def step(carry, key):
            env_state, timestep = carry
            obs = timestep.next_obs
            key_action, key_step = jax.random.split(key)
            dist, value = self.network.apply(state.params, obs)
            assert value.ndim == 1, (
                f"expected value of shape (batch,), got {value.shape}"
            )
            action, log_prob = dist.sample_and_log_prob(seed=key_action)
            env_state, timestep = jax.vmap(self.env.step)(
                jax.random.split(key_step, self.config.num_envs),
                env_state,
                action,
            )
            transition = timestep.transition(obs=obs)
            return (env_state, timestep), (transition, log_prob, value)

        keys = jax.random.split(key, self.config.num_steps_rollout)
        (env_state, timestep), (transitions, log_probs, values) = jax.lax.scan(
            step, (state.env_state, state.timestep), keys
        )
        _, value_last = self.network.apply(state.params, timestep.next_obs)
        state.env_state = env_state
        state.timestep = timestep
        state.step += self.config.num_steps_loop
        return state, transitions, log_probs, values, value_last

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

    def update(self, key: Key, state: PPOState, batch: PyTree) -> PPOState:
        batch = jax.tree.map(lambda x: x.reshape(-1, *x.shape[2:]), batch)

        def loss(params, minibatch):
            transitions, log_probs_old, values_old, advantages, returns = minibatch
            if self.config.normalize_advantage:
                advantages = (advantages - advantages.mean()) / (
                    advantages.std() + 1e-8
                )

            dist, values = self.network.apply(params, transitions.obs)
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

    def evaluate(
        self, key: Key, state: PPOState, num_steps: int, num_envs: int
    ) -> dict[str, PyTree]:
        def policy(key, policy_state, timestep):
            del key
            state = policy_state
            obs = jax.tree.map(lambda x: x[None], timestep.next_obs)
            dist, _ = self.network.apply(state.params, obs)
            action = jax.tree.map(lambda a: a[0], dist.mode())
            return action, policy_state

        episodic_return = evaluate_steps(
            key,
            self.env,
            policy,
            state,
            num_steps,
            num_envs,
        )

        return {"return": episodic_return}

    def train(self, key: Key, state: PPOState, num_steps: int) -> PPOState:
        num_loops = self.config.num_loops(num_steps)
        assert num_loops >= 1, (
            f"num_steps ({num_steps}) is smaller than one PPO update "
            f"({self.config.num_steps_loop} = num_envs {self.config.num_envs} * "
            f"num_steps_rollout {self.config.num_steps_rollout})"
        )

        def loop(state, key):
            key_rollout, key_update = jax.random.split(key)
            state, transitions, log_probs, values, value_last = self.rollout(
                key_rollout, state
            )
            advantages = self.advantages(transitions, values, value_last)
            returns = advantages + values
            batch = (transitions, log_probs, values, advantages, returns)
            state = self.update(key_update, state, batch)
            lox.log(
                {"return": jnp.mean(state.timestep.info["episodic_return"])},
            )
            return state, None

        keys = jax.random.split(key, num_loops)
        state, _ = jax.lax.scan(loop, state, keys)

        return state
