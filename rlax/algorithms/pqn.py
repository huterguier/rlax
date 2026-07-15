from dataclasses import dataclass
from typing import Any

import gxm
import jax
import lox
import optax
import tqdx
from gxm.wrappers import RecordEpisodeStatistics
import jax.numpy as jnp

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
class PQNState(AlgorithmState):
    params: Params
    opt_state: optax.OptState
    env_state: gxm.EnvironmentState
    timestep: gxm.Timestep


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


class PQN(Algorithm[PQNState]):
    config: PQNConfig
    network: Any
    optimizer: Optimizer
    env: Environment

    def __init__(
        self,
        config: PQNConfig,
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

    def init(self, key: Array) -> PQNState:
        key_init, key_network = jax.random.split(key, 2)
        keys_init = jax.random.split(key_init, self.config.num_envs)
        env_state, timestep = jax.vmap(self.env.init)(keys_init)
        params = self.network.init(key_network, timestep.next_obs)
        opt_state = self.optimizer.init(params)

        return PQNState(
            step=jnp.int32(0),
            params=params,
            opt_state=opt_state,
            env_state=env_state,
            timestep=timestep,
        )

    def rollout(self, key: Key, state: PQNState) -> tuple[PQNState, Transition, Array]:

        def pi_epsilon(key, q):
            key_e, key_action = jax.random.split(key)
            epsilon = optax.linear_schedule(
                self.config.epsilon_start,
                self.config.epsilon_end,
                self.config.epsilon_steps,
            )(state.step)
            action_random: Array = self.env.action_space.sample(
                key_action, (self.config.num_envs,)
            )
            action_greedy = jnp.argmax(q, axis=-1)
            action = jnp.where(
                jax.random.uniform(key_e, (self.config.num_envs,)) < epsilon,
                action_random,
                action_greedy,
            )
            return action

        def step(carry, key):
            env_state, timestep, q = carry
            obs = timestep.next_obs
            key_pi, key_step = jax.random.split(key)
            action = pi_epsilon(key_pi, q)
            env_state, timestep = jax.vmap(self.env.step)(
                jax.random.split(key_step, self.config.num_envs),
                env_state,
                action,
            )
            transition = timestep.transition(obs=obs)
            q_next = self.network.apply(state.params, timestep.next_obs)
            return (env_state, timestep, q_next), (transition, q_next)

        keys = jax.random.split(key, self.config.num_steps_rollout)
        env_state = state.env_state
        timestep = state.timestep
        q = self.network.apply(state.params, timestep.next_obs)
        (env_state, timestep, _), (transitions, qs_next) = jax.lax.scan(
            step, (env_state, timestep, q), keys
        )
        state.env_state = env_state
        state.timestep = timestep
        state.step += self.config.num_envs * self.config.num_steps_rollout
        return state, transitions, qs_next

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

    def update(self, key: Key, state: PQNState, minibatches: PyTree) -> PQNState:
        del key

        def loss(params, minibatch):
            batch, targets = minibatch
            qs = self.network.apply(params, batch.obs)
            qs_pred = jnp.take_along_axis(
                qs, batch.action[:, None], axis=-1
            ).squeeze(-1)
            loss_value = jnp.mean((qs_pred - targets) ** 2)
            lox.log(
                {"loss": loss_value, "q_pred": jnp.mean(qs_pred)}
            )
            return loss_value

        def update_step(state, minibatch):
            grads = jax.grad(loss)(state.params, minibatch)
            updates, state.opt_state = self.optimizer.update(grads, state.opt_state)
            state.params = optax.apply_updates(state.params, updates)
            return state, None

        def update_epoch(state_minibatches, _):
            state, minibatches = state_minibatches
            state = jax.lax.scan(update_step, state, minibatches)[0]
            state_minibatches = (state, minibatches)
            return state_minibatches, None

        (state, _), _ = jax.lax.scan(
            update_epoch, (state, minibatches), length=self.config.num_update_epochs
        )

        return state

    def evaluate(
        self, key: Key, state: PQNState, num_steps: int, num_envs: int
    ) -> dict[str, PyTree]:
        def policy(key, policy_state, timestep):
            del key
            state = policy_state
            obs = jax.tree.map(lambda x: x[None], timestep.next_obs)
            q = self.network.apply(state.params, obs)
            action = jnp.argmax(q, axis=-1)[0]
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

    def train(self, key: Key, state: PQNState, num_steps: int) -> PQNState:

        def loop(state, key):
            key_rollout, key_minibatches, key_update = jax.random.split(key, 3)
            state, transitions, qs_next = self.rollout(key_rollout, state)
            minibatches = self.minibatches(key_minibatches, transitions, qs_next)
            state = self.update(key_update, state, minibatches)
            lox.log(
                {"return": jnp.mean(state.timestep.info["episodic_return"])},
            )
            return state, None

        keys = jax.random.split(key, self.config.num_loops(num_steps))
        state, _ = tqdx.scan(loop, state, keys)

        return state
