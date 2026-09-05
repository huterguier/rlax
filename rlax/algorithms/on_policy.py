from dataclasses import dataclass
from typing import Protocol

import gxm
import jax
import jax.numpy as jnp
import lox
from gxm.wrappers import RecordEpisodeStatistics

from rlax.agent import Agent
from rlax.algorithms.algorithm import Algorithm, AlgorithmState
from rlax.evaluation import evaluate_steps
from rlax.typing import Environment, Key, PyTree, Transition


class OnPolicyConfig(Protocol):
    num_envs: int
    num_steps_rollout: int


@jax.tree_util.register_dataclass
@dataclass
class OnPolicyState(AlgorithmState):
    agent_state: PyTree
    carry: PyTree
    env_state: gxm.EnvironmentState
    timestep: gxm.Timestep


class OnPolicy(Algorithm[OnPolicyState]):
    """Roll out ``num_steps_rollout`` steps in ``num_envs`` environments, then
    hand the batch to ``agent.update``. The only place environments are
    stepped."""

    agent: Agent
    env: Environment
    config: OnPolicyConfig

    def __init__(self, agent: Agent, env: Environment, config: OnPolicyConfig):
        if not env.has_wrapper(RecordEpisodeStatistics):
            env = RecordEpisodeStatistics(env)
        self.agent = agent
        self.env = env
        self.config = config

    @property
    def num_steps_loop(self) -> int:
        return self.config.num_envs * self.config.num_steps_rollout

    def num_loops(self, num_steps: int) -> int:
        return int(num_steps // self.num_steps_loop)

    def init(self, key: Key) -> OnPolicyState:
        key_env, key_agent, key_carry = jax.random.split(key, 3)
        keys_env = jax.random.split(key_env, self.config.num_envs)
        env_state, timestep = jax.vmap(self.env.init)(keys_env)
        agent_state = self.agent.init(key_agent, timestep)
        carry = self.agent.initialize_carry(key_carry, agent_state, timestep)
        return OnPolicyState(
            step=jnp.int32(0),
            agent_state=agent_state,
            carry=carry,
            env_state=env_state,
            timestep=timestep,
        )

    def rollout(
        self, key: Key, state: OnPolicyState
    ) -> tuple[OnPolicyState, Transition, PyTree]:
        def step(carry, key):
            agent_state, agent_carry, env_state, timestep = carry
            obs = timestep.next_obs
            key_act, key_step = jax.random.split(key)
            action, agent_state, agent_carry, aux = self.agent.act(
                key_act, agent_state, agent_carry, timestep
            )
            env_state, timestep = jax.vmap(self.env.step)(
                jax.random.split(key_step, self.config.num_envs), env_state, action
            )
            carry = (agent_state, agent_carry, env_state, timestep)
            return carry, (timestep.transition(obs=obs), aux)

        keys = jax.random.split(key, self.config.num_steps_rollout)
        carry = (state.agent_state, state.carry, state.env_state, state.timestep)
        carry, (transitions, aux) = jax.lax.scan(step, carry, keys)
        state.agent_state, state.carry, state.env_state, state.timestep = carry
        state.step += self.num_steps_loop
        return state, transitions, aux

    def train(self, key: Key, state: OnPolicyState, num_steps: int) -> OnPolicyState:
        num_loops = self.num_loops(num_steps)
        assert num_loops >= 1, (
            f"num_steps ({num_steps}) is smaller than one rollout "
            f"({self.num_steps_loop} = num_envs {self.config.num_envs} * "
            f"num_steps_rollout {self.config.num_steps_rollout})"
        )

        def loop(state, key):
            key_rollout, key_update = jax.random.split(key)
            state, transitions, aux = self.rollout(key_rollout, state)
            state.agent_state = self.agent.update(
                key_update, state.agent_state, transitions, aux
            )
            lox.log({"return": jnp.mean(state.timestep.info["episodic_return"])})
            return state, None

        keys = jax.random.split(key, num_loops)
        state, _ = jax.lax.scan(loop, state, keys)
        return state

    def evaluate(
        self, key: Key, state: OnPolicyState, num_steps: int, num_envs: int
    ) -> dict[str, PyTree]:
        # evaluate_steps runs one unbatched environment per vmap lane, while
        # agents expect a leading batch axis; add and strip a batch of one.
        def policy(key, policy_state, timestep):
            agent_state, carry = policy_state
            batched = jax.tree.map(lambda x: x[None], timestep)
            action, agent_state, carry, _ = self.agent.act(
                key, agent_state, carry, batched, evaluation=True
            )
            return jax.tree.map(lambda a: a[0], action), (agent_state, carry)

        key_carry, key_eval = jax.random.split(key)
        sample = jax.tree.map(lambda x: x[:1], state.timestep)
        carry = self.agent.initialize_carry(key_carry, state.agent_state, sample)
        episodic_return = evaluate_steps(
            key_eval, self.env, policy, (state.agent_state, carry), num_steps, num_envs
        )
        return {"return": episodic_return}
