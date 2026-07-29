import distrax
import flax.linen as nn
import gxm
import jax
import optax

from rlax.algorithms.ppo import PPO, PPOConfig


class ActorCritic(nn.Module):
    n_actions: int

    @nn.compact
    def __call__(self, x):
        actor = nn.Dense(features=64)(x)
        actor = nn.tanh(actor)
        actor = nn.Dense(features=64)(actor)
        actor = nn.tanh(actor)
        logits = nn.Dense(features=self.n_actions)(actor)

        critic = nn.Dense(features=64)(x)
        critic = nn.tanh(critic)
        critic = nn.Dense(features=64)(critic)
        critic = nn.tanh(critic)
        value = nn.Dense(features=1)(critic)

        return distrax.Categorical(logits=logits), value.squeeze(-1)


if __name__ == "__main__":
    learning_rate = 2.5e-4
    config = PPOConfig(
        num_envs=8,
        num_steps_rollout=128,
        num_minibatches=4,
        num_update_epochs=4,
        gamma=0.99,
        lam=0.95,
        ratio_clip=0.2,
        value_coefficient=0.5,
        entropy_coefficient=0.01,
        normalize_advantage=True,
        max_grad_norm=0.5,
        value_clip=0.2,
    )
    env = gxm.make("Gymnax/CartPole-v1")
    network = ActorCritic(env.action_space.n)
    optimizer = optax.adam(learning_rate)

    ppo = PPO(config, env, network, optimizer)

    key = jax.random.key(0)
    key_init, key_train = jax.random.split(key)
    ppo_state = ppo.train(key_train, ppo.init(key_init), num_steps=int(5e5))
    print(ppo.evaluate(key, ppo_state, num_steps=1000, num_envs=16))
