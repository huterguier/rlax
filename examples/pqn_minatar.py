from rlax.algorithms.pqn import PQN, PQNConfig

import jax
import flax.linen as nn
import optax
import gxm
from gxm.wrappers import StickyAction


class Network(nn.Module):
    n_actions: int = 4

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=16, kernel_size=(3, 3), strides=(1, 1))(x)
        x = nn.relu(x)
        x = nn.Conv(features=32, kernel_size=(3, 3), strides=(1, 1))(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(features=256)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.n_actions)(x)
        return x

if __name__ == "__main__":
    learning_rate = 3e-4
    config = PQNConfig(
        num_envs=128,
        num_steps_rollout=32,
        num_minibatches=4,
        num_update_epochs=2,
        gamma=0.99,
        lam=0.9,
        max_grad_norm=10,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_steps=int(1e5),
    )
    env = StickyAction(gxm.make("Gymnax/Breakout-MinAtar"), stickiness=0.1)
    network = Network(env.action_space.n)
    optimizer = optax.radam(learning_rate)

    pqn = PQN(config, env, network, optimizer)

    key = jax.random.key(0)
    key_init, key_train = jax.random.split(key)
    pqn_state = pqn.train(key_train, pqn.init(key_init), num_steps=int(1e7))
    print(pqn.evaluate(key, pqn_state, num_steps=20000, num_envs=32))
