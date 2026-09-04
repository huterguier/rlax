import gxm
import jax
import optax
from console_logger import ConsoleLogger

from rlax.algorithms.ppo import PPO, PPOConfig
from rlax.networks import ActorCritic, CategoricalHead
from rlax.wrappers import Trainer

if __name__ == "__main__":
    num_steps = int(5e5)
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
    network = ActorCritic(CategoricalHead(env.action_space.n))
    optimizer = optax.adam(learning_rate=2.5e-4)

    ppo = PPO(config, env, network, optimizer)
    trainer = Trainer(
        ppo,
        num_epochs=10,
        num_steps_eval=1000,
        num_envs_eval=16,
        logger=ConsoleLogger(progress={"step": num_steps}),
    )

    def train(key):
        key_init, key_train = jax.random.split(key)
        ppo_state = trainer.train(
            key_train, trainer.init(key_init), num_steps=num_steps
        )
        return ppo_state

    key = jax.random.key(0)
    keys = jax.random.split(key, 5)
    jax.vmap(train)(keys)
