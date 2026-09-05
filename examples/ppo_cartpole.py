import gxm
import jax
import optax
from console_logger import ConsoleLogger

from rlax.agents import Normalize
from rlax.algorithms.ppo import PPO, PPOConfig
from rlax.networks import Actor, CategoricalHead, Critic
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
    actor = Actor(CategoricalHead(env.action_space.n))
    critic = Critic()
    optimizer = optax.adam(learning_rate=2.5e-4)

    # Running observation normalization, as an agent wrapper.
    ppo = PPO(config, env, actor, critic, optimizer, wrappers=(Normalize,))
    trainer = Trainer(
        ppo,
        num_epochs=10,
        num_steps_eval=1000,
        num_envs_eval=16,
        logger=ConsoleLogger(progress={"step": num_steps}),
    )

    key = jax.random.key(0)
    ppo_state = trainer.init(key)
    ppo_state = trainer.train(key, ppo_state, num_steps=num_steps)
