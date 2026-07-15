# Reinforcement Learning in JAX

## Algorithms
- PPO
- PQN

## Usage
```python
algorithm = Algorithm(config, env, network, optimizer)
alg_state = algorithm.init(key)
alg_state = algorithm.train(key, alg_state, num_steps=int(5e5))
result = algorithm.evaluate(key, alg_state, num_steps=1000, num_envs=16)
```
See `examples/` for full runnable scripts.
