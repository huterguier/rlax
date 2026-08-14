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

### Epochs
`Trainer` splits a run into equally sized epochs and evaluates after each one.
```python
trainer = Trainer(algorithm, num_epochs=10)
alg_state = trainer.train(key, alg_state, num_steps=int(1e7))  # 10 epochs of 1e6 steps
```
The evaluation metrics are logged under an `eval/` prefix, so the learning curve is
obtained by spooling the run.
```python
alg_state, logs = lox.spool(trainer.train)(key, alg_state, num_steps=int(1e7))
logs["eval/return"]  # shape (10,)
```
