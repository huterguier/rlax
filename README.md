# Reinforcement Learning in JAX

## Algorithms
- PPO
- PQN

## Networks
`rlax.networks` has the stock pieces: an `ActorCritic` module built from `MLP` torsos and a
policy head, `CategoricalHead` for discrete actions, `GaussianHead` for continuous ones.
`GaussianHead(squash=True)` returns a Gaussian over pre-tanh actions with a tanh-aware
entropy; use it with an environment wrapped in `gxm.wrappers.SquashActions`.
```python
network = ActorCritic(GaussianHead(action_size, squash=True), policy_layers=(128, 128))
```
Any module returning `(dist, value)` works in place of it; `rlax.networks.ActorCriticNetwork`
spells out the contract.

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

### Logging
`Trainer` takes a `lox.Logger`, which receives the algorithm's own logs at full
resolution along with the `eval/` metrics, once per epoch.
```python
trainer = Trainer(algorithm, num_epochs=10, logger=WandbLogger(project="rlax"))
```
The logger defaults to an empty `MultiLogger`, so logging to one changes where
metrics go but not what `spool` returns. The algorithm's logs are spooled per epoch
and come back with an epoch axis, e.g. `logs["return"]` has shape
`(num_epochs, per_epoch)`; `reshape(-1)` flattens it.
