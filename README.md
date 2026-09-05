# Reinforcement Learning in JAX

## Algorithms
- PPO
- PQN

## Agents and algorithms
An `Agent` is a learning rule with no environment: `init`, `act`, `update`, and
`initialize_carry` for per-environment state such as a recurrent hidden. An
`Algorithm` owns the environment and feeds an agent. `OnPolicy` rolls out
`num_steps_rollout` steps in `num_envs` environments and calls `agent.update` on the
batch; `PPO` and `PQN` are `OnPolicy` around `PPOAgent` and `PQNAgent`.

Agent wrappers sit between the environment and the learning rule. `Normalize`
standardizes observations with running statistics, frozen per rollout so an update
sees its data the way it was acted upon.
```python
ppo = PPO(config, env, network, optimizer, wrappers=(Normalize,))
# or, explicitly
ppo = OnPolicy(Normalize(PPOAgent(config, network, optimizer)), env, config)
```

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
