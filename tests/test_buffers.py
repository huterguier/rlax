import jax
import jax.numpy as jnp
import pytest

from rlax import ItemBuffer, SequenceBuffer


@pytest.fixture
def key():
    return jax.random.key(0)


class TestItemBuffer:
    def test_add_any_leading_dims(self):
        buf = ItemBuffer(capacity=10)
        state = buf.init({"x": jnp.zeros((4, 2)), "t": jnp.zeros((4,), jnp.int32)})
        assert not buf.can_sample(state)
        state = buf.add(state, {"x": jnp.ones((2,)), "t": jnp.int32(0)})
        state = buf.add(state, {"x": jnp.ones((3, 2)), "t": jnp.arange(1, 4)})
        assert int(state.size) == 4 and int(state.index) == 4
        assert buf.can_sample(state)
        state = buf.add(
            state, {"x": jnp.ones((2, 2, 2)), "t": 4 + jnp.arange(4).reshape(2, 2)}
        )
        assert int(state.size) == 8 and int(state.index) == 8
        assert state.data["t"][:8].tolist() == list(range(8))

    def test_wrap_keeps_newest(self):
        buf = ItemBuffer(capacity=10)
        state = buf.init(jnp.zeros((1,), jnp.int32))
        state = buf.add(state, jnp.arange(7))
        state = buf.add(state, jnp.arange(7, 14))
        assert int(state.size) == 10 and int(state.index) == 4
        assert set(state.data.tolist()) == set(range(4, 14))
        # Oversized add keeps only its last ``capacity`` items.
        state = buf.add(state, jnp.arange(100, 125))
        assert int(state.size) == 10 and int(state.index) == 4
        assert set(state.data.tolist()) == set(range(115, 125))

    def test_sample_shape_and_validity(self, key):
        buf = ItemBuffer(capacity=10)
        state = buf.init({"x": jnp.zeros((1, 3)), "t": jnp.zeros((1,), jnp.int32)})
        state = buf.add(state, {"x": jnp.ones((6, 3)), "t": jnp.arange(6)})
        out = jax.jit(buf.sample, static_argnums=2)(key, state, (5, 4))
        assert out["x"].shape == (5, 4, 3) and out["t"].shape == (5, 4)
        assert bool(jnp.all(out["t"] < 6)), "sampled an unwritten slot"
        assert set(out["t"].ravel().tolist()) == set(range(6))

    def test_scan(self, key):
        buf = ItemBuffer(capacity=16)
        state = buf.init(jnp.zeros((1,), jnp.int32))

        def step(state, i):
            state = buf.add(state, i * 4 + jnp.arange(4))
            return state, None

        state, _ = jax.lax.scan(step, state, jnp.arange(8))
        assert int(state.size) == 16
        assert set(state.data.tolist()) == set(range(16, 32))


class TestSequenceBuffer:
    @staticmethod
    def batch(t0, n, lanes=2):
        # Item at global time t in lane l is 100 * l + t, so both are recoverable.
        t = t0 + jnp.arange(n)
        return 100 * jnp.arange(lanes)[None, :] + t[:, None]

    def test_lanes_from_init_batch(self):
        buf = SequenceBuffer(capacity=8, length=3)
        state = buf.init(jnp.zeros((3, 5)))
        assert state.data.shape == (3, 8, 5)

    def test_add_flattens_time(self):
        buf = SequenceBuffer(capacity=8, length=2)
        state = buf.init(jnp.zeros((2,), jnp.int32))
        state = buf.add(state, self.batch(0, 1)[0])  # (lanes,)
        state = buf.add(state, self.batch(1, 6).reshape(2, 3, 2))  # (2, 3, lanes)
        assert int(state.size) == 7
        assert state.data[0, :7].tolist() == list(range(7))
        assert state.data[1, :7].tolist() == [100 + t for t in range(7)]

    def test_windows_valid(self, key):
        buf = SequenceBuffer(capacity=8, length=3)
        state = buf.init(jnp.zeros((2,), jnp.int32))
        state = buf.add(state, self.batch(0, 2))
        assert not buf.can_sample(state)
        state = buf.add(state, self.batch(2, 3))
        assert buf.can_sample(state)
        state = buf.add(state, self.batch(5, 6))  # holds t = 3..10, head wrapped
        out = jax.jit(buf.sample, static_argnums=2)(key, state, (64,))
        assert out.shape == (3, 64)
        t, lane = out % 100, out // 100
        assert bool(jnp.all(lane[0] == lane)), "window spans lanes"
        assert bool(jnp.all(jnp.diff(t, axis=0) == 1)), "window not contiguous"
        assert bool(jnp.all((t >= 3) & (t <= 10))), "stale slot sampled"
        assert set(t[0].tolist()) == set(range(3, 9)), "not every start reachable"
        assert set(lane[0].tolist()) == {0, 1}

    def test_length_bounds(self):
        with pytest.raises(AssertionError):
            SequenceBuffer(capacity=4, length=5)
        with pytest.raises(AssertionError):
            SequenceBuffer(capacity=4, length=0)
