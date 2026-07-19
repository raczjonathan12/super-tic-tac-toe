import numpy as np
from network import build_network
from train import training_step


def test_training_step_reduces_loss_on_fixed_batch():
    model = build_network()
    rng = np.random.default_rng(0)

    def random_batch(n):
        boards = rng.random((n, 9, 3, 3, 2)).astype(np.float32)
        statuses = rng.random((n, 9, 4)).astype(np.float32)
        legals = rng.integers(0, 2, size=(n, 9)).astype(np.float32)
        policies = rng.dirichlet(np.ones(81), size=n).astype(np.float32)
        values = rng.uniform(-1, 1, size=n).astype(np.float32)
        return list(zip(boards, statuses, legals, policies, values))

    batch = random_batch(32)
    losses = []
    for _ in range(20):
        loss = training_step(model, batch)
        losses.append(loss[0])

    assert all(np.isfinite(l) for l in losses)
    assert losses[-1] < losses[0]
