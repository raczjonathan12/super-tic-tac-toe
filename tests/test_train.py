import numpy as np
from network import build_network
from train import training_step, training_loop


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


def test_training_loop_writes_log_file(tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"
    log_path = tmp_path / "training_log.txt"

    training_loop(
        num_iterations=1,
        games_per_iteration=1,
        num_simulations=5,
        batch_size=4,
        train_steps_per_iteration=2,
        checkpoint_dir=str(checkpoint_dir),
        eval_games=1,
        log_path=str(log_path),
    )

    assert log_path.exists()
    content = log_path.read_text()
    assert "iteration 0" in content
    assert "vs random" in content
    assert "vs heuristic" in content
    assert (checkpoint_dir / "model_iter0.keras").exists()


def test_training_loop_can_resume_from_a_checkpoint_without_overwriting_it(tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"
    log_path = tmp_path / "training_log.txt"

    training_loop(
        num_iterations=1,
        games_per_iteration=1,
        num_simulations=5,
        batch_size=4,
        train_steps_per_iteration=2,
        checkpoint_dir=str(checkpoint_dir),
        eval_games=1,
        log_path=str(log_path),
    )
    first_run_checkpoint = checkpoint_dir / "model_iter0.keras"
    assert first_run_checkpoint.exists()
    original_mtime = first_run_checkpoint.stat().st_mtime

    training_loop(
        num_iterations=1,
        games_per_iteration=1,
        num_simulations=5,
        batch_size=4,
        train_steps_per_iteration=2,
        checkpoint_dir=str(checkpoint_dir),
        eval_games=1,
        log_path=str(log_path),
        resume_from=str(first_run_checkpoint),
        start_iteration=1,
    )

    # the resumed run must not have overwritten the checkpoint it resumed from
    assert first_run_checkpoint.stat().st_mtime == original_mtime
    assert (checkpoint_dir / "model_iter1.keras").exists()

    content = log_path.read_text()
    assert "iteration 1" in content
