import numpy as np
from network import build_network
from train import training_step, training_loop, save_replay_buffer, load_replay_buffer, _next_start_iteration


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


def test_save_and_load_replay_buffer_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    original = [
        (
            rng.random((9, 3, 3, 2)).astype(np.float32),
            rng.random((9, 4)).astype(np.float32),
            rng.integers(0, 2, size=(9,)).astype(np.float32),
            rng.dirichlet(np.ones(81)).astype(np.float32),
            float(rng.uniform(-1, 1)),
        )
        for _ in range(10)
    ]
    path = tmp_path / "replay_buffer.pkl"

    save_replay_buffer(original, str(path))
    loaded = load_replay_buffer(str(path), maxlen=100)

    assert len(loaded) == len(original)
    for (b1, s1, l1, p1, v1), (b2, s2, l2, p2, v2) in zip(original, loaded):
        assert np.array_equal(b1, b2)
        assert np.array_equal(s1, s2)
        assert np.array_equal(l1, l2)
        assert np.array_equal(p1, p2)
        assert v1 == v2


def test_training_loop_resume_preserves_replay_buffer_contents(tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"
    log_path = tmp_path / "training_log.txt"
    replay_buffer_path = tmp_path / "replay_buffer.pkl"

    training_loop(
        num_iterations=1,
        games_per_iteration=1,
        num_simulations=5,
        batch_size=4,
        train_steps_per_iteration=2,
        checkpoint_dir=str(checkpoint_dir),
        eval_games=1,
        log_path=str(log_path),
        replay_buffer_path=str(replay_buffer_path),
    )
    first_run_checkpoint = checkpoint_dir / "model_iter0.keras"
    assert replay_buffer_path.exists()
    buffer_after_first_run = load_replay_buffer(str(replay_buffer_path), maxlen=100000)
    size_after_first_run = len(buffer_after_first_run)
    assert size_after_first_run > 0

    training_loop(
        num_iterations=1,
        games_per_iteration=1,
        num_simulations=5,
        batch_size=4,
        train_steps_per_iteration=2,
        checkpoint_dir=str(checkpoint_dir),
        eval_games=1,
        log_path=str(log_path),
        replay_buffer_path=str(replay_buffer_path),
        resume_from=str(first_run_checkpoint),
        start_iteration=1,
    )

    content = log_path.read_text()
    assert f"resumed replay buffer from {replay_buffer_path}" in content
    # the resumed run's buffer must have started from what the first run left
    # behind (plus whatever the resumed run's own self-play added), not 0
    buffer_after_resume = load_replay_buffer(str(replay_buffer_path), maxlen=100000)
    assert len(buffer_after_resume) > size_after_first_run


def test_training_loop_updates_latest_model_path_each_iteration_for_chained_resumes(tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"
    log_path = tmp_path / "training_log.txt"
    latest_model_path = tmp_path / "latest.keras"

    training_loop(
        num_iterations=1,
        games_per_iteration=1,
        num_simulations=5,
        batch_size=4,
        train_steps_per_iteration=2,
        checkpoint_dir=str(checkpoint_dir),
        eval_games=1,
        log_path=str(log_path),
        latest_model_path=str(latest_model_path),
    )
    assert latest_model_path.exists()
    first_mtime = latest_model_path.stat().st_mtime

    # a second run resuming from latest_model_path, exactly like a chained
    # overnight invocation would, must succeed and update it again
    training_loop(
        num_iterations=1,
        games_per_iteration=1,
        num_simulations=5,
        batch_size=4,
        train_steps_per_iteration=2,
        checkpoint_dir=str(checkpoint_dir),
        eval_games=1,
        log_path=str(log_path),
        latest_model_path=str(latest_model_path),
        resume_from=str(latest_model_path),
        start_iteration=1,
    )
    assert latest_model_path.stat().st_mtime >= first_mtime
    assert (checkpoint_dir / "model_iter1.keras").exists()


def test_next_start_iteration_finds_highest_existing_checkpoint(tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()

    assert _next_start_iteration(str(checkpoint_dir)) == 0

    for n in [0, 1, 2, 5, 14]:
        (checkpoint_dir / f"model_iter{n}.keras").write_bytes(b"")
    (checkpoint_dir / "replay_buffer.pkl").write_bytes(b"")  # should be ignored

    assert _next_start_iteration(str(checkpoint_dir)) == 15


def test_next_start_iteration_returns_zero_for_missing_directory(tmp_path):
    missing = tmp_path / "does_not_exist"
    assert _next_start_iteration(str(missing)) == 0
