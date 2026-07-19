import random
from collections import deque
import numpy as np

from network import build_network
from self_play import play_self_play_game
from evaluate import evaluate_vs_opponent


def training_step(model, batch):
    boards, statuses, legals, policy_targets, value_targets = zip(*batch)
    boards = np.stack(boards, axis=0)
    statuses = np.stack(statuses, axis=0)
    legals = np.stack(legals, axis=0)
    policy_targets = np.stack(policy_targets, axis=0)
    value_targets = np.array(value_targets, dtype=np.float32).reshape(-1, 1)

    loss = model.train_on_batch(
        [boards, statuses, legals],
        {"value": value_targets, "policy": policy_targets},
    )
    return loss


def training_loop(num_iterations, games_per_iteration, num_simulations, batch_size,
                   train_steps_per_iteration, replay_maxlen=20000,
                   checkpoint_dir="./checkpoints", eval_games=10):
    import os
    import time

    os.makedirs(checkpoint_dir, exist_ok=True)
    model = build_network()
    replay_buffer = deque(maxlen=replay_maxlen)
    start_time = time.time()

    for iteration in range(num_iterations):
        for _ in range(games_per_iteration):
            examples = play_self_play_game(model, num_simulations)
            replay_buffer.extend(examples)

        losses = []
        for _ in range(train_steps_per_iteration):
            if len(replay_buffer) >= batch_size:
                batch = random.sample(replay_buffer, batch_size)
                loss = training_step(model, batch)
                losses.append(loss[0])

        avg_loss = sum(losses) / len(losses) if losses else None
        elapsed = time.time() - start_time
        print(f"iteration {iteration} — {elapsed:.0f}s elapsed — replay_buffer={len(replay_buffer)} avg_loss={avg_loss}")

        model.save(f"{checkpoint_dir}/model_iter{iteration}.keras")

        vs_random = evaluate_vs_opponent(model, num_simulations, "random", eval_games)
        vs_heuristic = evaluate_vs_opponent(model, num_simulations, "heuristic", eval_games)
        print(f"iteration {iteration} vs random: {vs_random}")
        print(f"iteration {iteration} vs heuristic: {vs_heuristic}")


if __name__ == "__main__":
    # Real training run, scaled up from the validated short config.
    # num_simulations=100 (4x the validation run) gives MCTS meaningfully
    # more search depth for both self-play data quality and evaluation
    # accuracy, at ~4x the per-game cost. num_iterations=15 targets a
    # ~2-2.5 hour total runtime at ~8.5 min/iteration.
    training_loop(
        num_iterations=15,
        games_per_iteration=20,
        num_simulations=100,
        batch_size=64,
        train_steps_per_iteration=100,
        replay_maxlen=50000,
        eval_games=15,
    )
