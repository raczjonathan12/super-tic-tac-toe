import random
from collections import deque
import numpy as np

from network import build_network
from self_play import play_self_play_batch
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
                   checkpoint_dir="./checkpoints", eval_games=10,
                   log_path="./training_log.txt", self_play_chunk_size=5,
                   log_every_train_steps=25, resume_from=None, start_iteration=0):
    import os
    import time
    from tensorflow import keras

    os.makedirs(checkpoint_dir, exist_ok=True)
    log_file = open(log_path, "a")

    def log(message):
        print(message)
        log_file.write(message + "\n")
        log_file.flush()

    if resume_from is not None:
        model = keras.models.load_model(resume_from)
        log(f"resuming from {resume_from} at iteration {start_iteration}")
    else:
        model = build_network()
    replay_buffer = deque(maxlen=replay_maxlen)
    start_time = time.time()

    for offset in range(num_iterations):
        iteration = start_iteration + offset
        log(f"iteration {iteration} — {time.time() - start_time:.0f}s elapsed — starting self-play ({games_per_iteration} games)")

        games_played = 0
        while games_played < games_per_iteration:
            chunk = min(self_play_chunk_size, games_per_iteration - games_played)
            examples = play_self_play_batch(model, chunk, num_simulations)
            replay_buffer.extend(examples)
            games_played += chunk
            log(f"iteration {iteration} — {time.time() - start_time:.0f}s elapsed — self-play {games_played}/{games_per_iteration} games, replay_buffer={len(replay_buffer)}")

        losses = []
        for step in range(train_steps_per_iteration):
            if len(replay_buffer) >= batch_size:
                batch = random.sample(replay_buffer, batch_size)
                loss = training_step(model, batch)
                losses.append(loss[0])
                if (step + 1) % log_every_train_steps == 0:
                    running_avg = sum(losses) / len(losses)
                    log(f"iteration {iteration} — {time.time() - start_time:.0f}s elapsed — train step {step + 1}/{train_steps_per_iteration}, avg_loss_so_far={running_avg}")

        avg_loss = sum(losses) / len(losses) if losses else None
        elapsed = time.time() - start_time
        log(f"iteration {iteration} — {elapsed:.0f}s elapsed — replay_buffer={len(replay_buffer)} avg_loss={avg_loss}")

        model.save(f"{checkpoint_dir}/model_iter{iteration}.keras")
        log(f"iteration {iteration} — {time.time() - start_time:.0f}s elapsed — checkpoint saved, starting evaluation")

        vs_random = evaluate_vs_opponent(model, num_simulations, "random", eval_games)
        log(f"iteration {iteration} — {time.time() - start_time:.0f}s elapsed — vs random: {vs_random}")

        vs_heuristic = evaluate_vs_opponent(model, num_simulations, "heuristic", eval_games)
        log(f"iteration {iteration} — {time.time() - start_time:.0f}s elapsed — vs heuristic: {vs_heuristic}")

    log_file.close()


if __name__ == "__main__":
    import os

    # Resumes from final_model.keras (the last run's final checkpoint) if it
    # exists, continuing the iteration count from where it left off, instead
    # of starting a fresh network from scratch.
    _resume_path = "./final_model.keras"
    _resume_from = _resume_path if os.path.exists(_resume_path) else None
    _start_iteration = 15 if _resume_from else 0

    training_loop(
        num_iterations=15,
        games_per_iteration=20,
        num_simulations=100,
        batch_size=64,
        train_steps_per_iteration=100,
        replay_maxlen=50000,
        eval_games=15,
        resume_from=_resume_from,
        start_iteration=_start_iteration,
    )
