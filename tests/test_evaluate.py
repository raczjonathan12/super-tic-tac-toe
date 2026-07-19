import time
from network import build_network
from evaluate import evaluate_vs_opponent


def test_evaluate_vs_random_runs_and_returns_valid_results():
    model = build_network()
    results = evaluate_vs_opponent(model, num_simulations=5, opponent="random", num_games=2)
    assert set(results.keys()) == {"win", "loss", "draw", "incomplete"}
    assert sum(results.values()) == 2


def test_evaluate_vs_heuristic_runs_and_returns_valid_results():
    model = build_network()
    results = evaluate_vs_opponent(model, num_simulations=5, opponent="heuristic", num_games=2)
    assert set(results.keys()) == {"win", "loss", "draw", "incomplete"}
    assert sum(results.values()) == 2


def test_evaluate_vs_another_model_runs_and_returns_valid_results():
    model_a = build_network()
    model_b = build_network()
    results = evaluate_vs_opponent(model_a, num_simulations=5, opponent=model_b, num_games=2)
    assert set(results.keys()) == {"win", "loss", "draw", "incomplete"}
    assert sum(results.values()) == 2


def test_evaluate_batching_is_faster_than_one_game_at_a_time():
    model = build_network()

    start = time.time()
    evaluate_vs_opponent(model, num_simulations=15, opponent="heuristic", num_games=4)
    batched_time = time.time() - start

    start = time.time()
    for _ in range(4):
        evaluate_vs_opponent(model, num_simulations=15, opponent="heuristic", num_games=1)
    sequential_time = time.time() - start

    assert batched_time < sequential_time
