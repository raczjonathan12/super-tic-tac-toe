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
