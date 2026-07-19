import random
from game import Game, random_legal_action, heuristic_action
from mcts import run_mcts, select_action


def evaluate_vs_opponent(model, num_simulations, opponent, num_games):
    results = {"win": 0, "loss": 0, "draw": 0, "incomplete": 0}

    for _ in range(num_games):
        game = Game()
        agent_player = random.randint(1, 2)
        move_count = 0

        while not game.game_over and move_count < 100:
            if game.current_player == agent_player:
                root = run_mcts(game, model, num_simulations, add_noise=False)
                action = select_action(root, temperature=1e-3)
                sub_board, cell = action // 9, action % 9
            elif opponent == "random":
                sub_board, cell = random_legal_action(game)
            elif opponent == "heuristic":
                sub_board, cell = heuristic_action(game)
            else:
                root = run_mcts(game, opponent, num_simulations, add_noise=False)
                action = select_action(root, temperature=1e-3)
                sub_board, cell = action // 9, action % 9

            game.execute_move(sub_board, cell)
            move_count += 1

        if game.game_over:
            if game.winner == agent_player:
                results["win"] += 1
            elif game.winner is None:
                results["draw"] += 1
            else:
                results["loss"] += 1
        else:
            results["incomplete"] += 1

    return results
