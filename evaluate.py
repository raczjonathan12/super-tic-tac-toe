import random
from game import Game, random_legal_action, heuristic_action
from mcts import run_mcts_batch, select_action


def evaluate_vs_opponent(model, num_simulations, opponent, num_games):
    """Plays num_games games between model (agent) and opponent simultaneously,
    batching every MCTS-driven side's network calls across all active games
    each round (via run_mcts_batch) instead of finishing one game at a time.
    opponent: 'random', 'heuristic', or a keras model to play greedily via MCTS."""
    games = [Game() for _ in range(num_games)]
    agent_players = [random.randint(1, 2) for _ in range(num_games)]
    move_counts = [0] * num_games
    active = list(range(num_games))

    while active:
        agent_turn = [i for i in active if games[i].current_player == agent_players[i]]
        opponent_turn = [i for i in active if games[i].current_player != agent_players[i]]

        actions = {}

        if agent_turn:
            agent_games = [games[i] for i in agent_turn]
            roots = run_mcts_batch(agent_games, model, num_simulations, add_noise=False)
            for i, root in zip(agent_turn, roots):
                actions[i] = select_action(root, temperature=1e-3)

        if opponent_turn:
            if opponent == "random":
                for i in opponent_turn:
                    sub_board, cell = random_legal_action(games[i])
                    actions[i] = sub_board * 9 + cell
            elif opponent == "heuristic":
                for i in opponent_turn:
                    sub_board, cell = heuristic_action(games[i])
                    actions[i] = sub_board * 9 + cell
            else:
                opponent_games = [games[i] for i in opponent_turn]
                roots = run_mcts_batch(opponent_games, opponent, num_simulations, add_noise=False)
                for i, root in zip(opponent_turn, roots):
                    actions[i] = select_action(root, temperature=1e-3)

        finished = []
        for i in active:
            action = actions[i]
            sub_board, cell = action // 9, action % 9
            games[i].execute_move(sub_board, cell)
            move_counts[i] += 1
            if games[i].game_over or move_counts[i] >= 100:
                finished.append(i)

        active = [i for i in active if i not in finished]

    results = {"win": 0, "loss": 0, "draw": 0, "incomplete": 0}
    for i in range(num_games):
        game = games[i]
        if game.game_over:
            if game.winner == agent_players[i]:
                results["win"] += 1
            elif game.winner is None:
                results["draw"] += 1
            else:
                results["loss"] += 1
        else:
            results["incomplete"] += 1

    return results
