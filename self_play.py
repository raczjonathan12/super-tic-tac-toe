from game import Game
from network import encode_state
from mcts import run_mcts, run_mcts_batch, get_policy_target, select_action


def play_self_play_game(model, num_simulations, c_puct=1.5, temperature_moves=10,
                         dirichlet_alpha=0.3, dirichlet_epsilon=0.25, max_moves=100):
    game = Game()
    history = []
    move_count = 0

    while not game.game_over and move_count < max_moves:
        root = run_mcts(
            game, model, num_simulations, c_puct=c_puct,
            dirichlet_alpha=dirichlet_alpha, dirichlet_epsilon=dirichlet_epsilon,
            add_noise=True,
        )
        policy_target = get_policy_target(root)
        temperature = 1.0 if move_count < temperature_moves else 1e-3
        action = select_action(root, temperature)

        board, status, legal = encode_state(game, game.current_player)
        history.append([board, status, legal, policy_target, game.current_player])

        sub_board, cell = action // 9, action % 9
        game.execute_move(sub_board, cell)
        move_count += 1

    examples = []
    for board, status, legal, policy_target, mover in history:
        if game.winner is None:
            value_target = 0.0
        elif game.winner == mover:
            value_target = 1.0
        else:
            value_target = -1.0
        examples.append((board, status, legal, policy_target, value_target))

    return examples


def play_self_play_batch(model, num_games, num_simulations, c_puct=1.5, temperature_moves=10,
                          dirichlet_alpha=0.3, dirichlet_epsilon=0.25, max_moves=100):
    """Plays num_games self-play games simultaneously, batching each move's
    MCTS network calls across all still-active games (via run_mcts_batch)
    instead of finishing one game fully before starting the next. Returns
    the combined list of training examples from all games."""
    games = [Game() for _ in range(num_games)]
    histories = [[] for _ in range(num_games)]
    move_counts = [0] * num_games
    active = list(range(num_games))

    while active:
        active_games = [games[i] for i in active]
        roots = run_mcts_batch(
            active_games, model, num_simulations, c_puct=c_puct,
            dirichlet_alpha=dirichlet_alpha, dirichlet_epsilon=dirichlet_epsilon,
            add_noise=True,
        )

        finished = []
        for root, i in zip(roots, active):
            game = games[i]
            policy_target = get_policy_target(root)
            temperature = 1.0 if move_counts[i] < temperature_moves else 1e-3
            action = select_action(root, temperature)

            board, status, legal = encode_state(game, game.current_player)
            histories[i].append([board, status, legal, policy_target, game.current_player])

            sub_board, cell = action // 9, action % 9
            game.execute_move(sub_board, cell)
            move_counts[i] += 1

            if game.game_over or move_counts[i] >= max_moves:
                finished.append(i)

        active = [i for i in active if i not in finished]

    examples = []
    for i in range(num_games):
        game = games[i]
        for board, status, legal, policy_target, mover in histories[i]:
            if game.winner is None:
                value_target = 0.0
            elif game.winner == mover:
                value_target = 1.0
            else:
                value_target = -1.0
            examples.append((board, status, legal, policy_target, value_target))

    return examples
