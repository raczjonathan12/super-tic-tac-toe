import math
import numpy as np
from game import legal_action_mask
from network import encode_state


class MCTSNode:
    __slots__ = ("prior", "visit_count", "value_sum", "children")

    def __init__(self, prior):
        self.prior = prior
        self.visit_count = 0
        self.value_sum = 0.0
        self.children = {}

    def is_expanded(self):
        return len(self.children) > 0

    def value(self):
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


def batch_evaluate_and_expand(nodes, games, model):
    """Runs the network once for a batch of (node, game) pairs, expanding
    each node's children with legality-renormalized policy priors. Returns
    a list of values, one per node, in the same order."""
    boards, statuses, legals = [], [], []
    for game in games:
        b, s, l = encode_state(game, game.current_player)
        boards.append(b)
        statuses.append(s)
        legals.append(l)
    boards = np.stack(boards, axis=0)
    statuses = np.stack(statuses, axis=0)
    legals = np.stack(legals, axis=0)

    value_out, policy_out = model([boards, statuses, legals])
    values = value_out.numpy()[:, 0]
    policies = policy_out.numpy()

    results = []
    for node, game, value, policy in zip(nodes, games, values, policies):
        mask = legal_action_mask(game)
        legal_actions = np.nonzero(mask)[0]
        legal_probs = policy[legal_actions]
        total = legal_probs.sum()
        if total > 1e-8:
            legal_probs = legal_probs / total
        else:
            legal_probs = np.full(len(legal_actions), 1.0 / len(legal_actions))

        for action, prob in zip(legal_actions, legal_probs):
            node.children[int(action)] = MCTSNode(prior=float(prob))

        results.append(float(value))

    return results


def evaluate_and_expand(node, game, model):
    return batch_evaluate_and_expand([node], [game], model)[0]


def select_child(node, c_puct):
    best_score = -float("inf")
    best_action = None
    best_child = None
    for action, child in node.children.items():
        q = -child.value()
        u = c_puct * child.prior * math.sqrt(node.visit_count) / (1 + child.visit_count)
        score = q + u
        if score > best_score:
            best_score = score
            best_action = action
            best_child = child
    return best_action, best_child


def _add_root_noise(root, dirichlet_alpha, dirichlet_epsilon):
    if len(root.children) == 0:
        return
    actions = list(root.children.keys())
    noise = np.random.dirichlet([dirichlet_alpha] * len(actions))
    for action, n in zip(actions, noise):
        child = root.children[action]
        child.prior = child.prior * (1 - dirichlet_epsilon) + n * dirichlet_epsilon


def _terminal_or_none(sim_game):
    """Returns the terminal value if sim_game is over, else None. See the
    comment in run_mcts for why this is from the loser's perspective."""
    if not sim_game.game_over:
        return None
    return -1.0 if sim_game.winner is not None else 0.0


def run_mcts(game, model, num_simulations, c_puct=1.5, dirichlet_alpha=0.3,
             dirichlet_epsilon=0.25, add_noise=False):
    root = MCTSNode(prior=1.0)
    evaluate_and_expand(root, game, model)

    if add_noise:
        _add_root_noise(root, dirichlet_alpha, dirichlet_epsilon)

    for _ in range(num_simulations):
        node = root
        sim_game = game.clone()
        path = [node]

        while node.is_expanded() and not sim_game.game_over:
            action, node = select_child(node, c_puct)
            sub_board, cell = action // 9, action % 9
            sim_game.execute_move(sub_board, cell)
            path.append(node)

        # execute_move does not flip current_player on a game-ending move, so
        # a terminal value must be from the perspective of whoever's turn it
        # would conceptually be (the loser), to stay consistent with the
        # non-terminal case and the alternating-sign backup.
        terminal_value = _terminal_or_none(sim_game)
        if terminal_value is not None:
            value = terminal_value
        else:
            value = evaluate_and_expand(node, sim_game, model)

        for path_node in reversed(path):
            path_node.value_sum += value
            path_node.visit_count += 1
            value = -value

    return root


def get_policy_target(root):
    target = np.zeros(81, dtype=np.float32)
    total_visits = sum(child.visit_count for child in root.children.values())
    if total_visits == 0:
        for action, child in root.children.items():
            target[action] = child.prior
    else:
        for action, child in root.children.items():
            target[action] = child.visit_count / total_visits
    return target


def select_action(root, temperature):
    actions = list(root.children.keys())
    visit_counts = np.array([root.children[a].visit_count for a in actions], dtype=np.float64)
    if temperature <= 1e-3:
        best_idx = int(np.argmax(visit_counts))
        return actions[best_idx]
    powered = visit_counts ** (1.0 / temperature)
    probs = powered / powered.sum()
    return int(np.random.choice(actions, p=probs))


def run_mcts_batch(games, model, num_simulations, c_puct=1.5, dirichlet_alpha=0.3,
                    dirichlet_epsilon=0.25, add_noise=False):
    """Runs MCTS for a batch of independent games simultaneously. Within each
    of the num_simulations rounds, every game's tree descends to a leaf
    first, then all leaves needing a network evaluation that round are
    batched into a single model call — this is what makes it fast (one
    model call per round instead of one per game per round). Semantics
    (PUCT selection, terminal-value convention, backup) are identical to
    run_mcts; this only changes how leaf evaluations are dispatched.
    Returns a list of root MCTSNodes, one per game, in the same order."""
    n = len(games)
    roots = [MCTSNode(prior=1.0) for _ in range(n)]
    batch_evaluate_and_expand(roots, games, model)

    if add_noise:
        for root in roots:
            _add_root_noise(root, dirichlet_alpha, dirichlet_epsilon)

    for _ in range(num_simulations):
        paths = []
        pending_values = [None] * n
        leaf_nodes = []
        leaf_games = []
        leaf_indices = []

        for i in range(n):
            node = roots[i]
            sim_game = games[i].clone()
            path = [node]

            while node.is_expanded() and not sim_game.game_over:
                action, node = select_child(node, c_puct)
                sub_board, cell = action // 9, action % 9
                sim_game.execute_move(sub_board, cell)
                path.append(node)

            paths.append(path)
            terminal_value = _terminal_or_none(sim_game)
            if terminal_value is not None:
                pending_values[i] = terminal_value
            else:
                leaf_nodes.append(node)
                leaf_games.append(sim_game)
                leaf_indices.append(i)

        if leaf_nodes:
            values = batch_evaluate_and_expand(leaf_nodes, leaf_games, model)
            for idx, v in zip(leaf_indices, values):
                pending_values[idx] = v

        for i in range(n):
            value = pending_values[i]
            for path_node in reversed(paths[i]):
                path_node.value_sum += value
                path_node.visit_count += 1
                value = -value

    return roots
