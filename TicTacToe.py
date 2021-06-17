# This code learns to play tic-tac-toe either using Q-learning or Sarsa reinforcement learning.

# Board states are represented as 9 characters tuples (lists are not hashable, and thus
# cannot serve as keys in dictionaries).

# Represents a board with the following indices:
# 0 1 2
# 3 4 5
# 6 7 8


import random

REWARDS = {"X": 100,
           "O": -100,
           "full": -3,  # draw
           "partial": 0}

epsilon = 0.1
alpha = 0.8
gamma = 0.9


def full(board):
    return "partial" if None in board else "full"


def init_board():
    return tuple([None] * 9)


# returns a symbol that repeats 3 times at given indices, None otherwise
def same_symbol(board, i, j, k):
    return None if board[i] != board[j] or board[j] != board[k] else board[i]


# returns the symbol if either side won, or "full" if the board if full, None otherwise
def evaluate(board):
    return (same_symbol(board, 0, 1, 2)  # rows
            or same_symbol(board, 3, 4, 5)
            or same_symbol(board, 6, 7, 8)
            or same_symbol(board, 0, 3, 6)  # columns
            or same_symbol(board, 1, 4, 7)
            or same_symbol(board, 2, 5, 8)
            or same_symbol(board, 0, 4, 8)  # diagonals
            or same_symbol(board, 2, 4, 6)
            or full(board))


# returns indices of empty fields in a board
def possible_moves(board):
    return [i for i, val in enumerate(board) if not val]


def reverse(symbol):
    return "O" if symbol == "X" else "X"


def update_Q(Q, board, action, reward, best_value):
    current = Q.get((board, action), random.randint(-10, 11))
    Q[board, action] = current + alpha * (reward + gamma * best_value - current)


def move(board, action, symbol):
    return tuple(val if i != action else symbol for i, val in enumerate(board))


def epsilon_greedy(Q, board, symbol):
    actions = possible_moves(board)
    if not actions:
        return None, 0

    def action_value(action):
        return Q.get((board, action), 0) if symbol == "X" else -Q.get((board, action), 0)

    best = max(actions, key=action_value)
    return random.choice(actions) if random.random() < epsilon else best, action_value(best)


def print_board(board):
    print('\n'.join(["---"] + [''.join(e if e else "." for e in s) for s in board[:3], board[3:6], board[6:]]))


def episode(Q, mode):
    symbol = "X"
    board = init_board()
    action, _ = epsilon_greedy(Q, board, symbol)
    while action is not None:
        next_board = move(board, action, symbol)
        symbol = reverse(symbol)
        status = evaluate(next_board)  # can only be prev symbol
        next_action, best_value = epsilon_greedy(Q, next_board, symbol) if status == "partial" else (None, 0)
        update_value = Q.get((next_board, next_action), 0) if mode == "sarsa" else best_value
        reward = (REWARDS[symbol] / 20) if status == "partial" else REWARDS[status]  # assign opposite small neg reward
        update_Q(Q, board, action, reward, update_value)
        board, action = next_board, next_action


def learn(mode):
    Q = {}
    bucket = 1000
    max_iteration = 100 * bucket
    for iteration in xrange(max_iteration):
        episode(Q, mode)
        if not iteration % bucket:
            print("iteration: " + str(iteration) + " out of " + str(max_iteration))
    print("Visited " + str(len(Q)) + " board states out of 3^9 = 19683")
    return Q


def example_play(Q):
    symbol = "X"
    board = init_board()
    while evaluate(board) is "partial":
        board = move(board, epsilon_greedy(Q, board, symbol)[0], symbol)
        symbol = reverse(symbol)
        print_board(board)


learning_mode = "q_learning"  # or "sarsa", anything else is treated as Q-learning in fact
print("First, I'm learning from scratch")
Q = learn(learning_mode)
print("-------------")
print("Now I'm showing an example game")
example_play(Q)


