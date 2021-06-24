# This code learns to play tic-tac-toe from scratch, either using Q-learning or Sarsa reinforcement learning.

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


# Returns an empty board
def init_board():
    return tuple([None] * 9)


# Checks if the board is fully filled
def full(board):
    return "partial" if None in board else "full"


# Prints the board
def print_board(board):
    print('\n'.join([''.join(e if e else "." for e in s) for s in (board[:3], board[3:6], board[6:])]))


# Returns a symbol that repeats 3 times at given indices, None otherwise
def same_symbol(board, i, j, k):
    return None if board[i] != board[j] or board[j] != board[k] else board[i]


# Returns the symbol if either side won, or whether board is full
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


# Returns indices of still empty fields in a board
def possible_moves(board):
    return [i for i, val in enumerate(board) if not val]


# Returns the opposite player symbol to the given one
def reverse(symbol):
    return "O" if symbol == "X" else "X"


# Learns, i.e. updates the Q table based on the reward from an action
def update_Q(Q, board, action, reward, best_value):
    current = Q.get((board, action), random.randint(-1, 2))
    Q[board, action] = current + alpha * (reward + gamma * best_value - current)


# Puts the symbol in the field with action index on the board
def move(board, action, symbol):
    return tuple(val if i != action else symbol for i, val in enumerate(board))


# Returns the best action or, with epsilon probability, a random action
def epsilon_greedy(Q, board, symbol, epsilon):
    actions = possible_moves(board)
    if not actions:
        return None, 0

    def action_value(action):
        return Q.get((board, action), 0) if symbol == "X" else -Q.get((board, action), 0)

    best = max(actions, key=action_value)
    return random.choice(actions) if random.random() < epsilon else best, action_value(best)


# Plays a single learning game
def episode(Q, mode):
    symbol = "X"
    board = init_board()
    action, _ = epsilon_greedy(Q, board, symbol, epsilon)
    while action is not None:
        next_board = move(board, action, symbol)
        symbol = reverse(symbol)
        status = evaluate(next_board)  # can only be prev symbol
        next_action, best_value = epsilon_greedy(Q, next_board, symbol, epsilon) if status == "partial" else (None, 0)
        update_value = Q.get((next_board, next_action), 0) if mode == "sarsa" else best_value
        reward = (REWARDS[symbol] / 20) if status == "partial" else REWARDS[status]  # assign opposite small neg reward
        update_Q(Q, board, action, reward, update_value)
        board, action = next_board, next_action


# Learns by playing multiple games and returns the learnt Q table
def learn(mode, max_iteration):
    Q = {}
    bucket = 1000
    for iteration in range(max_iteration):
        episode(Q, mode)
        if not iteration % bucket:
            print("iteration: " + str(iteration) + " out of " + str(max_iteration))
    print("Visited " + str(len(Q)) + " board states out of 3^9 = 19683")
    return Q


# Prints an example game
def example_play(Q):
    symbol = "X"
    board = init_board()
    while evaluate(board) == "partial":
        board = move(board, epsilon_greedy(Q, board, symbol, 0)[0], symbol)
        symbol = reverse(symbol)
        print("")
        print_board(board)


def play(Q, symbol):
    print("Let's play. I'll play the X symbol, and you'll play the O")
    print("When providing your move, please enter the index of the field according to:")
    print("0 1 2")
    print("3 4 5")
    print("6 7 8")
    board = init_board()
    while evaluate(board) == "partial":
        action = epsilon_greedy(Q, board, symbol, 0)[0] if symbol == "X" else int(input("Please provide your move index: "))
        board = move(board, action, symbol)
        symbol = reverse(symbol)
        print("")
        print_board(board)
    result = evaluate(board)
    if result == "X":
        print("I won!")
    elif result == "O":
        print("Congratulations, you won!")
    else:
        print("Oh well, it's a draw")


print("First, I'm learning from scratch")
Q = learn("q_learning", 1000000) # either "sarsa" or anything else is treated as Q-learning
print("-------------")
print("Now I'm showing an example game")
example_play(Q)
print("And now, let's play")
again = "y"
while again == "y":
    play(Q, random.choice(["X", "O"]))
    again = str(input("Again? y/n "))
