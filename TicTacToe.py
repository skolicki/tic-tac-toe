# This code learns to play tic-tac-toe either using Q-learning or Sarsa reinforcement learning.

# States are represented as 9 characters tuples (lists are not hashable, and thus
# cannot serve as keys in dictionaries).

# Represents a board with the following indices:
# 0 1 2
# 3 4 5
# 6 7 8


import random

REWARDS = {"X"  : 100,
           "O"  : -100,
           True : -3,  # draw
           False: 0}

epsilon = 0.1
alpha = 0.8
gamma = 0.9

Q = {}

def full(state):
  return not None in state

def init():
  return tuple([None] * 9)

# returns a symbol that repeats 3 times at given indices
def same_symbol(state, i ,j, k):
  return None if state[i] != state[j] or state[j] != state[k] else state[i]

# returns the symbol if either side won, or "full" if the board if full, None otherwise
def evaluate(state):
  return (same_symbol(state, 0, 1, 2)  # rows
    or same_symbol(state, 3, 4, 5)
    or same_symbol(state, 6, 7, 8)
    or same_symbol(state, 0, 3, 6)   # columns
    or same_symbol(state, 1, 4, 7)
    or same_symbol(state, 2, 5, 8)
    or same_symbol(state, 0, 4, 8)   # diagonals
    or same_symbol(state, 2, 4, 6)
    or full(state))

def possible_moves(state):
  return [i for i, val in enumerate(state) if not val]

def reverse(symbol):
  return "O" if symbol == "X" else "X"

def update_Q(state, action, reward, best_value):
  current = Q.get((state, action), random.randint(-10, 11))
  Q[state, action] = current + alpha * (reward + gamma * best_value - current)

def move(state, action, symbol):
  return tuple(val if i != action else symbol for i, val in enumerate(state))

def epsilon_greedy(state, symbol):
  actions = possible_moves(state)
  if not actions:
    return None, 0
  value_fun = lambda a: Q.get((state, a), 0) if symbol == "X" else -Q.get((state, a), 0)
  best = max(actions, key = value_fun)
  return random.choice(actions) if random.random() < epsilon else best, value_fun(best)

def episode(mode, debug = False):
  symbol = "X"
  state = init()
  action, _ = epsilon_greedy(state, symbol)
  while action != None:
    next_state = move(state, action, symbol)
    symbol = reverse(symbol)
    status = evaluate(next_state)  # can only be prev symbol
    next_action, best_value = epsilon_greedy(next_state, symbol) if not status else (None, 0)
    update_value = Q.get((next_state, next_action), 0) if mode == "sarsa" else best_value
    reward = (REWARDS[symbol] / 20) if not status else REWARDS[status]   # assign opposite small neg reward
    update_Q(state, action, reward, update_value)
    state, action = next_state, next_action
    if debug:
      print '\n'.join(["---"] + [''.join(e if e else "." for e in s) for s in state[:3], state[3:6], state[6:]])

def learn(mode):
  bucket = 1000
  for iteration in xrange(100 * bucket):
    episode(mode)
    if not iteration % bucket:
      print iteration
  episode(mode, True)
  print "Visited " + str(len(Q)) + " states out of 3^9 = 19683"

learn("q_learning")  # or "sarsa", anything else is treated as Q-learning in fact
