# States are represented as 9 characters tuples (lists are not hashable, and thus
# cannot serve as keys in dictionaries).

# Represents a board with the following indices:
# 0 1 2
# 3 4 5
# 6 7 8


import random

DRAW_REWARD = -3
WIN_REWARD = 100
REWARDS = {"X"  : WIN_REWARD,
           "O"  : -WIN_REWARD,
           True : DRAW_REWARD,
           False: 0}

epsilon = 0.1
alpha = 0.8
gamma = 0.9

Q = {}

DEBUG = False

def full(state):
  return not None in state

# TODO(skolicki): should we start from random states? (with #X in [#O, #O + 1])
def init():
  return tuple([None] * 9)
  
# This creates invalid states. Meh, so what.
def init_random():
  state = []
  for i in range(8):
    state.append(random.choice(["X", "O", None]))
  point = random.randint(0, 9)
  state = state[:point] + [None] + state[point:]  # make sure we're not full
  return tuple(state)

# returns a symbol that repeats 3 times at given indices
def check_triple(state, i ,j, k):
  return None if state[i] != state[j] or state[j] != state[k] else state[i]  # works for 3 x None too

# returns the symbol if either side won, or True/False if the board if full/not full.
def evaluate(state):
  return (check_triple(state, 0, 1, 2)  # rows
    or check_triple(state, 3, 4, 5)
    or check_triple(state, 6, 7, 8)
    or check_triple(state, 0, 3, 6)   # columns
    or check_triple(state, 1, 4, 7)
    or check_triple(state, 2, 5, 8)
    or check_triple(state, 0, 4, 8)   # diagonals
    or check_triple(state, 2, 4, 6)
    or full(state))
    
def possible_moves(state):
  return [i for i, val in enumerate(state) if not val]

def reverse(symbol):
  return "O" if symbol == "X" else "X"

def update_Q(state, action, reward, best_value):
  current = Q.get((state, action), random.randint(-10, 11))  # TODO(skolicki): Use a random initial value?
  Q[state, action] = current + alpha * (reward + gamma * best_value - current)   
  
def move(state, action, symbol):
  return tuple(val if i != action else symbol for i, val in enumerate(state)), reverse(symbol)

def epsilon_greedy(state, symbol):
  actions = possible_moves(state)
  if not actions:
    return None, 0
  value_fun = lambda a: Q.get((state, a), 0) if symbol == "X" else -Q.get((state, a), 0)
  best = max(actions, key = value_fun)
  if random.random() > epsilon:
    return best, value_fun(best)
  else:
    if DEBUG:
      print "random"
    return random.choice(actions), value_fun(best)

def episode(mode, debug = False):
  symbol = "X"
  state = init()
  action, _ = epsilon_greedy(state, symbol)
  rewards_sum = 0
  steps = 0
  while action != None:
    next_state, symbol = move(state, action, symbol)
    status = evaluate(next_state)  # can only be prev symbol
    next_action, best_value = epsilon_greedy(next_state, symbol) if not status else (None, 0)
    update_value = Q.get((next_state, next_action), 0) if mode == "sarsa" else best_value
    reward = (REWARDS[symbol] / 20) if not status else REWARDS[status]   # assign opposite small neg reward
    rewards_sum += reward if symbol == "O" else -reward
    steps += 1
    update_Q(state, action, reward, update_value)
    state, action = next_state, next_action
    if debug:
      printState(state)
  return rewards_sum / steps
      
def printState(state):
  print "---"
  print ''.join(e if e else "." for e in state[:3])
  print ''.join(e if e else "." for e in state[3:6])
  print ''.join(e if e else "." for e in state[6:])

def learn(mode):
  rewards = 0
  bucket = 1000
  for iteration in xrange(100000):
    rewards += episode(mode)
    if not iteration % bucket:
      print "%d %d" % (iteration, rewards / bucket)
      rewards = 0
  DEBUG = True
  episode(mode, True)
  print "Visited " + str(len(Q)) + " states out of 3^9 = 19683"

learn("q_learning")


