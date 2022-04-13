from params import inf_value
from params import search_max_depth, iterative, test_positions
from search import Search as Search
import random
import time

MATE_SCORE_VALUE = inf_value

random.seed(4)

class PlayerAlphaBeta:
    def __init__(self, white, use_pruning=False):
        self.white = white      # White is True if the player uses the white pieces, and False if the opposite
        self.use_pruning = use_pruning
        self.c_search = Search(self, use_pruning)
        self.choosing_moves_time = []

    def set_use_pruning(self, use):
        if use:
            self.use_pruning = True
        else:
            self.use_pruning = False

        self.c_search = Search(self, use)
        self.choosing_moves_time = []

    def choose_move(self, state):
        start_time = time.time()
        self.c_search.iterative_deepening(state, search_max_depth, iterative=iterative)
        move = self.c_search.best_move
        if not test_positions:
            self.c_search.reset_after_ply()
        end_time = time.time()
        self.choosing_moves_time.append(round(end_time-start_time, 1))
        return move
