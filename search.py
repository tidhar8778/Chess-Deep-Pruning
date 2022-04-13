from params import inf_value, use_quiesce, search_max_depth, layer_depth_pruning, quiesce_max_depth, margin_from_beta, \
    collect_data, max_beta_data_collect, min_beta_data_collect, verbose, iterative
from params import input_size, hidden_sizes, num_classes, dropout, folder_model
from utils import is_endgame
from utils import pawntable, knightstable, bishopstable, rookstable, queenstable, kingstable_middlegame, \
    kingstable_endgame
import chess
import time
from nn import evaluate
from NeuralNetwork import NeuralNetwork
import torch
import copy
import numpy as np

piecetypes = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
tables = [pawntable, knightstable, bishopstable, rookstable, queenstable, kingstable_middlegame, kingstable_endgame]
piecevalues = [100, 320, 330, 500, 900]
par_indi_del = np.load(folder_model+'/par_indi_del.npy')
path_for_model = folder_model+'/model.ckpt'


def init_evaluate_board(board):
    """
    Set boardvalue (in centipawns) as a heuristic of the evaluated board.
    Based on the quantity of pieces each side has and their position on the board.
    boardvalue is with respective to White, always
    :param board:
    :return:
    """
    global boardvalue

    kingstable = kingstable_endgame if is_endgame(board) else kingstable_middlegame

    wp = len(board.pieces(chess.PAWN, chess.WHITE))
    bp = len(board.pieces(chess.PAWN, chess.BLACK))
    wn = len(board.pieces(chess.KNIGHT, chess.WHITE))
    bn = len(board.pieces(chess.KNIGHT, chess.BLACK))
    wb = len(board.pieces(chess.BISHOP, chess.WHITE))
    bb = len(board.pieces(chess.BISHOP, chess.BLACK))
    wr = len(board.pieces(chess.ROOK, chess.WHITE))
    br = len(board.pieces(chess.ROOK, chess.BLACK))
    wq = len(board.pieces(chess.QUEEN, chess.WHITE))
    bq = len(board.pieces(chess.QUEEN, chess.BLACK))

    material = 100 * (wp - bp) + 320 * (wn - bn) + 330 * (wb - bb) + 500 * (wr - br) + 900 * (wq - bq)

    pawnsq = sum([pawntable[i] for i in board.pieces(chess.PAWN, chess.WHITE)])
    pawnsq = pawnsq + sum([-pawntable[chess.square_mirror(i)]
                           for i in board.pieces(chess.PAWN, chess.BLACK)])
    knightsq = sum([knightstable[i] for i in board.pieces(chess.KNIGHT, chess.WHITE)])
    knightsq = knightsq + sum([-knightstable[chess.square_mirror(i)]
                               for i in board.pieces(chess.KNIGHT, chess.BLACK)])
    bishopsq = sum([bishopstable[i] for i in board.pieces(chess.BISHOP, chess.WHITE)])
    bishopsq = bishopsq + sum([-bishopstable[chess.square_mirror(i)]
                               for i in board.pieces(chess.BISHOP, chess.BLACK)])
    rooksq = sum([rookstable[i] for i in board.pieces(chess.ROOK, chess.WHITE)])
    rooksq = rooksq + sum([-rookstable[chess.square_mirror(i)]
                           for i in board.pieces(chess.ROOK, chess.BLACK)])
    queensq = sum([queenstable[i] for i in board.pieces(chess.QUEEN, chess.WHITE)])
    queensq = queensq + sum([-queenstable[chess.square_mirror(i)]
                             for i in board.pieces(chess.QUEEN, chess.BLACK)])
    kingsq = sum([kingstable[i] for i in board.pieces(chess.KING, chess.WHITE)])
    kingsq = kingsq + sum([-kingstable[chess.square_mirror(i)]
                           for i in board.pieces(chess.KING, chess.BLACK)])

    boardvalue = material + pawnsq + knightsq + bishopsq + rooksq + queensq + kingsq

    return boardvalue



# Get current board evaluation. Not used
def evaluate_board(board):
    """
    Return score (in centipawns) as a heuristic of the evaluated board.
    Based on the quantity of pieces each side has and their position on the board.
    Score is with respective to White alwqays.

    :param board:
    :return:
    """

    kingstable = kingstable_endgame if is_endgame(board) else kingstable_middlegame

    wp = len(board.pieces(chess.PAWN, chess.WHITE))
    bp = len(board.pieces(chess.PAWN, chess.BLACK))
    wn = len(board.pieces(chess.KNIGHT, chess.WHITE))
    bn = len(board.pieces(chess.KNIGHT, chess.BLACK))
    wb = len(board.pieces(chess.BISHOP, chess.WHITE))
    bb = len(board.pieces(chess.BISHOP, chess.BLACK))
    wr = len(board.pieces(chess.ROOK, chess.WHITE))
    br = len(board.pieces(chess.ROOK, chess.BLACK))
    wq = len(board.pieces(chess.QUEEN, chess.WHITE))
    bq = len(board.pieces(chess.QUEEN, chess.BLACK))

    material = 100 * (wp - bp) + 320 * (wn - bn) + 330 * (wb - bb) + 500 * (wr - br) + 900 * (wq - bq)

    pawnsq = sum([pawntable[i] for i in board.pieces(chess.PAWN, chess.WHITE)])
    pawnsq = pawnsq + sum([-pawntable[chess.square_mirror(i)]
                           for i in board.pieces(chess.PAWN, chess.BLACK)])
    knightsq = sum([knightstable[i] for i in board.pieces(chess.KNIGHT, chess.WHITE)])
    knightsq = knightsq + sum([-knightstable[chess.square_mirror(i)]
                               for i in board.pieces(chess.KNIGHT, chess.BLACK)])
    bishopsq = sum([bishopstable[i] for i in board.pieces(chess.BISHOP, chess.WHITE)])
    bishopsq = bishopsq + sum([-bishopstable[chess.square_mirror(i)]
                               for i in board.pieces(chess.BISHOP, chess.BLACK)])
    rooksq = sum([rookstable[i] for i in board.pieces(chess.ROOK, chess.WHITE)])
    rooksq = rooksq + sum([-rookstable[chess.square_mirror(i)]
                           for i in board.pieces(chess.ROOK, chess.BLACK)])
    queensq = sum([queenstable[i] for i in board.pieces(chess.QUEEN, chess.WHITE)])
    queensq = queensq + sum([-queenstable[chess.square_mirror(i)]
                             for i in board.pieces(chess.QUEEN, chess.BLACK)])
    kingsq = sum([kingstable[i] for i in board.pieces(chess.KING, chess.WHITE)])
    kingsq = kingsq + sum([-kingstable[chess.square_mirror(i)]
                           for i in board.pieces(chess.KING, chess.BLACK)])

    score = material + pawnsq + knightsq + bishopsq + rooksq + queensq + kingsq

    return score

def update_eval(board, mov, side, is_reverse):
    """
    After each move, update the evaluation considering: the pieces changes for each player and their new position
    :param board:
    :param mov:
    :param side:
    :param is_reverse:
    :return:
    """
    global boardvalue

    def reverse_castling(mov):
        global boardvalue
        if movingpiece == chess.KING and (mov.from_square == chess.E1) and (mov.to_square == chess.G1):
            boardvalue = boardvalue + rookstable[chess.H1]
            boardvalue = boardvalue - rookstable[chess.F1]
        elif movingpiece == chess.KING and (mov.from_square == chess.E1) and (mov.to_square == chess.C1):
            boardvalue = boardvalue + rookstable[chess.A1]
            boardvalue = boardvalue - rookstable[chess.D1]
        elif movingpiece == chess.KING and (mov.from_square == chess.E8) and (mov.to_square == chess.G8):
            boardvalue = boardvalue - rookstable[chess.square_mirror(chess.H8)]
            boardvalue = boardvalue + rookstable[chess.square_mirror(chess.F8)]
        elif movingpiece == chess.KING and (mov.from_square == chess.E8) and (mov.to_square == chess.C8):
            boardvalue = boardvalue - rookstable[chess.square_mirror(chess.A8)]
            boardvalue = boardvalue + rookstable[chess.square_mirror(chess.D8)]

        return

    # update piecequares
    movingpiece = board.piece_type_at(mov.from_square)

    king_table_shifting = 0
    if movingpiece == chess.KING and is_endgame(board):
        king_table_shifting = 1
    if side:
        if is_reverse:
            boardvalue = boardvalue - tables[movingpiece - 1 + king_table_shifting][
                chess.square_mirror(mov.from_square)]
        else:
            boardvalue = boardvalue - tables[movingpiece - 1 + king_table_shifting][mov.from_square]
        # update castling
        if is_reverse:
            reverse_castling(mov)
        else:
            if movingpiece == chess.KING and (mov.from_square == chess.E1) and (mov.to_square == chess.G1):
                boardvalue = boardvalue - rookstable[chess.H1]
                boardvalue = boardvalue + rookstable[chess.F1]
            elif movingpiece == chess.KING and (mov.from_square == chess.E1) and (mov.to_square == chess.C1):
                boardvalue = boardvalue - rookstable[chess.A1]
                boardvalue = boardvalue + rookstable[chess.D1]
    else:
        if is_reverse:
            boardvalue = boardvalue + tables[movingpiece - 1 + king_table_shifting][mov.from_square]
        else:
            boardvalue = boardvalue + tables[movingpiece - 1 + king_table_shifting][
                chess.square_mirror(mov.from_square)]
        # update castling
        if is_reverse:
            reverse_castling(mov)
        else:
            if movingpiece == chess.KING and (mov.from_square == chess.E8) and (mov.to_square == chess.G8):
                boardvalue = boardvalue + rookstable[chess.square_mirror(chess.H8)]
                boardvalue = boardvalue - rookstable[chess.square_mirror(chess.F8)]
            elif movingpiece == chess.KING and (mov.from_square == chess.E8) and (mov.to_square == chess.C8):
                boardvalue = boardvalue + rookstable[chess.square_mirror(chess.A8)]
                boardvalue = boardvalue - rookstable[chess.square_mirror(chess.D8)]

    if side:
        if is_reverse:
            boardvalue = boardvalue + tables[movingpiece - 1 + king_table_shifting][chess.square_mirror(mov.to_square)]
        else:
            boardvalue = boardvalue + tables[movingpiece - 1 + king_table_shifting][mov.to_square]
    else:
        if is_reverse:
            boardvalue = boardvalue - tables[movingpiece - 1 + king_table_shifting][mov.to_square]
        else:
            boardvalue = boardvalue - tables[movingpiece - 1 + king_table_shifting][chess.square_mirror(mov.to_square)]

    # update material
    if board.is_capture(mov):
        if board.is_en_passant(mov):
            victim = chess.PAWN
            if is_reverse:
                if side:
                    victim_square = mov.to_square + 8
                else:
                    victim_square = mov.to_square - 8
            else:
                if side:
                    victim_square = mov.to_square - 8
                else:
                    victim_square = mov.to_square + 8

        else:
            victim = board.piece_at(mov.to_square).piece_type
            victim_square = mov.to_square

        if side:
            boardvalue = boardvalue + piecevalues[victim - 1]
            if is_reverse:
                boardvalue = boardvalue + tables[victim - 1][victim_square]
            else:
                boardvalue = boardvalue + tables[victim - 1][chess.square_mirror(victim_square)]

        else:
            boardvalue = boardvalue - piecevalues[victim - 1]
            if is_reverse:
                boardvalue = boardvalue - tables[victim - 1][chess.square_mirror(victim_square)]
            else:
                boardvalue = boardvalue - tables[victim - 1][victim_square]

    # update promotion
    if mov.promotion != None:
        if side:
            boardvalue = boardvalue + piecevalues[mov.promotion - 1] - piecevalues[movingpiece - 1]
            if is_reverse:
                boardvalue = boardvalue - tables[movingpiece - 1][chess.square_mirror(mov.to_square)] \
                             + tables[mov.promotion - 1][chess.square_mirror(mov.to_square)]
            else:
                boardvalue = boardvalue - tables[movingpiece - 1][mov.to_square] \
                             + tables[mov.promotion - 1][mov.to_square]
        else:
            boardvalue = boardvalue - piecevalues[mov.promotion - 1] + piecevalues[movingpiece - 1]
            if is_reverse:
                boardvalue = boardvalue + tables[movingpiece - 1][mov.to_square] \
                             - tables[mov.promotion - 1][mov.to_square]
            else:
                boardvalue = boardvalue + tables[movingpiece - 1][chess.square_mirror(mov.to_square)] \
                             - tables[mov.promotion - 1][chess.square_mirror(mov.to_square)]

    # update kings scores if enter to end game
    # attention! This update is done at make_move and unmake_move, better performance

    return mov


class Search:
    def __init__(self, player, use_layer_pruning):
        self.player = player
        self.use_layer_pruning = use_layer_pruning
        self.nodes_visited = 0
        self.nodes_visited_quiesce = 0
        self.nodes_visited_total = 0
        self.nodes_expanded = 0
        self.nodes_expanded_quiesce = 0
        self.nodes_evaluated = 0
        self.total_time_evals = 0
        self.best_move = None
        self.best_value = None
        self.search_fens = []
        self.data_collected = []  # each place contains dictionary: {fen: , beta: , pruned: [0,1], value (if pruned): }
        model = NeuralNetwork(input_size - len(par_indi_del), hidden_sizes, num_classes, dropout)
        model.load_state_dict(torch.load(path_for_model))
        model.eval()
        self.model = model
        self.counter_layer_depth_pruning = 0
        self.pruned_by_net = 0
        self.counter_layer_depth_pruning_ply = 0
        self.pruned_by_net_ply = 0
        self.iter_nodes_visited = []  # [nodes visited where max depth = 1, " " where max depth = 2...]
        self.iter_best_value = []
        self.moves_time = []
        self.nodes_visited_quiesce_by_level = [] # number_level_one (depth=max), number_level_two...
        self.best_sequence = []     # sequence of moves. Principal variation
        self.check_best_sequence = True  # if True, poss_move_idx of all levels is 0, follow the best_sequence to get good baseline.
        self.temp_best_sequence = []

    def reset_after_ply(self):
        # Set None at Search_item.best_move and Search_item.best_value
        # Set zero at Search_item.nodes_visited and Search_item.nodes_visited_quiesce
        # Set [] at Search_item.best_squence and Search_item.best_sequence_val
        self.best_move = None
        self.best_value = None
        self.nodes_visited = 0
        self.nodes_visited_quiesce = 0
        self.counter_layer_depth_pruning_ply = 0
        self.pruned_by_net_ply = 0
        self.best_sequence = []
        self.temp_best_sequence = []

    def iterative_deepening(self, state, max_depth, iterative=True):
        # idea: use window of about 1/3 pawn around last value, as done in Crafty
        global iter_max_depth
        if iterative:
            for i in range(max_depth):
                iter_max_depth = i + 1
                self.check_best_sequence = True
                value = self.search(-inf_value, inf_value, state, iter_max_depth, True)
                self.iter_best_value.append(value)

        else:
            iter_max_depth = max_depth
            value = self.search(-inf_value, inf_value, state, max_depth, True)

    def search(self, alpha, beta, state, depth, firstCall=False):
        """
        Main search algorithm. Based on alpha-beta search
        :param alpha:
        :param firstCall:
        :return:

        """
        if firstCall:
            start_time = time.time()
            init_evaluate_board(state)
            self.search_fens = []
            self.iter_nodes_visited.append(0)

        self.nodes_visited += 1
        self.nodes_visited_total += 1
        if iterative:
            self.iter_nodes_visited[iter_max_depth - 1] += 1

        if verbose:
            print(f'\rIteration up to depth {iter_max_depth} Nodes visited for this ply: {self.nodes_visited+self.nodes_visited_quiesce}', end='')
        if state.is_checkmate():
            return -(inf_value - (search_max_depth - depth))
        # self.search_fens.append(state.fen())
        if state.is_stalemate():
            return 0
        if state.is_insufficient_material():
            return 0
        if state.can_claim_fifty_moves():
            return 0
        # Check for draw by repetition
        if state.is_repetition(3):
            return 0

        if self.use_layer_pruning:
            if depth == search_max_depth - layer_depth_pruning:
                self.counter_layer_depth_pruning_ply += 1
                self.counter_layer_depth_pruning += 1
                if beta < max_beta_data_collect and beta > min_beta_data_collect:
                    if self.beta_pruning(state, alpha, beta):
                        self.pruned_by_net_ply += 1
                        self.pruned_by_net += 1
                        return beta
        if depth == 0:
            if use_quiesce:
                self.nodes_visited_quiesce -= 1  # Current node counted in self.nodes_visited. Subtract that node from nodes_visited_quiesce
                self.nodes_visited_total -= 1
                if iterative:
                    self.iter_nodes_visited[iter_max_depth - 1] -= 1
                return self.quiesce(alpha, beta, state, depth=quiesce_max_depth)
            else:
                return self.evaluate(state)

        if collect_data:
            if depth == search_max_depth - layer_depth_pruning:
                if beta < max_beta_data_collect and beta > min_beta_data_collect:
                    self.data_collected.append({'fen': state.fen(), 'beta': beta, 'pruned': 0, 'value': None})

        self.nodes_expanded += 1
        possible_moves = list(state.generate_legal_moves())
        continue_order = 1
        level = iter_max_depth - depth

        for poss_move_idx in range(len(possible_moves)):
            if iterative and depth != 1 and self.check_best_sequence:
                possibleMove = self.best_sequence[-1-level]
                # switch between possible_moves[0] and self.best_move
                for i in range(len(possible_moves)):
                    if possible_moves[i] == possibleMove:
                        possible_moves[0], possible_moves[i] = possible_moves[i], possible_moves[0]
                        break
            elif continue_order:
                possible_moves, continue_order = self.order_in_spot_next_move(state, possible_moves, poss_move_idx)
                possibleMove = possible_moves[poss_move_idx]
            else:
                possibleMove = possible_moves[poss_move_idx]

            copy_temp_best_sequence = copy.deepcopy(self.temp_best_sequence)

            self.make_move(state, possibleMove)
            value = -self.search(-beta, -alpha, state, depth - 1)
            self.unmake_move(state)

            if self.check_best_sequence:
                self.check_best_sequence = False    # Done checking best sequence as the first one.
            # choose the first move as baseline
            if poss_move_idx == 0:  # check if can delete this paragraph
                if firstCall:
                    if depth == 1:
                        self.best_move = possibleMove
                        self.best_value = value
            if value > alpha:
                alpha = value
                if firstCall:
                    self.best_move = possibleMove
                    self.best_value = value
                if value < beta:
                    try:
                        self.temp_best_sequence[depth-1] = possibleMove
                    except:
                        self.temp_best_sequence.append(possibleMove)

                    if depth == iter_max_depth:
                        self.best_sequence = copy.deepcopy(self.temp_best_sequence)

                # pruning
                else:   # value >= beta:
                    if collect_data:
                        if depth == search_max_depth - layer_depth_pruning:
                            if beta < max_beta_data_collect and beta > min_beta_data_collect:
                                self.data_collected[-1] = {'fen': state.fen(), 'beta': beta, 'pruned': 1, 'value': value}
                    return beta

            else:   # value <= alpha
                self.temp_best_sequence = copy_temp_best_sequence

        if firstCall and depth == search_max_depth:
            end_time = time.time()
            self.best_sequence.reverse()
            if verbose:
                print(f'\nCutoff by net: {self.pruned_by_net_ply} nodes from {self.counter_layer_depth_pruning_ply}')
                print('Time: {:.2f} seconds'.format(end_time - start_time))
                print('Total nodes visited for player: {}'.format(self.nodes_visited_total))
                print('Best move: {}, value for player ({}): {}'.format(self.best_move, "White" if state.turn else "Black", self.best_value))
                print('Principal variation: {}\n'.format(self.best_sequence))
            self.moves_time.append(end_time - start_time)

        return alpha

    def make_move(self, board, move):
        before_end_game = is_endgame(board)
        update_eval(board, move, board.turn, False)
        board.push(move)
        after_end_game = is_endgame(board)

        global boardvalue

        if not before_end_game and after_end_game:
            white_king = board.pieces(chess.KING, chess.WHITE).pop()
            black_king = board.pieces(chess.KING, chess.BLACK).pop()
            boardvalue = boardvalue - tables[chess.KING - 1][white_king]
            boardvalue = boardvalue + tables[chess.KING][white_king]
            boardvalue = boardvalue + tables[chess.KING - 1][chess.square_mirror(black_king)]
            boardvalue = boardvalue - tables[chess.KING][chess.square_mirror(black_king)]

        elif before_end_game and not after_end_game:
            white_king = board.pieces(chess.KING, chess.WHITE).pop()
            black_king = board.pieces(chess.KING, chess.BLACK).pop()
            boardvalue = boardvalue - tables[chess.KING][white_king]
            boardvalue = boardvalue + tables[chess.KING - 1][white_king]
            boardvalue = boardvalue + tables[chess.KING][chess.square_mirror(black_king)]
            boardvalue = boardvalue - tables[chess.KING - 1][chess.square_mirror(black_king)]

        return move

    def unmake_move(self, board):
        before_end_game = is_endgame(board)
        white_king = board.pieces(chess.KING, chess.WHITE).pop()
        black_king = board.pieces(chess.KING, chess.BLACK).pop()
        rev_move = board.pop()
        update_eval(board, rev_move, not board.turn, True)
        after_end_game = is_endgame(board)
        global boardvalue

        if not before_end_game and after_end_game:
            boardvalue = boardvalue + tables[chess.KING][white_king]
            boardvalue = boardvalue - tables[chess.KING - 1][white_king]
            boardvalue = boardvalue - tables[chess.KING][chess.square_mirror(black_king)]
            boardvalue = boardvalue + tables[chess.KING - 1][chess.square_mirror(black_king)]

        elif before_end_game and not after_end_game:
            boardvalue = boardvalue + tables[chess.KING - 1][white_king]
            boardvalue = boardvalue - tables[chess.KING][white_king]
            boardvalue = boardvalue - tables[chess.KING - 1][chess.square_mirror(black_king)]
            boardvalue = boardvalue + tables[chess.KING][chess.square_mirror(black_king)]

        return rev_move

    def evaluate(self, state):
        t1 = time.time()
        ev_score = boardvalue / 100
        if not state.turn:  # black's turn
            ev_score = -ev_score
        if ev_score == -0.0:
            ev_score = 0.0
        time_eval = time.time() - t1
        self.nodes_evaluated += 1
        self.total_time_evals += time_eval
        return ev_score

    # Get position and beta value and find either to prune or not
    def beta_pruning(self, state, alpha, beta):
        predicted = evaluate(state, beta, self.model)
        return predicted

    def quiesce(self, alpha, beta, state, depth):
        self.nodes_visited_quiesce += 1
        self.nodes_visited_total += 1
        if iterative:
            self.iter_nodes_visited[iter_max_depth - 1] += 1
        if depth != quiesce_max_depth:
            try:
                self.nodes_visited_quiesce_by_level[quiesce_max_depth - depth] += 1
            except:
                self.nodes_visited_quiesce_by_level.append(1)
            if state.is_checkmate():
                return -(inf_value - search_max_depth - (quiesce_max_depth - depth))
            if state.is_stalemate():
                return 0
            if state.is_insufficient_material():
                return 0
            if state.can_claim_fifty_moves():  # Check only for captures anyway, so don't need to check for 3-fold repetition
                return 0

        if depth <= 0 and not state.is_check():
            return self.evaluate(state)

        stand_pat = self.evaluate(state)
        if stand_pat >= beta:
            return beta
        if stand_pat > alpha:
            alpha = stand_pat

        self.nodes_expanded_quiesce += 1
        possible_moves = list(state.generate_legal_moves())
        continue_order = 1
        for poss_move_idx in range(len(possible_moves)):
            if continue_order:
                possible_moves, continue_order = self.order_in_spot_next_move(state, possible_moves, poss_move_idx)
            move = possible_moves[poss_move_idx]  # mvv/lva
            if not state.is_capture(move):  # If you want to search for checks, should be treated specially
                break

            # If the capturing piece is more valuable from the captured piece and
            # the capturing is not the capturing of the last piece of second player and
            # the SEE value is lower than 0 then
            # cut (break).
            # Idea from Crafty code.
            piece_at = state.piece_type_at(move.to_square)
            if state.is_en_passant(move):
                piece_at = chess.PAWN
            if state.piece_type_at(move.from_square) > piece_at:
                if self.number_of_pieces(state, not state.turn) > 1:
                    static_see = self.see_score(state, move.to_square, move)
                    if static_see < 0:
                        continue

            self.make_move(state, move)
            score = -self.quiesce(-beta, -alpha, state, depth - 1)
            self.unmake_move(state)

            if score >= beta:
                return beta
            if score > alpha:
                alpha = score

        return alpha

    def number_of_pieces(self, board, turn):
        """
        Return how many pieces not include the king there are for turn player

        :return:
        """
        count = 0
        for p in board.piece_map().values():
            if p.color == turn and p.piece_type != chess.KING:
                count += 1

        return count

    def see_score(self, board, square, move):
        square_set_attackers_first_side = board.attackers(board.turn, square)
        first_side_attackers_squares = [square_set_attackers_first_side.pop() for i in range(
            len(board.attackers(board.turn, square)))]  # Does include the starting attacker
        first_side_attackers = [board.piece_type_at(sq) for sq in first_side_attackers_squares if
                                square in board.pin(board.turn, sq)]  # Does include the starting attacker
        # Exclude the original capture from
        first_side_attackers_squares.remove(move.from_square)
        first_side_attackers.remove(
            board.piece_type_at(move.from_square))  # Remove will remove only the first occurrence of selected value
        first_side_attackers_dicts = [{board.piece_type_at(sq): sq} for sq in first_side_attackers_squares if
                                      sq != move.from_square and square in board.pin(board.turn,
                                                                                     sq)]  # Does not include the starting attacker

        square_set_attackers_second_side = board.attackers(not board.turn, square)
        second_side_attackers_squares = [square_set_attackers_second_side.pop() for i in
                                         range(len(board.attackers(not board.turn, square)))]
        second_side_attackers = [board.piece_type_at(sq) for sq in second_side_attackers_squares if
                                 square in board.pin(not board.turn, sq)]
        second_side_attackers_dicts = [{board.piece_type_at(sq): sq} for sq in second_side_attackers_squares if
                                       square in board.pin(not board.turn, sq)]

        static_score = piecevalues[board.piece_type_at(square) - 1]  # Score of captured piece
        attacked_piece_value = 10 ** 6 if board.piece_type_at(move.from_square) == 6 else piecevalues[
            board.piece_type_at(move.from_square) - 1]

        x_ray_return = self.get_x_ray_attacks(board, square, move.from_square)
        if x_ray_return:
            x_ray_piece, x_ray_square = x_ray_return
            if x_ray_piece.color == board.turn:
                first_side_attackers.append(x_ray_piece.piece_type)
                first_side_attackers_dicts.append({x_ray_piece.piece_type: x_ray_square})
            else:
                second_side_attackers.append(x_ray_piece.piece_type)
                second_side_attackers_dicts.append({x_ray_piece.piece_type: x_ray_square})

        side_to_move = not board.turn

        while 1:
            if side_to_move == board.turn:
                attackers = first_side_attackers
                attackers_dicts = first_side_attackers_dicts
                other_attackers = second_side_attackers
                other_attackers_dicts = second_side_attackers_dicts
                multiplier = 1
            else:
                attackers = second_side_attackers
                attackers_dicts = second_side_attackers_dicts
                other_attackers = first_side_attackers
                other_attackers_dicts = first_side_attackers_dicts
                multiplier = -1

            if not attackers:
                break

            new_attacked_piece_value = min(attackers)
            # Handle case of king capture
            if new_attacked_piece_value == 6 and other_attackers:
                break
            static_score = static_score + (multiplier * attacked_piece_value)
            attacked_piece_value = 10 ** 6 if new_attacked_piece_value == 6 else piecevalues[
                new_attacked_piece_value - 1]

            for i, d in enumerate(attackers_dicts):
                if new_attacked_piece_value in d.keys():
                    square_attacked = d[new_attacked_piece_value]
                    # Remove piece from attackers and attackers_dict
                    attackers.remove(new_attacked_piece_value)
                    attackers_dicts.pop(i)
                    break

            x_ray_return = self.get_x_ray_attacks(board, square, square_attacked)
            if x_ray_return:
                x_ray_piece, x_ray_square = x_ray_return
                if x_ray_piece.color == board.turn:
                    first_side_attackers.append(x_ray_piece.piece_type)
                    first_side_attackers_dicts.append({x_ray_piece.piece_type: x_ray_square})
                else:
                    second_side_attackers.append(x_ray_piece.piece_type)
                    second_side_attackers_dicts.append({x_ray_piece.piece_type: x_ray_square})

            side_to_move = not side_to_move

        return static_score / 100

    def get_x_ray_attacks(self, board, square, attacked_from_square):
        """
        Find x-ray attacks available after "attack" had been from attacked_from_square to square.
        Don't call this function if attacking piece is King or Knight, because no x-ray attack can be revealed.
        If piece attacking is bishop, search over the diagonal behind it.
        If piece attacking is rook, search over the file behind it.
        If piece attacking is queen, search over file or diagonal behind it, depend on which direction was the attack.
        :param board:
        :param square:
        :param attacked_from_square:
        :return: None or (piece, square) to add to first or second attackers list and attacker_dicts list
        """

        fr_file, fr_rank = chess.square_file(attacked_from_square), chess.square_rank(attacked_from_square)
        to_file, to_rank = chess.square_file(square), chess.square_rank(square)
        if fr_file == to_file:
            # check for queen or rook behind attacking piece vertical direction
            direction = 1 if fr_rank > to_rank else -1  # direction is 1 if we search upward or -1 if we search downward
            for i in range(fr_rank + direction, 8 if direction == 1 else -1, direction):
                piece = board.piece_at(chess.square(fr_file, i))
                if piece:
                    if piece.piece_type == chess.ROOK or piece == chess.QUEEN:
                        return piece, chess.square(fr_file, i)
                    return

        elif fr_rank == to_rank:
            # check for queen or rook behind attacking piece horizontal direction
            direction = 1 if fr_file > to_file else -1  # direction is 1 if we search to the right or -1 if we search to the left (white side perspective)
            for i in range(fr_file + direction, 8 if direction == 1 else -1, direction):
                piece = board.piece_at(chess.square(i, fr_rank))
                if piece:
                    if piece.piece_type == chess.ROOK or piece == chess.QUEEN:
                        return piece, chess.square(i, fr_rank)
                    return

        else:
            # Piece attacked from the diagonal, could be a bishop or a queen. If king (or knight) attacked this function should not be called.
            direction_file = 1 if fr_file > to_file else -1
            direction_rank = 1 if fr_rank > to_rank else -1
            step = 0
            while 1:
                step += 1
                check_file = fr_file + (direction_file * step)
                check_rank = fr_rank + (direction_rank * step)
                if check_file < 0 or check_file > 7 or check_rank < 0 or check_rank > 7:
                    return
                piece = board.piece_at(chess.square(check_file, check_rank))
                if piece:
                    if piece.piece_type == chess.BISHOP or piece == chess.QUEEN:
                        return piece, chess.square(check_file, check_rank)
                    return

        return

    def order_in_spot_next_move(self, board, legal_moves_list, current_move_index):
        """
        Get the board, legal moves list and the current index on the list to check, and update the list so the
        current index in list holds the best move to check upon remaining legal moves
        :param board:
        :param legal_moves_list:
        :param current_move_index:
        :return:
        """
        best_score = 0
        best_move_idx = current_move_index
        for move_idx in range(current_move_index, len(legal_moves_list)):
            move = legal_moves_list[move_idx]
            move_score = 0
            # capture moves
            if board.is_capture(move):
                if board.is_en_passant(move):
                    victim = chess.PAWN
                else:
                    victim = board.piece_at(move.to_square).piece_type
                attacker = board.piece_at(move.from_square).piece_type
                move_score = 100 * victim + (6 - attacker)  # mvv/lva score
            # check moves
            # if board.gives_check(move):
            #     move_score = 300

            # update
            if move_score > best_score:
                best_score = move_score
                best_move_idx = move_idx

        # swap current move spot with best move spot
        legal_moves_list[current_move_index], legal_moves_list[best_move_idx] = \
            legal_moves_list[best_move_idx], legal_moves_list[current_move_index]

        return legal_moves_list, best_score
