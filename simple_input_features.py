'''
This file used for extracting simple feature set for a chess position.
The feature set is a binary vector of size 768 (64*6*2), where each unit
is 1 if there is a piece type at position at the specified color, or 0 if not.
Positions are numbered between 1 and 64, where 1 is A1, 2 is A2 and 64 is H8.
Piece types are sorted as: (King, Queen, Rook, Bishop, Knight, Pawn).
Color can be one of the following: (White, Black).
'''

import chess

USE_KING_SQ = False
NUM_SQ = 64
NUM_PT = 10 if USE_KING_SQ else 12
NUM_PLANES = (NUM_SQ * NUM_PT) if USE_KING_SQ else NUM_PT


def orient(is_white_pov: bool, sq: int):
    return (63 * (not is_white_pov)) ^ sq

# halfka
def att_idx(is_white_pov: bool, king_sq: int, sq: int, p: chess.Piece):
    p_idx = (p.piece_type - 1) * 2 + (p.color != is_white_pov)

    king_idx = king_sq * NUM_PLANES if USE_KING_SQ else 0
    return orient(is_white_pov, sq) + p_idx * NUM_SQ + king_idx


def get_features(board: chess.Board):
    indices = [0] * (NUM_SQ*NUM_PLANES)
    for sq, p in board.piece_map().items():
        if USE_KING_SQ:
            if p.piece_type == chess.KING:
                continue
        indices[att_idx(board.turn, orient(board.turn, board.king(board.turn)), sq, p)] = 1.0
    return indices


# initial_board = chess.Board()
# print(initial_board)
# #
# vector = get_features(initial_board)
# #
# print(vector)
