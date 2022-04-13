import chess
import chess.pgn
import chess.svg
from Player import PlayerAlphaBeta
import time
from params import collect_data, white_use_pruning, black_use_pruning, test_positions, test_positions_full
import pickle
# from IPython.display import SVG
# from chess_gif.gif_maker import Gifmaker

# SVG(chess.svg.board(board=board,size=400))


def play_turn(white, current_board):
    '''

    :param white: true or false, is it white turn?
    :param current_board: chess.board(), current state of the board
    :return:
    '''
    global player_white
    global player_black
    print(f'\r{"White" if white else "Black"} playing ply number {len(move_history)+1}', end='\n')
    if current_board.is_stalemate():
        print('Stalemate: white and black are in tie')
        return 0
    elif current_board.is_insufficient_material():
        print('Tie by insufficient material')
        return 0
    elif current_board.can_claim_draw():
        print('Draw by threefold repetition or 50-moves rule')
        return 0
    else:
        if white:
            if not current_board.is_checkmate():
                move = player_white.choose_move(current_board)
                current_board.push(move)
                move_history.append(move)
                if collect_data or test_positions:
                    return 0
                return play_turn(not white, current_board)

            else:  # checkmate
                print('Black wins')
                return -1

        else:
            if not current_board.is_checkmate():
                move = player_black.choose_move(current_board)
                current_board.push(move)
                move_history.append(move)
                if collect_data or test_positions:
                    return 0
                # return #todo: comment for full game or uncomment for one-ply simulate
                return play_turn(not white, current_board)

            else:  # checkmate
                print('White wins')
                return 1


def play(game, board):
    global move_history
    move_history = []
    result = play_turn(board.turn, board)

    # save board
    boardsvg = chess.svg.board(board)
    outputfile = open('test.svg', "w")
    outputfile.write(boardsvg)
    outputfile.close()

    return result, move_history

def test_games():
    # position score by stockfish 12
    position_score = [0.27, 0.06, -0.03, -0.1, 0.01, 0.29, 0.11, 0.14,  0.15, 0.0]
    position_list = [#'1k6/3Q4/8/2K5/8/8/8/8 w - - 9 6',
        'r2r1k2/1pp1ppbp/1p6/3P1p2/2P5/P1NR4/1P3PPP/3R2K1 w - - 4 19',
                     'r2qkb1r/1p1b1p1p/p3p1p1/nB1n4/Q2P4/2N2N2/PP3PPP/R1B1K2R w KQkq - 0 12',
                     'r3kb1r/1p3p1p/p5p1/3pN3/2nP4/8/PP1B1PPP/R3K2R w KQ - 2 17',
                     '6k1/4Rp1p/6p1/p7/r2nPn2/5P2/1P1R1KPP/8 w - - 0 30',
                     'r4rk1/1p4p1/p1p4p/3p1qp1/PP1P4/1R2P1QP/2R2PP1/6K1 w - - 8 29',
                     'r1r2k2/p3qpp1/4p2p/2n1N3/8/Q3P3/P4PPP/2R2RK1 b - - 2 22',
                     '8/pp3p2/6p1/1P2k1P1/P3P3/1P2K3/8/8 b - - 2 43',
                     'r4rk1/2q1bppp/bnn1p3/p1ppP3/1p3B1P/3PNNP1/PPPQ1PB1/R3R1K1 b - - 3 15',
                     'r4k2/4qpp1/1p1pp2p/nP1bP3/3P1n2/5N1P/3N1PP1/R1Q2BK1 b - - 0 24',
                     '4rk1r/pp1b2pp/1qnb1p2/3pn3/3p4/1NPB1N1P/PP1BQPP1/1R2R1K1 b - - 1 16'
                     ]

    for position in position_list:
        yield position


def initialize_game(game_number, start_position, type_start_position, white_use_pruning, black_use_pruning):
    player_white.set_use_pruning(white_use_pruning)
    player_black.set_use_pruning(black_use_pruning)
    game = chess.pgn.Game()
    board = chess.Board()
    if type_start_position == 'fen':
        board.set_fen(start_position)
    elif type_start_position == 'epd':
        board.set_epd(start_position)

    current_player = player_white if board.turn else player_black
    game.setup(board)
    # print('Game number: {}'.format(game_number))
    # print('Start board:\n', board)
    t_start = time.time()
    result, moves = play(game, board)
    t_end = time.time()
    game.add_line(moves)
    games_dictionary.append({
        'Game number': game_number,
        'Start position': start_position,
        'White use pruning': white_use_pruning,
        'Black use pruning': black_use_pruning,
        'Result': result,
        'Moves': moves,
        'Time running game': t_end - t_start,
        'White nodes visited': player_white.c_search.nodes_visited_total,
        'White nodes expanded': player_white.c_search.nodes_expanded + player_white.c_search.nodes_expanded_quiesce,
        'White nodes evaluated': player_white.c_search.nodes_evaluated,
        'White nodes cutoff by net': player_white.c_search.pruned_by_net,
        'Black nodes visited': player_black.c_search.nodes_visited_total,
        'Black nodes expanded': player_black.c_search.nodes_expanded + player_black.c_search.nodes_expanded_quiesce,
        'Black nodes evaluated': player_black.c_search.nodes_evaluated,
        'Black nodes cutoff by net': player_black.c_search.pruned_by_net,
        'White moves time': player_white.choosing_moves_time,
        'Black moves time': player_black.choosing_moves_time
    })
    print('\nGame took {} minutes and {} seconds'.format(int((t_end - t_start) / 60),
                                                         round((t_end - t_start) - int((t_end - t_start) / 60) * 60)))
    if collect_data:
        return

    if test_positions or test_positions_full:
        game_data = {
            'fen': start_position,
            'stockfish evaluation': None,
            'turn': current_player.white,
            'use_pruning': current_player.use_pruning,
            'total nodes visited': current_player.c_search.nodes_visited_total,
            'evaluation': current_player.c_search.best_value,
            'best move': current_player.c_search.best_move,
            'pruned by net': current_player.c_search.pruned_by_net,
            'nodes check for pruning': current_player.c_search.counter_layer_depth_pruning,
            'time in seconds': t_end - t_start,
            'result': result,
            'num of plys': len(moves)
        }

        return game_data

    print('Node visited quiesce by level for white:\n', player_white.c_search.nodes_visited_quiesce_by_level)
    print('Node visited quiesce by level for black:\n', player_black.c_search.nodes_visited_quiesce_by_level)
    # return  # todo: uncomment
    # save game
    print(game, file=open("game" + str(game_number) + ".pgn", "w"), end="\n\n")
    print(game.mainline_moves())
    # save game gif
    # obj = Gifmaker('game.pgn', h_margin=20, v_margin=80)
    # obj.make_gif('chess_game.gif')
    print('For white:  {} nodes visited, {} nodes expanded, {} nodes evaluated, {} nodes cutoff by net'.format(
        player_white.c_search.nodes_visited_total,
        player_white.c_search.nodes_expanded + player_white.c_search.nodes_expanded_quiesce,
        player_white.c_search.nodes_evaluated,
        player_white.c_search.pruned_by_net))
    print('For black:  {} nodes visited, {} nodes expanded, {} nodes evaluated, {} nodes cutoff by net'.format(
        player_black.c_search.nodes_visited_total,
        player_black.c_search.nodes_expanded + player_black.c_search.nodes_expanded_quiesce,
        player_black.c_search.nodes_evaluated,
        player_black.c_search.pruned_by_net))
    #print('For black from total: {} quiesce nodes visited, {} quiesce nodes expanded, all nodes evaluated from quiesce'.format(
    #    player_black.c_search.nodes_visited_quiesce, player_black.c_search.nodes_expanded_quiesce
    #))
    print('Total:  {} nodes visited, {} nodes expanded, {} nodes evaluated, {} nodes cutoff by net'.format(
        player_white.c_search.nodes_visited_total + player_black.c_search.nodes_visited_total,
        player_white.c_search.nodes_expanded + player_white.c_search.nodes_expanded_quiesce + player_black.c_search.nodes_expanded + player_black.c_search.nodes_expanded_quiesce,
        player_white.c_search.nodes_evaluated + player_black.c_search.nodes_evaluated,
        player_white.c_search.pruned_by_net + player_black.c_search.pruned_by_net))
    print('Time for choosing moves for white (seconds):\n', player_white.choosing_moves_time)
    print('Time for choosing moves for black (seconds):\n', player_black.choosing_moves_time)
    print('Total time for white moves (seconds): {}, average for move: {}'.format(
        round(sum(player_white.choosing_moves_time)),

        round(sum(player_white.choosing_moves_time) / len(player_white.choosing_moves_time) if len(player_white.choosing_moves_time) else 0)))
    print('Total time for black moves (seconds): {}, average for move: {}'.format(
        round(sum(player_black.choosing_moves_time)),
        round(sum(player_black.choosing_moves_time) / len(player_black.choosing_moves_time) if len(player_black.choosing_moves_time) else 0)))


def play_main(init_board_fen):

    # # White use pruning
    # game_number = 0
    # for position in test_games():
    #     initialize_game(game_number, position, 'fen', True, False)
    #     game_number += 1
    #
    # # Black use pruning
    # game_number = 0
    # for position in test_games():
    #     initialize_game(game_number, position, 'fen', False, True)
    #     game_number += 1
    #
    # with open('Test games results margin 1.pkl', 'wb') as file_to_save:
    #     pickle.dump(games_dictionary, file_to_save)
    game_number = 10
    game_data = initialize_game(game_number, init_board_fen, 'fen', white_use_pruning, black_use_pruning)
    with open('game' + str(game_number) + '.pkl', 'wb') as file_to_save:
        pickle.dump(games_dictionary, file_to_save)

    if test_positions or test_positions_full:
        return game_data

    if player_white.c_search.data_collected:
        return player_white.c_search.data_collected
    else:
        return player_black.c_search.data_collected

player_white = PlayerAlphaBeta(white=True)
player_black = PlayerAlphaBeta(white=False)
games_dictionary = []

if __name__ == '__main__':
    init_board_fen = '8/8/3kp1R1/3b1pP1/2B5/1P2P2r/3K4/8 b - - 1 44'
    #init_board_fen = chess.Board().fen()
    play_main(init_board_fen)
    #Principal variation: [Move.from_uci('b7d5'), Move.from_uci('c3c4'), Move.from_uci('d5b7'), Move.from_uci('d4d5'), Move.from_uci('d8f6')]
