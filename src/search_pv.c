/*-------------------------------------------------------------------------------
  tucano is a chess playing engine developed by Alcides Schulz.
  Copyright (C) 2011-present - Alcides Schulz

  tucano is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  tucano is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You can find the GNU General Public License at http://www.gnu.org/licenses/
-------------------------------------------------------------------------------*/

#include "globals.h"

//-------------------------------------------------------------------------------------------------
//  Principal variation search.
//-------------------------------------------------------------------------------------------------
int search_pv(GAME *game, UINT incheck, int alpha, int beta, int depth)
{
    MOVE_LIST   ml;
    MOVE    best_move = MOVE_NONE;
    int     best_score = -MAX_SCORE;
    int     score = 0;
    UINT    gives_check;
    int     move_count = 0;
    int     extensions;
    int     ply = get_ply(&game->board);
    int     turn = side_on_move(&game->board);
    MOVE    move;
    int     reductions;

    assert(incheck == 0 || incheck == 1);
    assert(alpha >= -MAX_SCORE && alpha <= MAX_SCORE);
    assert(beta >= -MAX_SCORE && beta <= MAX_SCORE);
    assert(beta > alpha);
    assert(depth <= MAX_DEPTH);
    
    // begin my code
    int cur_depth_min = 4;
    int depth_equal = 2;
    int data_collect_now = 0;
    if (rand() % 100 == 0)
    {
        data_collect_now = 1;
    }
    
    // my add
    extern count_visit_cutoff;
    extern count_visit;

    // games where it not stop at depth 2 for some reason and new line will never be added to dataset, so abort.
    // if (game->search.extract_data && game->search.cur_depth >= cur_depth_min + 1) game->search.abort = TRUE;
    // get board 'fen representation'
    char fen[1024];
    util_get_board_fen(&game->board, fen);
    if (game->search.cur_depth >= cur_depth_min && depth == depth_equal)
    {
        // send 'fen' and 'beta' to python function called 'evaluate(fen, beta)'
        // this function returns 0 or 1.
        // if this function returns 1, use high-fail pruning and return
        if (game->search.use_bpruning == 1)
        {
            long prune_result = deep_prune(fen, beta, depth);
            game->search.nodes_checked_cutting_b++;
            if (prune_result == 1)
            {
                // add 1 node beta pruned
                game->search.nodes_cutted_b++;
                return beta;
            }
        }
       
    }
    // end my code

    if (depth <= 0) return quiesce(game, incheck, alpha, beta, 0);

    game->pv_line.pv_size[ply] = ply;
    game->search.nodes++;

    if (ply > 0 && is_draw(&game->board)) return 0;

    check_time(game);

    if (game->search.abort) return 0;

    assert(ply >= 0 && ply <= MAX_PLY);
    if (ply >= MAX_PLY) return evaluate(game, alpha, beta);

    //  Mate pruning.
    alpha = MAX(-MATE_VALUE + ply, alpha);
    beta = MIN(MATE_VALUE - ply, beta);
    if (alpha >= beta) return alpha;

    //  Get move hint from transposition table
    TT_RECORD tt_record;
    tt_read(game->board.key, &tt_record);
    MOVE trans_move = tt_record.info.move;

    // Reduction when position is not on transposition table. Idea from Prodeo chess engine (from Ed Schroder).
    
    // clear
    /*if (depth > 3 && trans_move == MOVE_NONE && !incheck) {
        depth--;
    }*/

    game->eval_hist[ply] =  (game, alpha, beta);

    //  Loop through move list
    select_init(&ml, game, incheck, trans_move, FALSE);
    while ((move = next_move(&ml)) != MOVE_NONE) {

        assert(is_valid(&game->board, move));
        
        if (!is_pseudo_legal(&game->board, ml.pins, move)) continue;

        move_count++;

        reductions = 0;
        extensions = 0;

        gives_check = is_check(&game->board, move);

        // extension if move puts opponent in check
        if (gives_check && (depth < 4 || see_move(&game->board, move) >= 0)) {
            extensions = 1;
        }

        // singular move extension
        if (ply > 0 && tt_record.data && move == trans_move && tt_record.info.flag >= TT_LOWER && depth >= 8 && !extensions) {
            score = score_from_tt(tt_record.info.score, game->board.ply);
            if (tt_record.info.depth >= depth - 3 && !is_mate_score(score)) {
                int reduced_beta = score - 4 * depth;
                score = search_singular(game, incheck, reduced_beta, depth / 2, move);
                if (game->search.abort) return 0;
                if (score < reduced_beta) {
                    extensions = 1;
                }
            }
        }

        // Pruning or depth reductions
        // clear
        //if (!incheck && !extensions && move_count > 1 && move_is_quiet(move)) {
        //    if (!is_killer(&game->move_order, turn, ply, move)) {
        //        if (!is_counter_move(&game->move_order, flip_color(turn), get_last_move_made(&game->board), move)) {
        //            // Futility pruning: eval + margin below beta.
        //            if (depth < 10) {
        //                int pruning_margin = depth * (50 + get_pruning_margin(&game->move_order, turn, move));
        //                if (evaluate(game, alpha, beta) + pruning_margin < alpha) {
        //                    continue;
        //                }
        //            }
        //            // Late move reductions: reduce depth for later moves
        //            if (move_count > 3 && depth > 2) {
        //                reductions = reduction_table[MIN(depth, MAX_DEPTH - 1)][MIN(move_count, MAX_MOVE - 1)];
        //                if (reductions > 0 && !get_has_bad_history(&game->move_order, turn, move)) reductions--;
        //            }
        //        }
        //    }
        //}

        // Make move and search new position.
        make_move(&game->board, move);

        assert(valid_is_legal(&game->board, move));

        // clear
        extensions = reductions = 0;

        if (best_score == -MAX_SCORE) {
            score = -search_pv(game, gives_check, -beta, -alpha, depth - 1 + extensions - reductions);
        }

        // clear
        // do only search_pv and never call search_zw
        /*else {
            score = -search_pv(game, gives_check, -beta, -alpha, depth - 1 + extensions - reductions);
        }*/
        
        else {
            score = -search_zw(game, gives_check, -alpha, depth - 1 + extensions - reductions);
            if (!game->search.abort && score > alpha && reductions) {
                score = -search_zw(game, gives_check, -alpha, depth - 1 + extensions);
            }
            if (!game->search.abort && score > alpha) {
                score = -search_pv(game, gives_check, -beta, -alpha, depth - 1 + extensions);
            }
        }

        undo_move(&game->board);
        if (game->search.abort) return 0;

        //  Score verification.
        if (score > best_score) {
            if (score > alpha) {
                update_pv(&game->pv_line, ply, move);
                if (ply == 0) post_info(game, score, depth);
                alpha = score;
                best_move = move;
                if (score >= beta) {
                    if (move_is_quiet(move)) {
                        save_beta_cutoff_data(&game->move_order, turn, ply, move, &ml, get_last_move_made(&game->board));
                    }
                    tt_record.info.move = move;
                    tt_record.info.depth = (S8)depth;
                    tt_record.info.flag = TT_LOWER;
                    tt_record.info.score = score_to_tt(score, ply);
                    tt_save(game->board.key, &tt_record);
                    // my add
                    if (game->search.cur_depth >= cur_depth_min && depth == depth_equal && !is_mate_score(beta))  count_visit_cutoff++;
                    if (game->search.cur_depth >= cur_depth_min && depth == depth_equal && !is_mate_score(beta)) count_visit++;
                    
                    if (game->search.extract_data)
                    {
                        if (game->search.cur_depth >= cur_depth_min && depth == depth_equal && data_collect_now && !is_mate_score(beta))
                        {
                            // found dot labeld with "pruned"
                            add_datarow_to_dataset(fen, beta / 200.0, 1);
                        }
                    }
                    return score;
                }
            }
            best_score = score;
        }
    }

    // my add
    if (game->search.cur_depth >= cur_depth_min && depth == depth_equal && !is_mate_score(beta)) count_visit++;

    if (game->search.cur_depth >= cur_depth_min && depth == depth_equal && data_collect_now && !is_mate_score(beta))
    {
        if (game->search.extract_data)
        {
            // found dot labeld with "not pruned"
            add_datarow_to_dataset(fen, beta / 200.0, 0);
        }
    }

    //  Draw or checkmate.
    if (best_score == -MAX_SCORE) {
        return (incheck ? -MATE_VALUE + ply : 0);
    }

    if (best_move != MOVE_NONE) {
        tt_record.info.move = best_move;
        tt_record.info.depth = (S8)depth;
        tt_record.info.flag = TT_EXACT;
        tt_record.info.score = score_to_tt(best_score, ply);
    }
    else {
        tt_record.info.move = MOVE_NONE;
        tt_record.info.depth = (S8)depth;
        tt_record.info.flag = TT_UPPER;
        tt_record.info.score = score_to_tt(best_score, ply);
    }
    tt_save(game->board.key, &tt_record);

    return best_score;
}

// end
