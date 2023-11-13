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
//  Zero window search.
//-------------------------------------------------------------------------------------------------

#define STAT_NULL_DEPTH 4
int STAT_NULL_MARGIN[STAT_NULL_DEPTH] = { 0, 100, 200, 400 };

#define RAZOR_DEPTH 4
int RAZOR_MARGIN[RAZOR_DEPTH] = { 0, 300, 600, 1200 };

//-------------------------------------------------------------------------------------------------
//  Search
//-------------------------------------------------------------------------------------------------
int search_zw(GAME *game, UINT incheck, int beta, int depth)
{
    MOVE_LIST   ml;
    int     best_score = -MAX_SCORE;
    int     move_count = 0;
    int     score = 0;
    UINT    gives_check;
    int     ply = get_ply(&game->board);
    int     turn = side_on_move(&game->board);
    MOVE    move;
    int     alpha;
    int     razor_beta;

    assert(incheck == 0 || incheck == 1);
    assert(beta >= -MAX_SCORE && beta <= MAX_SCORE);
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

    check_time(game);
    if (game->search.abort) return 0;

    if (depth <= 0) return quiesce(game, incheck, beta - 1, beta, 0);

    game->search.nodes++;
	game->pv_line.pv_size[ply] = ply;

    if (ply > 0 && is_draw(&game->board)) return 0;

    assert(ply >= 0 && ply <= MAX_PLY);
    if (ply >= MAX_PLY) return evaluate(game, beta - 1, beta);

    //  Mate pruning.
    alpha = beta - 1;
    alpha = MAX(-MATE_VALUE + ply, alpha);
    beta = MIN(MATE_VALUE - ply, beta);
    if (alpha >= beta) return alpha;

    // transposition table score or move hint
    TT_RECORD tt_record;
    tt_read(game->board.key, &tt_record);
    if (tt_record.data && tt_record.info.depth >= depth && !EVAL_TUNING) {
        score = score_from_tt(tt_record.info.score, ply);
        if (tt_record.info.flag == TT_EXACT) return score;
        if (score >= beta && tt_record.info.flag == TT_LOWER) return score;
        if (score <= beta - 1 && tt_record.info.flag == TT_UPPER) return score;
    }
    MOVE trans_move = tt_record.info.move;

#ifdef EGTB_SYZYGY
    // endgame tablebase probe
    U32 tbresult = egtb_probe_wdl(&game->board, depth, ply);
    if (tbresult != TB_RESULT_FAILED) {
        game->search.tbhits++;
        S8 tt_flag;

        switch (tbresult) {
        case TB_WIN:
            score = EGTB_WIN - ply;
            tt_flag = TT_LOWER;
            break;
        case TB_LOSS:
            score = -EGTB_WIN + ply;
            tt_flag = TT_UPPER;
            break;
        default:
            score = 0;
            tt_flag = TT_EXACT;
            break;
        }

        if (tt_flag == TT_EXACT || (tt_flag == TT_LOWER && score >= beta) || (tt_flag == TT_UPPER && score < beta)) {
            tt_record.info.move = MOVE_NONE;
            tt_record.info.depth = (S8)depth;
            tt_record.info.flag = tt_flag;
            tt_record.info.score = score_to_tt(score, ply);
            tt_save(game->board.key, &tt_record);
            return score;
        }
    }
#endif 

    int eval_score = evaluate(game, beta - 1, beta);
    game->eval_hist[ply] = eval_score;
    int improving = ply > 1 && game->eval_hist[ply] > game->eval_hist[ply - 2];
    
    // Razoring
    // clear
    /*if (!incheck && depth < RAZOR_DEPTH && !is_mate_score(beta)) {
        if (eval_score + RAZOR_MARGIN[depth] < beta && !has_pawn_on_rank7(&game->board, turn)) {
            razor_beta = beta - RAZOR_MARGIN[depth];
            score = quiesce(game, FALSE, razor_beta - 1, razor_beta, 0);
            if (game->search.abort) return 0;
            if (score < razor_beta) return score;
        }
    }*/

    // Null move heuristic: side to move has advantage that even allowing an extra move to opponent, still keeps advantage.
    // clear
    //if (!incheck && !is_mate_score(beta) && has_pieces(&game->board, turn) && !has_recent_null_move(&game->board)) {
    //    // static null move
    //    if (depth < STAT_NULL_DEPTH && eval_score - STAT_NULL_MARGIN[depth] >= beta && improving) {
    //        return eval_score - STAT_NULL_MARGIN[depth];
    //    }
    //    // null move search
    //    if (depth >= 2 && eval_score >= beta) {
    //        int null_depth = depth - 4 - ((depth - 2) / 4) - MIN(3, (eval_score - beta) / 200);
    //        make_move(&game->board, NULL_MOVE);
    //        score = -search_zw(game, incheck, 1 - beta, null_depth);
    //        undo_move(&game->board);
    //        if (game->search.abort) return 0;
    //        if (score >= beta) {
    //            if (is_mate_score(score)) score = beta;
    //            return score;
    //        }
    //    }
    //}

    //  Prob-Cut: after a capture a low depth with reduced beta indicates it is safe to ignore this node
    // clear
    /*if (depth >= 5 && !incheck && !is_mate_score(beta)) {
        int beta_cut = beta + 100;
        MOVE_LIST mlpc;
        select_init(&mlpc, game, incheck, trans_move, TRUE);
        while ((move = next_move(&mlpc)) != MOVE_NONE) {
            if (move_is_quiet(move) || eval_score + see_move(&game->board, move) < beta_cut) continue;
            if (!is_pseudo_legal(&game->board, mlpc.pins, move)) continue;
            make_move(&game->board, move);
            score = -search_zw(game, is_incheck(&game->board, side_on_move(&game->board)), 1 - beta_cut, depth - 4);
            undo_move(&game->board);
            if (game->search.abort) return 0;
            if (score >= beta_cut) return score;
        }
    }*/

    // Reduction when position is not on transposition table. Idea from Prodeo chess engine (from Ed Schroder).
    // clear
    /*if (depth > 3 && trans_move == MOVE_NONE && !incheck) {
        depth--;
    }*/

    select_init(&ml, game, incheck, trans_move, FALSE);
    while ((move = next_move(&ml)) != MOVE_NONE) {

        assert(is_valid(&game->board, move));

        if (!is_pseudo_legal(&game->board, ml.pins, move)) continue;

        move_count++;

        int reductions = 0;
        int extensions = 0;

        gives_check = is_check(&game->board, move);
        
        // extension if move puts opponent in check
        if (gives_check && (depth < 4 || see_move(&game->board, move) >= 0)) {
            extensions = 1;
        }

        // pruning or depth reductions
        // clear
        //if (!extensions && move_count > 1 && move_is_quiet(move)) {
        //    if (!is_killer(&game->move_order, turn, ply, move)) {
        //        if (!is_counter_move(&game->move_order, flip_color(turn), get_last_move_made(&game->board), move)) {
        //            int move_has_bad_history = get_has_bad_history(&game->move_order, turn, move);
        //            // Move count pruning: prune late moves based on move count.
        //            if (move_has_bad_history && depth < 10 && !incheck) {
        //                int pruning_threshold = 4 + depth * 2;
        //                if (!improving) pruning_threshold = pruning_threshold - 3;
        //                if (move_count > pruning_threshold) continue;
        //            }
        //            // Futility pruning: eval + margin below beta. Uses beta cutoff history.
        //            if (depth < 10) {
        //                int pruning_margin = depth * (50 + get_pruning_margin(&game->move_order, turn, move));
        //                if (eval_score + pruning_margin < beta) continue;
        //            }
        //            // Late move reductions: reduce depth for later moves
        //            if (move_count > 3 && depth > 2) {
        //                reductions = reduction_table[MIN(depth, MAX_DEPTH - 1)][MIN(move_count, MAX_MOVE - 1)];
        //                if (move_has_bad_history || !improving || (incheck && unpack_piece(move) == KING)) reductions++;
        //            }
        //        }
        //    }
        //}

        // Make move and search new position.
        make_move(&game->board, move);

        assert(valid_is_legal(&game->board, move));

        // clear
        extensions = reductions = 0;

        score = -search_zw(game, gives_check, 1 - beta, depth - 1 + extensions - reductions);
        if (!game->search.abort && score >= beta && reductions > 0) {
            score = -search_zw(game, gives_check, 1 - beta, depth - 1);
        }
        
        undo_move(&game->board);
        if (game->search.abort) return 0;

        // score verification
        if (score > best_score) {
            if (score >= beta) {
                update_pv(&game->pv_line, ply, move);
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

    if (!EVAL_TUNING) {
        tt_record.info.move = MOVE_NONE;
        tt_record.info.depth = (S8)depth;
        tt_record.info.flag = TT_UPPER;
        tt_record.info.score = score_to_tt(best_score, ply);
        tt_save(game->board.key, &tt_record);
    }

    return best_score;
}

// end
