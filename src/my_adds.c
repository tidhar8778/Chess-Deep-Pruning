#include "globals.h"
#define MAXLINES  10
#define MAXCHARSPERLINE 80
#define DATASETMAXLINESFENS 1000000

char res_file[] = "C:\\Users\\User\\Documents\\tidhar\\barilan\\DeepLearning\\thesis\\chess\\tucano\\Tucano\\test result\\res_special.txt";
char test_file[] = "C:\\Users\\User\\Documents\\tidhar\\barilan\\DeepLearning\\thesis\\chess\\tucano\\Tucano\\test result\\test_special.txt";
const char filename_fen_test[] = "C:\\Users\\User\\Documents\\tidhar\\barilan\\DeepLearning\\thesis\\chess\\tucano\\Tucano\\data\\test\\fens example.txt";
char dataset_file[] = "C:\\Users\\User\\Documents\\tidhar\\barilan\\DeepLearning\\thesis\\chess\\tucano\\Tucano\\data\\train_model\\fen_labels.txt";
char best_move_deep[MAXLINES][10];
char best_move_regular[MAXLINES][10];
char best_move_base[MAXLINES][10];
double evaluation_deep[MAXLINES];
double evaluation_regular[MAXLINES];
double evaluation_base[MAXLINES];
double time_deep[MAXLINES];
double time_regular[MAXLINES];
double time_base[MAXLINES];
int depth_deep[MAXLINES];
int depth_regular[MAXLINES];
int depth_base[MAXLINES];
double cutted_percent[MAXLINES];


void data_print_headline()
{
    FILE* fptr;
    fptr = fopen(res_file, "r");
    if (!fptr)
    {
        fptr = fopen(res_file, "w");
        fprintf(fptr, "Ply\tNodes visited\tNodes deep-cut\tNodes checked cutting\tEvaluation\tBest move\tIs black?\tTime (sec)\tNodes per second\tDepth reached\tUsing bpruning?\tFEN\t");

    }
    fclose(fptr);
}


set_game_settings_max_nodes(SETTINGS* my_settings)
{
    my_settings->max_nodes = 1000000;
}

set_game_settings_more_nodes(SETTINGS* my_settings)
{
    my_settings->max_nodes = 10000000;
}

set_game_settings_normal_move_time(SETTINGS* my_settings)
{
    my_settings->single_move_time = 1000 * 60 * 90;
}


set_game_search_bpruning(GAME* game, int bpruning)
{
    game->search.use_bpruning = bpruning;
}


set_settings_extarct_data(GAME* game, int extract_data)
{
    game->search.extract_data = extract_data;
}


add_datarow_to_dataset(char* fen, double beta, int label)
{
    FILE* fptr;
    fptr = fopen(dataset_file, "r");
    if (!fptr)
    {
        fptr = fopen(dataset_file, "w");
        fprintf(fptr, "FEN\tbeta\tlabel (1-cut, 0-no)\ttime elapsed");

    }
    fclose(fptr);
    
    if (fopen_s(&fptr, dataset_file, "a") == 0)
    {
        if (fptr != 0)
        {
            fprintf(fptr, "\n%s\t%.2lf\t%d\t%.3lf", fen, beta, label, main_game.search.elapsed_time / 1000.0);
        }
    fclose(fptr);

    main_game.search.abort = TRUE;
    }
}


void data_print_search(char* fen, int game_number)
{
    char best_move_string[10];
    util_get_move_string(main_game.search.best_move, best_move_string);
    // Evaluation score baseline is 200 centipawns, so we have to adjust it for display.
    double scoreqq = main_game.search.best_score;
    if (is_eval_score(main_game.search.best_score)) scoreqq /= 2;

    // tucano formatted output 
    U64 total_node_count = main_game.search.nodes + get_additional_threads_nodes();
    double score_display = scoreqq / 100.0;
    if (side_on_move(&main_game.board) == WHITE) score_display = -score_display; // Beacuse current side_on_move is the opposite of the played move side_on_move
    double time_display = ((float)main_game.search.elapsed_time / 1000.0);

    FILE* fptr;
    if (fopen_s(&fptr, res_file, "a") == 0)
    {
        if (fptr != 0)
        {
            fprintf(fptr, "\n%u\t%" PRIu64 "\t%" PRIu64 "\t%" PRIu64 "\t%4.2f\t%s\t%u\t%10.2f\t%7.0lf\t%d\t%d\t%s",
                main_game.board.histply, total_node_count, main_game.search.nodes_cutted_b,
                main_game.search.nodes_checked_cutting_b, score_display, best_move_string,
                !main_game.board.side_on_move, time_display, main_game.search.nodes / time_display,
                main_game.search.cur_depth, main_game.search.use_bpruning, fen);
        }
        fclose(fptr);
    }

    // save data
    if (main_game.search.use_bpruning == 1)
    {
        strcpy(best_move_deep[game_number], best_move_string);
        evaluation_deep[game_number] = score_display;
        time_deep[game_number] = time_display;
        depth_deep[game_number] = main_game.search.cur_depth;
        if (main_game.search.nodes_checked_cutting_b != 0)
        {
            cutted_percent[game_number] = main_game.search.nodes_cutted_b / (double)main_game.search.nodes_checked_cutting_b;
        }
        else
        {
            cutted_percent[game_number] = 0.0;
        }
    }

    else if (main_game.search.use_bpruning == 0)
    {
        strcpy(best_move_regular[game_number], best_move_string);
        evaluation_regular[game_number] = score_display;
        time_regular[game_number] = time_display;
        depth_regular[game_number] = main_game.search.cur_depth;
    }

    else if (main_game.search.use_bpruning == -1)
    {
        strcpy(best_move_base[game_number], best_move_string);
        evaluation_base[game_number] = score_display;
        time_base[game_number] = time_display;
        depth_base[game_number] = main_game.search.cur_depth;
    }
        
}


void read_fen_test(char* fenst[MAXLINES][MAXCHARSPERLINE], const char filename[])
{
    // Open File
    int nlines = 0;

    FILE* f;
    int n;
    char buf[MAXCHARSPERLINE];
    if ((f = fopen(filename, "r")) == NULL)
    {
        fprintf(stderr, "%s: can't open for reading\n", filename);
        return;
    }
    for (nlines = 0; fgets(buf, MAXCHARSPERLINE, f) != NULL; nlines++) {
        n = strlen(buf);
        if (n <= 0 || buf[n - 1] != '\n')
            fprintf(stderr, "%s:%d read partial line, increase MAXCHARSPERLINE\n", filename, nlines + 1);
        buf[n - 1] = '\0';
        if (nlines >= MAXLINES)
            fprintf(stderr, "%s:%d increase MAXLINES\n", filename, nlines + 1);
        strcpy(fenst[nlines], buf);
    }
    fclose(f);
       
    for (int i = 0; i < nlines; i++)
        printf("%d: %s\n", i + 1, fenst[i]);

}


void run_fens_test()
{   
    // This function runs with 'test' command

    char* fens[MAXLINES][MAXCHARSPERLINE];
    read_fen_test(fens, filename_fen_test);
    int num_of_fens = MAXLINES;
    
    data_print_headline();
    for (int i = 0; i < num_of_fens; i++)
    {
        set_game_settings_normal_move_time(&game_settings);
        set_settings_extarct_data(&main_game, 0);
        
        // run for base results (more nodes)
        set_game_settings_more_nodes(&game_settings);
        set_fen(&main_game.board, fens[i]);
        set_game_search_bpruning(&main_game, -1);
        search_run(&main_game, &game_settings);
        // make and print best move found
        make_move(&main_game.board, main_game.search.best_move);
        data_print_search(fens[i], i);

        // run regular
        set_game_settings_max_nodes(&game_settings);
        set_fen(&main_game.board, fens[i]);
        set_game_search_bpruning(&main_game, 0);
        search_run(&main_game, &game_settings);
        // make and print best move found
        make_move(&main_game.board, main_game.search.best_move);
        data_print_search(fens[i], i);

        // run again with bpruning
        set_game_settings_max_nodes(&game_settings);
        set_fen(&main_game.board, fens[i]);
        set_game_search_bpruning(&main_game, 1);
        search_run(&main_game, &game_settings);
        // make and print best move found
        make_move(&main_game.board, main_game.search.best_move);
        data_print_search(fens[i], i);

    }

    // analyze
    int equal_move_regular = 0;
    int equal_move_deep = 0;
    double eval_diff_regular = 0.0;
    double eval_diff_deep = 0.0;
    int total_depth_deep = 0;
    int total_depth_regular = 0;
    int total_depth_base = 0;
    double total_time_deep = 0.0;
    double total_time_regular = 0.0;
    double total_time_base = 0.0;
    double total_cutted_percent = 0.0;
    for (int i = 0; i < num_of_fens; i++)
    {
        if (strncmp(best_move_regular[i], best_move_base[i], sizeof(best_move_base[i])) == 0)
        {
            equal_move_regular++;
        }
        else
        {
            eval_diff_regular += fabs(evaluation_regular[i] - evaluation_base[i]);
        }

        if (strncmp(best_move_deep[i], best_move_base[i], sizeof(best_move_base[i])) == 0)
        {
            equal_move_deep++;
        }
        else
        {
            eval_diff_deep += fabs(evaluation_deep[i] - evaluation_base[i]);
        }

        total_depth_deep += depth_deep[i];
        total_depth_regular += depth_regular[i];
        total_depth_base += depth_base[i];
        total_time_deep += time_deep[i];
        total_time_regular += time_regular[i];
        total_time_base += time_base[i];
        total_cutted_percent += cutted_percent[i];
    }

    double percent_equal_regular = equal_move_regular / (double)num_of_fens;
    double percent_equal_deep = equal_move_deep / (double)num_of_fens;
    double avg_eval_diff_regular = eval_diff_regular / (num_of_fens - equal_move_regular);
    double avg_eval_diff_deep = eval_diff_deep / (num_of_fens - equal_move_deep);
    int avg_depth_deep = total_depth_deep / num_of_fens;
    int avg_depth_regular = total_depth_regular / num_of_fens;
    int avg_depth_base = total_depth_base / num_of_fens;
    double avg_time_deep = total_time_deep / num_of_fens;
    double avg_time_regular = total_time_regular / num_of_fens;
    double avg_time_base = total_time_base / num_of_fens;
    double avg_cutted_percent = total_cutted_percent / num_of_fens;

    printf("Regular equal percentage: %2.0f%%\nRegular evaluation difference vs BASE when not equal move: %5.2f\nDeep equal percentage: %2.0f%%\nDeep evaluation difference vs BASE when not equal move: %5.2f\nAverage depth:\nDeep - %d\nRegular - %d\nBase - %d\nAverage time Deep (seconds):\nDeep - %5.1f\nRegular - %5.1f\nBase - %5.1f\nAverage cutting nodes percentage: %2.0f%%\n", 
        percent_equal_regular*100, avg_eval_diff_regular, percent_equal_deep*100, avg_eval_diff_deep, avg_depth_deep, avg_depth_regular, avg_depth_base, avg_time_deep, avg_time_regular, avg_time_base, avg_cutted_percent*100);

    FILE* fptr;
    if (fopen_s(&fptr, test_file, "w") == 0)
    {
        if (fptr != 0)
        {
            fprintf(fptr, "Regular equal percentage: %2.0f%%\nRegular evaluation difference vs BASE when not equal move: %5.2f\nDeep equal percentage: %2.0f%%\nDeep evaluation difference vs BASE when not equal move: %5.2f\nAverage depth:\nDeep - %d\nRegular - %d\nBase - %d\nAverage time Deep (seconds):\nDeep - %5.1f\nRegular - %5.1f\nBase - %5.1f\nAverage cutting nodes percentage: %2.0f%%\n",
                percent_equal_regular * 100, avg_eval_diff_regular, percent_equal_deep * 100, avg_eval_diff_deep, avg_depth_deep, avg_depth_regular, avg_depth_base, avg_time_deep, avg_time_regular, avg_time_base, avg_cutted_percent * 100);
                
        }
        fclose(fptr);
    }

}


void create_dataset()
{
    // This function runs with 'dataset' command
    extern count_visit;
    extern count_visit_cutoff;
    const char filename[] = "C:\\Users\\User\\Documents\\tidhar\\barilan\\DeepLearning\\thesis\\chess\\tucano\\Tucano\\data\\train_model\\fens.txt"; // File consists 1 miliion of fens from different games
    int num_of_fens = DATASETMAXLINESFENS;
    
    // read fens    
    FILE* file = fopen(filename, "r");
    if (file == NULL) {
        // Handle error.
        return 1;
    }

    char** fens = malloc(num_of_fens * sizeof(char*));
    for (int i = 0; i < num_of_fens; i++) {
        fens[i] = malloc(MAXCHARSPERLINE * sizeof(char));
    }
    int line_count = 0;
    while (fgets(fens[line_count++], MAXCHARSPERLINE, file) != NULL) {
        if (line_count >= DATASETMAXLINESFENS) break;
    }
    // Close the file.
    fclose(file);

    set_game_search_bpruning(&main_game, 0);
    set_settings_extarct_data(&main_game, 1);
    game_settings.post_flag = POST_NONE;
    for (int i = 0; i < num_of_fens; i++)
    {
        // run search and save data
        set_fen(&main_game.board, fens[i]);
        search_run(&main_game, &game_settings);
        printf("%d %d %d %f\n", i + 1, count_visit_cutoff, count_visit, count_visit_cutoff/(double)count_visit);
        
        count_visit = 0;
        count_visit_cutoff = 0;

    }

    
    // Free the memory.
    for (int i = 0; i < num_of_fens; i++) {
        free(fens[i]);
    }
    free(fens);
}