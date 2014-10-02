#ifndef COCOS_ALL_DIST__
#define COCOS_ALL_DIST__

#include "globals.h"

void all_distant ( real* beam_str, real* validity_solutions, int v_id,
                   int num_blocks, int num_threads, int n_bytes=0 );
void check_all_dist   ( real * local_point_list, int* check_success, int n_threads, int print_var=-1 );

#endif


