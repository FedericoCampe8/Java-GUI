#ifndef COCOS_CENTORID__
#define COCOS_CENTORID__

#include "globals.h"

void centroid ( real* beam_str, real* validity_solutions,
                aminoacid * aa_seq,
                int v_id, int n_blocks, int n_threads, int n_bytes=0 );
void check_centroid   ( real * local_point_list, int* check_success, int n_threads,
                        aminoacid * aa_seq, int print_failed_var=-1 );

#endif


