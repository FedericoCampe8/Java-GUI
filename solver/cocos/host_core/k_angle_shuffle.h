#ifndef COCOS_K_ANGLE_SHUFFLE__
#define COCOS_K_ANGLE_SHUFFLE__

#include "globals.h"

void k_angle_shuffle ( int* vars_to_shuffle, int n_vars_to_shuffle,
                       real* all_domains, int* all_domains_idx,
                       real* current_str, real* beam_str,
                       int n_blocks, int n_threads, int smBytes );


#endif


