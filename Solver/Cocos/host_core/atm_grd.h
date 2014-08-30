#ifndef COCOS_ATM_GRID__
#define COCOS_ATM_GRID__

#include "globals.h"

void attraction_atom ( real * local_point_list, real * attraction_points, int n_threads );

void atom_grd ( real* beam_str, real* validity_solutions, aminoacid * aa_seq,
                int v_id, int num_blocks, int num_threads, int n_bytes=0 );
void check_atom_grd    ( real * local_point_list, real attraction[3], real* check_success, int n_threads, int print_var=-1 );
void check_atom_grd_cg ( real * local_point_list, real attraction[3], real* check_success, aminoacid * aa_seq, int n_threads, int print_var=-1 );

#endif


