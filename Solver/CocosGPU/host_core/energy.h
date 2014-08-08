/***************************************
 *                Energy               *
 ***************************************/
#ifndef COCOS_ENERGY__
#define COCOS_ENERGY__
 
#include "globals.h"

void host_set_structure ( real* cp_points, real* cp_set_of_points, int n_points, int n, int n_res );

void get_energy ( real* beam_str, real* beam_energies,
                  real* validity_solutions,
                  ss_type* secondary_s_info,
                  real* h_distances, real* h_angles,
                  real * contact_params, aminoacid * aa_seq,
                  real * tors, real * tors_corr,
                  real hydrogen_w, real contact_w, real correlation_w,
                  int bb_start, int bb_end, int n_res, int scope_start, int scope_end,
                  int n_bytes, int n_blocks, int n_threads );

void hydrogen_energy ( real * structure, real * h_values,
                      real * h_distances, real * h_angles,
                      ss_type* secondary_s_info,
                      int bb_start, int bb_end, int n_res, int threadIdx );

void contact_energy ( real * structure, real * con_values,
                     real * contact_params, aminoacid * aa_seq,
                     int bb_start, int bb_end, int n_res, int threadIdx );

void contact_energy_cg ( real * structure, real * contact_params, aminoacid * aa_seq,
                        int first_cg_idx, int second_cg_idx, real* c_energy );

void correlation_energy ( real * structure, real * corr_val,
                         real * tors, real * tors_corr, aminoacid * aa_seq,
                         int bb_start, int bb_end, int n_res, int v_id=0 );

int get_h_distance_bin ( real distance );
int get_h_angle_bin ( real angle );
int get_corr_aa_type ( aminoacid );


#endif


