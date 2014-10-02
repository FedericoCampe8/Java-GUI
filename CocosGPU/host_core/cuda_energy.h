/***************************************
 *                Energy               *
 ***************************************/
#ifndef COCOS_CUDA_ENERGY__
#define COCOS_CUDA_ENERGY__
 
#include "globals.h"

//void host_set_structure ( real* cp_points, real* cp_set_of_points, int n_points, int n, int n_res );
__global__ void cuda_set_structure ( real* cp_points, real* cp_set_of_points, int n_points, int n );

__global__ void cuda_energy ( real* beam_str, real* beam_energies,
                              real* validity_solutions,
                              ss_type * secondary_s_info,
                              real * h_distances, real* h_angles,
                              real * contact_params, aminoacid * aa_seq,
                              real * tors, real * tors_corr,
                              int bb_start, int bb_end, int n_res,
                              int scope_start, int scope_end,
                              agent_type a_type );
__device__ void hydrogen_energy ( real * structure, real * h_values,
                                  real * h_distances, real * h_angles,
                                  ss_type* secondary_s_info,
                                  int bb_start, int bb_end, int n_res );
__device__ void contact_energy ( real * structure, real * con_values,
                                 real * contact_params, aminoacid * aa_seq,
                                 int bb_start, int bb_end, int n_res );
__device__ void contact_energy_cg ( real * structure, real * contact_params, aminoacid * aa_seq,
                                    int first_cg_idx, int second_cg_idx, real* c_energy );
__device__ void correlation_energy ( real * structure, real * corr_val,
                                     real * tors, real * tors_corr, aminoacid * aa_seq,
                                     int bb_start, int bb_end, int n_res );
__device__ int get_h_distance_bin ( real distance );
__device__ int get_h_angle_bin ( real angle );
__device__ int get_corr_aa_type ( aminoacid );

#endif


