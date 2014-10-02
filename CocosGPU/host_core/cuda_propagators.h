#ifndef COCOS_CUDA_PROPAGATORS__
#define COCOS_CUDA_PROPAGATORS__

#include "globals.h"
__global__ void init_random ( curandState *state, unsigned long seed );
__global__ void cuda_prepare_init_set ( real* current_str,
                                        real* beam_str, real* validity_solutions,
                                        int best_label );
__global__ void cuda_update_set ( real* beam_str, real* beam_str_upd, real* validity_solutions );
/// sang
__global__ void cuda_c_sang ( real* current_str, real* beam_str,
                              real* all_domains, int* all_domains_idx, int v_id );
/// mang
__global__ void cuda_c_mang ( real * current_str, real * beam_str,
                              real * all_domains, int * all_domains_idx,
                              int * vars_to_shuffle, int * random_vals,
                              int n_vars_to_shuffle, int v_id, int dom_size,
                              int len_prot );

/// alldistant
__global__ void cuda_all_distant ( real* beam_str, real* validity_solutions );
__device__ void check_all_dist   ( real * local_point_list, int* check_success );
/// cg
__global__ void cuda_centroid  ( real* beam_str, real* validity_solutions, aminoacid * aa_seq    );
__device__ void check_centroid ( real * local_point_list, int* check_success, aminoacid * aa_seq );

__global__ void cuda_check_failure ( real* domain_states, int* events );

/// k_angle_shuffle
__global__ void cuda_k_angle_shuffle ( int* vars_to_shuffle, 
                                       real* all_domains, int* all_domains_idx,
                                       real* current_str, real* beam_str,
                                       curandState *random_vals,
                                       int n_vars_to_shuffle, int len_prot );
/// k_rang
__global__ void cuda_k_rang ( int v_id,
                              real* all_domains, int* all_domains_idx,
                              real* beam_str, real* beam_str_upd,
                              curandState * random_vals );

/// Dist
__global__ void cuda_distance ( real* beam_str, real* validity_solutions, int* cons_descriptions, int len_prot );

#endif
