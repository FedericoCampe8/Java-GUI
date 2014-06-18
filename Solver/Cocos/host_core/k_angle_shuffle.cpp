#include "k_angle_shuffle.h"
#include "utilities.h"

//#define CUDA_K_ANGLE_SHUFFLE_DEBUG

using namespace std;

/// @note: | V |    == blockDim.x
/// @note: | D_aa | == gridDim.x
void
k_angle_shuffle ( int* vars_to_shuffle, int n_vars_to_shuffle,
                  real* all_domains, int* all_domains_idx,
                  real* current_str, real* beam_str,
                  int n_blocks, int n_threads, int smBytes ) {
  /*
   * @note: here all structures are valid
   * int warp = WHICHWARP( blockIdx.x );
   * if ( !(domain_states[ MAX_DIM * v_id + warp ] & ((uint) 1<<(blockIdx.x))) ) return;
   */
  real local_curr_str[ gh_params.n_res * 15 ];
  for ( int blockIdx = 0; blockIdx < n_blocks; blockIdx++ ) {
    /// Set initial structure
    memcpy ( local_curr_str, current_str, gh_params.n_res * 15 * sizeof( real ) );
    int v_id;
    real phi, psi;
    for ( int i = 0; i < n_vars_to_shuffle; i++ ) {
      v_id = vars_to_shuffle[ i ];
      int random_idx = all_domains_idx[ v_id ] +
                       (rand () % ((uint) all_domains[ all_domains_idx[ v_id ] ]) ) * 2;
      phi = (all_domains[ random_idx + 1 ] * PI_VAL)/180.0;
      psi = (all_domains[ random_idx + 2 ] * PI_VAL)/180.0;
      for ( int threadIdx = 0; threadIdx < gh_params.n_res; threadIdx++ ) {
        /// Move PHI angle
        if ( threadIdx <= v_id )
          Utilities::move_phi( local_curr_str, phi, v_id, ( v_id * 5 + 1 ) * 3, 0, threadIdx );
        /// Move PSI angle
        if ( threadIdx >= v_id )
          Utilities::move_psi( local_curr_str, psi, v_id, ( v_id * 5 + 1 ) * 3, gh_params.n_res, threadIdx );
      }
    }//n_vars_to_shuffle
    /// Copy back the rotated structure
    memcpy ( &beam_str[ blockIdx * gh_params.n_res * 15 ], local_curr_str, gh_params.n_res * 15 * sizeof( real ) );
  }//blockIdx
}//k_angle_shuffle



