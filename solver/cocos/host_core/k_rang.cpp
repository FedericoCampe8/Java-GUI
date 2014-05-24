#include "k_rang.h"
#include "utilities.h"

//#define CUDA_K_RANG_DEBUG

/// @note: | V |    == blockDim.x
/// @note: | D_aa | == gridDim.x

void
k_rang ( int v_id,
         real* all_domains, int* all_domains_idx,
         real* beam_str, real* beam_str_upd,
         int n_blocks, int n_threads, int smBytes ) {
  /*
   * @note: here all structures are valid
   * int warp = WHICHWARP( blockIdx.x );
   * if ( !(domain_states[ MAX_DIM * v_id + warp ] & ((uint) 1<<(blockIdx.x))) ) return;
   */
  real local_curr_str[ n_threads * 15 ];
  real phi, psi;
  int selected_var_ca_pos = ( v_id * 5 + 1 ) * 3;
  for ( int blockIdx = 0; blockIdx < n_blocks; blockIdx++ ) {
    /// Set initial structure
    memcpy ( local_curr_str, &beam_str[ blockIdx * n_threads * 15 ], n_threads * 15 * sizeof(real) );
    int random_idx = all_domains_idx[ v_id ] +
                     ( rand() % ((uint) all_domains[ all_domains_idx[ v_id ] ]) ) * 2;
    phi = (all_domains[ random_idx + 1 ] * PI_VAL)/180.0;
    psi = (all_domains[ random_idx + 2 ] * PI_VAL)/180.0;
    
    /// Rotate structure
    for ( int threadIdx = 0; threadIdx < n_threads; threadIdx++ ) {
      /// Move PHI angle
      if ( threadIdx <= v_id )
        Utilities::move_phi( local_curr_str, phi, v_id, selected_var_ca_pos, 0, threadIdx );
      /// Move PSI angle
      if ( threadIdx >= v_id )
        Utilities::move_psi( local_curr_str, psi, v_id, selected_var_ca_pos, n_threads, threadIdx );
    }
    /// Copy back the rotated structure
    memcpy ( &beam_str_upd[ blockIdx * n_threads * 15 ],
             local_curr_str, n_threads * 15 * sizeof(real) );
  }//blockIdx
}//k_rang



