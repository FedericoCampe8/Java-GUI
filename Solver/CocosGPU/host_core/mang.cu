#include "mang.h"
#include "utilities.h"

//#define MANG_DEBUG

/// @note: | V |    == blockDim.x
/// @note: | D_aa | == gridDim.x

void
mang ( int v_id,
       int* vars_to_shuffle, int n_vars_to_shuffle,
       real* all_domains, int* all_domains_idx,
       real* current_str, real* beam_str,
       int len_prot, int n_blocks, int n_threads, int smbytes ) {
  /*
   * @note: here all structures are valid
   * int warp = WHICHWARP( blockIdx.x );
   * if ( !(domain_states[ MAX_DIM * v_id + warp ] & ((uint) 1<<(blockIdx.x))) ) return;
   */
  real local_curr_str[ n_threads * 15 ];
  
  for ( int blockIdx = 0; blockIdx < n_blocks; blockIdx++ ) {
    /// Set initial structure
    memcpy ( local_curr_str, current_str, n_threads * 15*sizeof(real) );
    int max_d_size;
    real phi;
    real psi;
    max_d_size = (int) all_domains[ all_domains_idx[ v_id ] ];
    phi = (all_domains[ all_domains_idx[ v_id ] + (blockIdx % max_d_size) * 2 + 1 ] * PI_VAL)/180.0;
    psi = (all_domains[ all_domains_idx[ v_id ] + (blockIdx % max_d_size) * 2 + 2 ] * PI_VAL)/180.0;
    /// Rotate following domain as usual
    for ( int j = v_id; j >= 0; j-- )
       Utilities::move_phi( local_curr_str, phi, v_id, ( v_id * 5 + 1 ) * 3, 0, j );
    for ( int j = v_id; j < len_prot; j++ )
       Utilities::move_psi( local_curr_str, psi, v_id, ( v_id * 5 + 1 ) * 3, len_prot, j );
    /// Rotate moving random domains
    if (blockIdx >= max_d_size) {
      int v_id_aux;
      for ( int i = 0; i < n_vars_to_shuffle; i++ ) {
          v_id_aux = vars_to_shuffle[ i ];
          if ( v_id_aux == v_id ) continue;
          int random_idx = all_domains_idx[ v_id_aux ] +
          ( rand() % ((uint) all_domains[ all_domains_idx[ v_id_aux ] ]) ) * 2;
          
          phi = (all_domains[ random_idx + 1 ] * PI_VAL)/180.0;
          psi = (all_domains[ random_idx + 2 ] * PI_VAL)/180.0;


          for ( int j = v_id_aux; j >= 0; j-- )
            Utilities::move_phi( local_curr_str, phi, v_id_aux, ( v_id_aux * 5 + 1 ) * 3, 0, j );
          for ( int j = v_id_aux; j < len_prot; j++ )
            Utilities::move_psi( local_curr_str, psi, v_id_aux, ( v_id_aux * 5 + 1 ) * 3, len_prot, j );
      }//i
    }
    /// Copy back the rotated structure
    memcpy ( &beam_str[ blockIdx * n_threads * 15 ],
             local_curr_str, n_threads * 15*sizeof(real) );
  }
}//mang

