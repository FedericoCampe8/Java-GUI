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
  // @note: here all structures are valid
  real phi, psi;
  real local_curr_str[ n_threads * 15 ];
  int v_id_aux, selected_var_ca_pos_aux;
  int selected_var_ca_pos = ( v_id * 5 + 1 ) * 3;
  int max_d_size = (int) all_domains[ all_domains_idx[ v_id ] ];
  for ( int blockIdx = 0; blockIdx < n_blocks; blockIdx++ ) {
    /// Set initial structure
    memcpy ( local_curr_str, current_str, n_threads * 15 * sizeof(real) );
    phi = (all_domains[ all_domains_idx[ v_id ] + (blockIdx % max_d_size) * 2 + 1 ] * PI_VAL)/180.0;
    psi = (all_domains[ all_domains_idx[ v_id ] + (blockIdx % max_d_size) * 2 + 2 ] * PI_VAL)/180.0;
    /// Rotate structure
    for ( int threadIdx = 0; threadIdx < n_threads; threadIdx++ ) {
      /// Move PHI angle
      if ( threadIdx <= v_id ) {
        //std::cout << "Rotate PHI about " << v_id << " with " << threadIdx << std::endl;
        Utilities::move_phi( local_curr_str, phi, v_id, selected_var_ca_pos, 0, threadIdx );
      }
      /// Move PSI angle
      if ( threadIdx >= v_id ) {
        //std::cout << "Rotate PSI about " << v_id << " with " << threadIdx << std::endl;
        Utilities::move_psi( local_curr_str, psi, v_id, selected_var_ca_pos, n_threads, threadIdx );
      }
    }
    
    
    /// Rotate moving random angles
    if ( blockIdx >= max_d_size ) {
      for ( int i = 0; i < n_vars_to_shuffle; i++ ) {
        
        v_id_aux = vars_to_shuffle[ i ];
        selected_var_ca_pos_aux = ( v_id_aux * 5 + 1 ) * 3;
        if ( v_id_aux == v_id ) continue;
        /// Selecting a random value from variable's domain
        int random_idx = all_domains_idx[ v_id_aux ] +
        ( rand() % ((uint) all_domains[ all_domains_idx[ v_id_aux ] ]) ) * 2;
        
        phi = (all_domains[ random_idx + 1 ] * PI_VAL)/180.0;
        psi = (all_domains[ random_idx + 2 ] * PI_VAL)/180.0;
        /// Rotate structure with random angles
        for ( int threadIdx = 0; threadIdx < n_threads; threadIdx++ ) {
          /// Move PHI angle
          if ( threadIdx <= v_id_aux ) {
            Utilities::move_phi( local_curr_str, phi, v_id_aux, selected_var_ca_pos_aux, 0, threadIdx );
          }
          /// Move PSI angle
          if ( threadIdx >= v_id_aux ) {
            Utilities::move_psi( local_curr_str, psi, v_id_aux, selected_var_ca_pos_aux, n_threads, threadIdx );
          }
        }
        
      }//i
    }
    
    
    /// Copy back the rotated structure
    memcpy ( &beam_str[ blockIdx * n_threads * 15 ],
            local_curr_str, n_threads * 15*sizeof(real) );
  }
}//mang

