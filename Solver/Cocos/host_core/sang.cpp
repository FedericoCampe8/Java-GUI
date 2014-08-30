#include "sang.h"
#include "utilities.h"

//#define SANG_DEBUG

void
sang ( real* current_str, real* beam_str,
       real* all_domains, int* all_domains_idx,
       int v_id, int n_blocks, int n_threads, int smBytes ) {
  
  // @note: here all structures are valid
  real local_curr_str[ n_threads * 15 ];
  
  for ( int blockIdx = 0; blockIdx < n_blocks; blockIdx++ ) {
    real phi = (all_domains[ all_domains_idx[ v_id ] + blockIdx * 2 + 1 ] * PI_VAL)/180.0;
    real psi = (all_domains[ all_domains_idx[ v_id ] + blockIdx * 2 + 2 ] * PI_VAL)/180.0;
    //std::cout << "phi " << phi << " " << psi << " " << all_domains_idx[ v_id ]  << std::endl;
    int selected_var_ca_pos = ( v_id * 5 + 1 ) * 3;
    
    /// Set initial structure
    memcpy ( local_curr_str, current_str, n_threads * 15*sizeof(real) );
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
    memcpy ( &beam_str[ blockIdx * n_threads * 15 ],
            local_curr_str, n_threads * 15*sizeof(real) );
  }//blockIdx
}//sang
