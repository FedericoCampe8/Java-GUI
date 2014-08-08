#include "propagator.h"
#include "logic_variables.h"
#include "aminoacid.h"
#include "utilities.h"

#include "constraint.h"
#include "distance.h"
#include "mang.h"
#include "cuda_propagators.h"

//#include "cuda_utilities.h"

//#define PROP_C_K_ANGLE_SHUFFLE_DEBUG
//#define PROP_C_DIST

/*******************
 **  PROPAGATORS  **
 *******************/

/********************** prop_c_sang **********************/
/*
 * prop_c_sang ( aa, D_aa, str, beam_str )
 * Rotate the aa^th amino acid of str with all the (valid) elements (i.e., angles) in D_x
 */
void
Propagator::prop_c_sang ( int v_id, int c_id, int c_idx ) {
  AminoAcid* aa = g_logicvars.cp_variables[ v_id ];
  int n_blocks  = aa->get_domain_size();
  int n_threads = gh_params.n_res;
  int smBytes   = gh_params.n_points * sizeof( real );
  
  cuda_c_sang<<< n_blocks, n_threads, smBytes >>>
  ( gd_params.curr_str, gd_params.beam_str,
    gd_params.all_domains, gd_params.all_domains_idx,
    v_id
   );
}//prop_c_sang
/*************************************************************/

/****************** prop_c_k_angle_shuffle ******************/
void
Propagator::prop_c_k_angle_shuffle ( int v_id, int c_id, int c_idx ) {
  /*
  int n_blocks  = 1;
  //int n_threads = max ( 32, gh_params.n_res ); n_threads = 32;
  int n_threads = 32;
  int* vars_to_shuffle;
  int smBytes  = gh_params.n_points * sizeof( real );
  int n_vars = gh_params.constraint_descriptions[ c_idx + 1 ];
  int k_size = gh_params.constraint_descriptions[ c_idx + 3 +
                                                  gh_params.constraint_descriptions[ c_idx + 1 ]
                                                 ];
  
  HANDLE_ERROR( cudaMalloc( ( void** )&vars_to_shuffle, n_vars * sizeof( int ) ) );
  HANDLE_ERROR( cudaMemcpy( vars_to_shuffle, &gh_params.constraint_descriptions[ c_idx + 3 ],
                            n_vars * sizeof( int ), cudaMemcpyHostToDevice ) );
  /// Init random numbers
  while ( n_blocks * MAX_N_THREADS < k_size ) n_blocks++;
  init_random<<<n_blocks, MAX_N_THREADS>>>( gd_params.random_state, time( NULL ) );

#ifdef PROP_C_K_ANGLE_SHUFFLE_DEBUG
  cout << " k_size (set size): " << k_size << " and n. threads: " << n_threads <<
  " var to shuffle " << n_vars << endl;
  cout << " smBytes " << smBytes << endl;
#endif

  cuda_k_angle_shuffle<<<k_size, n_threads, smBytes >>> ( vars_to_shuffle,
                                                          gd_params.all_domains, gd_params.all_domains_idx,
                                                          gd_params.curr_str, gd_params.beam_str,
                                                          gd_params.random_state,
                                                          n_vars, gh_params.n_res );
  /// Set fix point for this constraint
  g_constraints[ c_id ]->set_fix ();
  HANDLE_ERROR( cudaFree( vars_to_shuffle ) );
    */
}//prop_c_k_angle_shuffle
/*************************************************************/

/*********************** prop_c_k_rang ***********************/
void
Propagator::prop_c_k_rang ( int v_id, int c_id, int c_idx ) {
  int n_blocks  = 1;
  int n_threads = gh_params.n_res;
  int smBytes   = gh_params.n_points * sizeof( real );
  int k_size    = gh_params.set_size;
  
  /// Init random numbers
  while ( n_blocks * MAX_N_THREADS < k_size ) n_blocks++;
  init_random<<< n_blocks, MAX_N_THREADS >>>
  ( gd_params.random_state, time( NULL ) );
  cuda_k_rang<<< k_size, n_threads, smBytes >>>
  ( v_id,
    gd_params.all_domains,
    gd_params.all_domains_idx,
    gd_params.beam_str,
    gd_params.beam_str_upd,
    gd_params.random_state
   );
}//prop_c_k_rang
/*************************************************************/

/*********************** prop_c_mang ***********************/
void
Propagator::prop_c_mang ( int v_id, int c_id, int c_idx ) {
  int n_blocks     = gh_params.set_size;
  int n_threads    = 32;
  int smBytes      = gh_params.n_points * sizeof( real );
  int n_coeff      = gh_params.constraint_descriptions[ c_idx + 2 ];
  AminoAcid* aa    = g_logicvars.cp_variables[ v_id ];
  int k_size       = aa->get_domain_size();
  int ran_set_size = (n_blocks - k_size) * n_coeff; //-1

  ///Init random numbers
  for ( int i = 0; i < ran_set_size; i++ ) gh_params.random_array[ i ] = rand() % 1000;
  HANDLE_ERROR( cudaMemcpyAsync( gd_params.random_array, gh_params.random_array,
                                 ran_set_size * sizeof( int ), cudaMemcpyHostToDevice ) );
  cuda_c_mang<<< n_blocks, n_threads, smBytes >>>
  ( gd_params.curr_str, gd_params.beam_str,
    gd_params.all_domains, gd_params.all_domains_idx,
    gd_params.vars_to_shuffle, gd_params.random_array,
    n_coeff, v_id, k_size, gh_params.n_res
   );
}//prop_c_mang
/*************************************************************/


/********************** prop_c_all_dist **********************/
/*
 * prop_c_all_dist ( beam_str, min_distances )
 * Check that each pair of points i, j in the lists of points contained in beam_str
 * has Euclidean distance >= min_distances[ i, j ].
 */
void
Propagator::prop_c_all_dist ( int v_id, int c_id, int c_idx ) {
  int type_of_agent = gh_params.constraint_descriptions[ c_idx + 3 +
                                                         gh_params.constraint_descriptions[ c_idx + 1 ] +
                                                         gh_params.n_res + 1 ];
  int n_blocks;
  int n_threads = gh_params.n_res;
  int smBytes   = gh_params.n_points * sizeof(real);
  if ( type_of_agent == coordinator ) {
    /// Check consistency for solutions
    n_blocks = gh_params.set_size;
    if ( gd_params.beam_str_upd == NULL ) {
      cuda_all_distant<<< n_blocks, n_threads, smBytes >>>
      ( gd_params.beam_str, gd_params.validity_solutions );
    }
    else {
      cuda_all_distant<<< n_blocks, n_threads, smBytes >>>
      ( gd_params.beam_str_upd, gd_params.validity_solutions );
    }
  }
  else {
    AminoAcid* aa = g_logicvars.cp_variables[ v_id ];
    n_blocks  = aa->get_domain_size();
    cuda_all_distant<<< n_blocks, n_threads, smBytes >>>
    ( gd_params.beam_str, gd_params.validity_solutions );
  }
  /// To optimize...
  cuda_check_failure<<< 1, n_blocks >>>
  (
    gd_params.validity_solutions,
    gd_params.domain_events
  );
}//prop_c_all_dist
/*************************************************************/

/************************* prop_c_cg *************************/
void
Propagator::prop_c_cg ( int v_id, int c_id, int c_idx ) {
  int type_of_agent = gh_params.constraint_descriptions[ c_idx + 3 +
                                                        gh_params.constraint_descriptions[ c_idx + 1 ] +
                                                        gh_params.n_res + 1 ];
  int n_blocks;
  int n_threads = gh_params.n_res-2;
  int smBytes   = gh_params.n_points * sizeof(real);
  if ( type_of_agent == coordinator ) {
    /// Check consistency for solutions
    n_blocks = gh_params.set_size;
    if ( gd_params.beam_str_upd == NULL ) {
      cuda_centroid<<< n_blocks, n_threads, smBytes >>>
      ( gd_params.beam_str, gd_params.validity_solutions, gd_params.aa_seq );
    }
    else {
      cuda_centroid<<< n_blocks, n_threads, smBytes >>>
      ( gd_params.beam_str_upd, gd_params.validity_solutions, gd_params.aa_seq);
    }
  }
  else {
    AminoAcid* aa = g_logicvars.cp_variables[ v_id ];
    n_blocks  = aa->get_domain_size();
    cuda_centroid<<< n_blocks, n_threads, smBytes >>>
    ( gd_params.beam_str, gd_params.validity_solutions, gd_params.aa_seq );
  }
  /// To optimize...
  cuda_check_failure<<< 1, n_blocks >>>
  (
   gd_params.validity_solutions,
   gd_params.domain_events
   );
}//prop_c_cg
/*************************************************************/

/************************* prop_c_dist *************************/
void
Propagator::prop_c_dist ( int v_id, int c_id, int c_idx ) {
  int type_of_agent = gh_params.constraint_descriptions[ c_idx + 3 +
                                                         gh_params.constraint_descriptions[ c_idx + 1 ] +
                                                         gh_params.n_res + 1 ];
  if ( type_of_agent != coordinator ) return;
  
#ifdef PROP_C_DIST
  cout << "Dist constraint V_" << v_id << endl;
  for ( int i = 0; i < gh_params.constraint_descriptions[ c_idx + 2 ] / 3; i++ ) {
    cout <<
    gh_params.constraint_descriptions[ c_idx + 3 +
                                      gh_params.constraint_descriptions[ c_idx + 1 ] + 3i + 0 ]
    << " <-> " <<
    gh_params.constraint_descriptions[ c_idx + 3 +
                                      gh_params.constraint_descriptions[ c_idx + 1 ] + 3i + 1 ]
    << " d: " <<
    gh_params.constraint_descriptions[ c_idx + 3 +
                                      gh_params.constraint_descriptions[ c_idx + 1 ] + 3i + 2 ]
    << endl;
  }
  getchar();
#endif
  
  /// We assume that dist constraints are always propagated/checked by coordinator agents
  int n_blocks  = gh_params.set_size;
  int n_threads = gh_params.constraint_descriptions[ c_idx + 2 ] / 3;
  
  int * my_des;
  HANDLE_ERROR( cudaMalloc( ( void** )&my_des, 3 * n_threads * sizeof( int ) ) );
  HANDLE_ERROR( cudaMemcpy( my_des, &gh_params.constraint_descriptions[ c_idx + 3 + gh_params.constraint_descriptions[ c_idx + 1 ] ],
                            3 * n_threads * sizeof( int ), cudaMemcpyHostToDevice ) );
  
  cuda_distance <<< n_blocks, n_threads >>>
  ( gd_params.beam_str_upd, gd_params.validity_solutions, my_des, gh_params.n_res );
  
  HANDLE_ERROR( cudaFree( my_des ) );
}//prop_c_dist
/*************************************************************/

