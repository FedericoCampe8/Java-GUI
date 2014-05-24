#include "propagator.h"
#include "logic_variables.h"
#include "aminoacid.h"
#include "utilities.h"

#include "constraint.h"
#include "sang.h"
#include "all_distant.h"
#include "distance.h"
#include "centroid.h"
#include "mang.h"
#include "k_rang.h"
//#include "cuda_k_rang.h"
//#include "cuda_k_angle_shuffle.h"
//#include "cuda_utilities.h"

//#define PROP_DEBUG
//#define PROP_C_K_ANGLE_SHUFFLE_DEBUG
//#define PROP_C_DIST

/*******************
 ** CHECK FAILURE **
 *******************/

/// @note: | V |    == blockDim.x
/// @note: | D_aa | == gridDim.x
void
check_failure ( real* state, int* domain_events, int n_threads ) {
  int failed = 1;
  for (int threadIdx=0; threadIdx<n_threads; threadIdx++) {
    if ( state[ threadIdx ] > 0 ) {
      failed = 0;
      break;
    }
  }
  failed ? domain_events[ 0 ] = failed_event : domain_events[ 0 ] = events_size;
}//gpu_check


void
Propagator::prepare_init_set ( real* current_str, real* beam_str, real* states, int n_blocks ) {
  /// Set current_str in place of non valid structures
  for ( int blockIdx = 0; blockIdx < n_blocks; blockIdx++ ) {
    if ( states[ blockIdx ] == 0 ) {
      memcpy( &beam_str[ blockIdx * gh_params.n_res * 15 ], current_str, gh_params.n_res * 15 * sizeof( real ) );
    }
  }
}//prepare_init_set

void
Propagator::update_set ( real* current_beam, real* upd_beam, real* states, int n_blocks ) {
  for ( int blockIdx = 0; blockIdx < n_blocks; blockIdx++ ) {
    if ( states[ blockIdx ] > 0 ) {
      memcpy( &current_beam[ blockIdx * gh_params.n_res * 15 ],
              &upd_beam[ blockIdx * gh_params.n_res * 15 ], gh_params.n_res * 15 * sizeof( real ) );
    }
  }
}//update_set


void
Propagator::print_beam ( real* beam, int start, int end ) {
  for ( int i = start; i < end; i++ ) {
    cout << Utilities::output_pdb_format ( &beam[i*gh_params.n_points] ) << endl;
  }
}//print_beam

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
#ifdef PROP_DEBUG
  cout << "sang constraint V_" << v_id << endl;
  //getchar();
#endif
  
  AminoAcid* aa = g_logicvars.cp_variables[ v_id ];
  int n_blocks  = aa->get_domain_size();
  int n_threads = gh_params.n_res;
  int smBytes   = gh_params.n_points * sizeof( real );
  sang( gd_params.curr_str, gd_params.beam_str,
        gd_params.all_domains, gd_params.all_domains_idx,
        v_id, n_blocks, n_threads, smBytes );
  /*
  if (v_id == 15) {
    print_beam ( gd_params.beam_str, 0, n_blocks );
    getchar();
  }
   */
}//prop_c_sang
/*************************************************************/

/*********************** prop_c_k_rang ***********************/
void
Propagator::prop_c_k_rang ( int v_id, int c_id, int c_idx ) {
#ifdef PROP_DEBUG
  cout << "k_rang constraint V_" << v_id << endl;
  //getchar();
#endif
  
  k_rang( v_id,
          gd_params.all_domains,
          gd_params.all_domains_idx,
          gd_params.beam_str,
          gd_params.beam_str_upd,
          gh_params.set_size, gh_params.n_res,
          gh_params.n_points * sizeof( real )
        );
}//prop_c_k_rang
/*************************************************************/

/*********************** prop_c_mang ***********************/
void
Propagator::prop_c_mang ( int v_id, int c_id, int c_idx ) {
#ifdef PROP_DEBUG
  cout << "mang constraint V_" << v_id << endl;
  //getchar();
#endif
  
  int n_blocks  = 1;
  int n_threads = gh_params.n_res;
  int smBytes   = gh_params.n_points * sizeof( real );
  int n_vars    = gh_params.constraint_descriptions[ c_idx + 1 ];
  int n_coeff   = gh_params.constraint_descriptions[ c_idx + 2 ];
  int k_size    = gh_params.set_size;

  //AminoAcid* aa = g_logicvars.cp_variables[ v_id ];
  //k_size  = aa->get_domain_size();
  //n_threads = 32;
  mang( v_id, &gh_params.constraint_descriptions[ c_idx + 3 + n_vars ],
        n_coeff, gd_params.all_domains,
        gd_params.all_domains_idx,
        gd_params.curr_str,
        gd_params.beam_str,
        gh_params.n_res,
        k_size, n_threads, smBytes
      );
//  print_beam ( gd_params.beam_str, 349, 350 );
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
#ifdef PROP_DEBUG
  cout << "all_distant constraint V_" << v_id << endl;
  //getchar();
#endif
  
  int type_of_agent = gh_params.constraint_descriptions[ c_idx + 3 +
                                                         gh_params.constraint_descriptions[ c_idx + 1 ] +
                                                         gh_params.n_res + 1 ];
  int n_blocks;
  int n_threads = gh_params.n_res;
  int smBytes   = gh_params.n_points * sizeof(real);
  if ( type_of_agent == coordinator ) {
    n_blocks = gh_params.set_size;
    if ( gd_params.beam_str_upd == NULL ) {
      all_distant( gd_params.beam_str,
                   gd_params.validity_solutions,
                   v_id, n_blocks, n_threads, smBytes);
    }
    else {
      all_distant( gd_params.beam_str_upd,
                   gd_params.validity_solutions,
                   v_id, n_blocks, n_threads, smBytes);
    }
    
  }
  else {
    AminoAcid* aa = g_logicvars.cp_variables[ v_id ];
    n_blocks  = aa->get_domain_size();
    all_distant( gd_params.beam_str,
                 gd_params.validity_solutions,
                 v_id, n_blocks, n_threads, smBytes);
    
  }
  /// To optimize...
  check_failure(  gd_params.validity_solutions, gd_params.domain_events, n_blocks );
}//prop_c_all_dist
/*************************************************************/

/************************* prop_c_cg *************************/
void
Propagator::prop_c_cg ( int v_id, int c_id, int c_idx ) {
#ifdef PROP_DEBUG
  cout << "CG constraint V_" << v_id << endl;
  //getchar();
#endif
  
  int type_of_agent = gh_params.constraint_descriptions[ c_idx + 3 +
                                                         gh_params.constraint_descriptions[ c_idx + 1 ] +
                                                         gh_params.n_res + 1 ];
  int n_blocks;
  int n_threads = gh_params.n_res;
  int smBytes   = gh_params.n_points * sizeof(real);
  if ( type_of_agent == coordinator ) {
    n_blocks = gh_params.set_size;
    if ( gd_params.beam_str_upd == NULL ) {
      centroid ( gd_params.beam_str,
                 gd_params.validity_solutions,
                 gd_params.aa_seq,
                 v_id, n_blocks, n_threads, smBytes);
    }
    else {
      centroid ( gd_params.beam_str_upd,
                 gd_params.validity_solutions,
                 gd_params.aa_seq,
                 v_id, n_blocks, n_threads, smBytes);
    }
    
  }
  else {
    AminoAcid* aa = g_logicvars.cp_variables[ v_id ];
    n_blocks  = aa->get_domain_size();
    centroid ( gd_params.beam_str,
               gd_params.validity_solutions,
               gd_params.aa_seq,
               v_id, n_blocks, n_threads, smBytes);
    
  }
  /// To optimize...
  check_failure(  gd_params.validity_solutions, gd_params.domain_events, n_blocks );
  
}//prop_c_cg
/*************************************************************/

/************************* prop_c_dist *************************/
void
Propagator::prop_c_dist ( int v_id, int c_id, int c_idx ) {
  
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
  distance ( gd_params.beam_str_upd, gd_params.validity_solutions,
             &gh_params.constraint_descriptions[ c_idx + 3 +
                                                 gh_params.constraint_descriptions[ c_idx + 1 ] ],
             gh_params.n_res, n_blocks, n_threads );
}//prop_c_dist
/*************************************************************/

