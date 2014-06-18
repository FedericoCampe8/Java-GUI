#include "all_distant.h"
#include "utilities.h"
#include "mathematics.h"

//#define DEBUG_ALL_DISTANT

using namespace std;
using namespace Utilities;
using namespace Math;

/// @note: | V |    == blockDim.x
/// @note: | D_aa | == gridDim.x
void
all_distant ( real* beam_str, real* validity_solutions, int v_id, int n_blocks, int n_threads, int n_bytes ) {
  // int warp = WHICHWARP( blockIdx.x );//blockIdx.x>>5
  //if ( domain_states[ offset + (blockIdx.x>>5) ] & ((uint) (1<<(blockIdx.x%32))) ) {
  for ( int blockIdx = 0; blockIdx < n_blocks; blockIdx++ ) {
    int check_success = 1;
    /*
    if (v_id==15)
      check_all_dist ( &beam_str[ blockIdx * n_threads * 15 ], &check_success, n_threads, v_id );
    else
     */
    check_all_dist ( &beam_str[ blockIdx * n_threads * 15 ], &check_success, n_threads );
    if ( !check_success ) {
      validity_solutions[ blockIdx ] = 0;
    }
  }
  
}//all_distant

void
check_all_dist ( real * local_point_list, int* check_success, int n_threads, int print_failed_var ) {
  
  /// N - Ca - C - O (- H)
  real my_N [3];
  real my_Ca[3];
  real my_C [3];
  real my_O [3];
  int epsilon = 30;
  int N_radii  = Utilities::get_atom_radii ( 0 );
  int Ca_radii = Utilities::get_atom_radii ( 1 );
  int C_radii  = Utilities::get_atom_radii ( 2 );
  int O_radii  = Utilities::get_atom_radii ( 3 );
  
  for (int thr = 0; thr < n_threads; thr++) {
    my_N [ 0 ] = local_point_list[ thr * 15      ];
    my_Ca[ 0 ] = local_point_list[ thr * 15 + 3  ];
    my_C [ 0 ] = local_point_list[ thr * 15 + 6  ];
    my_O [ 0 ] = local_point_list[ thr * 15 + 9  ];
    my_N [ 1 ] = local_point_list[ thr * 15 + 1  ];
    my_Ca[ 1 ] = local_point_list[ thr * 15 + 4  ];
    my_C [ 1 ] = local_point_list[ thr * 15 + 7  ];
    my_O [ 1 ] = local_point_list[ thr * 15 + 10 ];
    my_N [ 2 ] = local_point_list[ thr * 15 + 2  ];
    my_Ca[ 2 ] = local_point_list[ thr * 15 + 5  ];
    my_C [ 2 ] = local_point_list[ thr * 15 + 8  ];
    my_O [ 2 ] = local_point_list[ thr * 15 + 11 ];
    
    for ( int i = thr + 2; i < n_threads; i++ ) {
      if ( *check_success == 0 ) break;
      if ( ( (Math::eucl_dist( my_N,  &local_point_list[ i*15   ] ))*100 < (N_radii + N_radii  - epsilon) ) ||
           ( (Math::eucl_dist( my_N,  &local_point_list[ i*15+3 ] ))*100 < (N_radii + Ca_radii - epsilon) ) ||
           ( (Math::eucl_dist( my_N,  &local_point_list[ i*15+6 ] ))*100 < (N_radii + C_radii  - epsilon) ) ||
           ( (Math::eucl_dist( my_N,  &local_point_list[ i*15+9 ] ))*100 < (N_radii + O_radii  - epsilon) ) ) {
        if ( print_failed_var >= 0 ) {
          cout << "Failed " << i << " <-> " << thr << " on N\n";
        }
        /// FAILED
        *check_success = 0;
        return;
      }
      if ( ( (Math::eucl_dist( my_Ca,  &local_point_list[ i*15   ] ))*100 < (Ca_radii + N_radii  - epsilon) ) ||
          ( (Math::eucl_dist( my_Ca,  &local_point_list[ i*15+3 ] ))*100 < (Ca_radii + Ca_radii - epsilon) ) ||
          ( (Math::eucl_dist( my_Ca,  &local_point_list[ i*15+6 ] ))*100 < (Ca_radii + C_radii  - epsilon) ) ||
          ( (Math::eucl_dist( my_Ca,  &local_point_list[ i*15+9 ] ))*100 < (Ca_radii + O_radii  - epsilon) ) ) {
        if ( print_failed_var >= 0 ) {
          cout << "Failed " << i << " <-> " << thr << " on Ca\n";
        }
        /// FAILED
        *check_success = 0;
        return;
      }
      if ( ( (Math::eucl_dist( my_C,  &local_point_list[ i*15   ] ))*100 < (C_radii + N_radii  - epsilon) ) ||
          ( (Math::eucl_dist( my_C,  &local_point_list[ i*15+3 ] ))*100 < (C_radii + Ca_radii - epsilon) ) ||
          ( (Math::eucl_dist( my_C,  &local_point_list[ i*15+6 ] ))*100 < (C_radii + C_radii  - epsilon) ) ||
          ( (Math::eucl_dist( my_C,  &local_point_list[ i*15+9 ] ))*100 < (C_radii + O_radii  - epsilon) ) ) {
        if ( print_failed_var >= 0 ) {
          cout << "Failed " << i << " <-> " << thr << " on C\n";
        }
        /// FAILED
        *check_success = 0;
        return;
      }
      if ( ( (Math::eucl_dist( my_O, &local_point_list[ i*15   ] ))*100 < (O_radii + N_radii  - epsilon) ) ||
          ( (Math::eucl_dist( my_O,  &local_point_list[ i*15+3 ] ))*100 < (O_radii + Ca_radii - epsilon) ) ||
          ( (Math::eucl_dist( my_O,  &local_point_list[ i*15+6 ] ))*100 < (O_radii + C_radii  - epsilon) ) ||
          ( (Math::eucl_dist( my_O,  &local_point_list[ i*15+9 ] ))*100 < (O_radii + O_radii  - epsilon) ) ) {
        if ( print_failed_var >= 0 ) {
          cout << "Failed " << i << " <-> " << thr << " on O\n";
        }
        /// FAILED
        *check_success = 0;
        return;
      }
    }//i
  }
}//check_consistency_fast