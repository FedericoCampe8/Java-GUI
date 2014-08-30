#include "atm_grd.h"
#include "utilities.h"
#include "mathematics.h"
#include "atom_grid.h"

//#define DEBUG_ALL_DISTANT

using namespace std;
using namespace Utilities;
using namespace Math;

/// @note: | V |    == blockDim.x
/// @note: | D_aa | == gridDim.x
void
atom_grd ( real* beam_str, real* validity_solutions, aminoacid * aa_seq,
           int v_id, int n_blocks, int n_threads, int n_bytes ) {
  
  int reference_atom;
  real check_success, weight = 1000.0;
  real * attract_atoms = new real [ 1 + n_threads * 15 ];
  real force_of_attraction[ 3 ];
  for ( int blockIdx = 0; blockIdx < n_blocks; blockIdx++ ) {
    check_success = attract_atoms[ 0 ] = 0;
    // Find a non-clashing atom (center of force)
    attraction_atom ( &beam_str[ blockIdx * n_threads * 15 ], attract_atoms, n_threads );
    if ( attract_atoms[ 0 ] == 0 ) {
      validity_solutions[ blockIdx ] = MAX_ENERGY;
      continue;
    }
    reference_atom = rand() % ((int)attract_atoms[ 0 ]);
    force_of_attraction[ 0 ] = attract_atoms[ 3 * reference_atom + 1 + 0 ];
    force_of_attraction[ 1 ] = attract_atoms[ 3 * reference_atom + 1 + 1 ];
    force_of_attraction[ 2 ] = attract_atoms[ 3 * reference_atom + 1 + 2 ];
    // Backbone check
    check_atom_grd ( &beam_str[ blockIdx * n_threads * 15 ], force_of_attraction, &check_success, n_threads );
    // Sidechain check
    check_atom_grd_cg ( &beam_str[ blockIdx * n_threads * 15 ], force_of_attraction,  &check_success, aa_seq, n_threads );
    // Sum penalities
    validity_solutions[ blockIdx ] += (check_success / weight);
  }
  
  delete [] attract_atoms;
}//all_distant

void
attraction_atom ( real * local_point_list, real * attraction_points, int n_threads ) {
  /// N - Ca - C - O (- H)
  point my_N;
  point my_Ca;
  point my_C;
  point my_O;
  point my_H;
  
  for ( int thr = 0; thr < n_threads; thr++ ) {
    my_N [ 0 ] = local_point_list[ thr * 15 + 0  ];
    my_N [ 1 ] = local_point_list[ thr * 15 + 1  ];
    my_N [ 2 ] = local_point_list[ thr * 15 + 2  ];
    
    my_Ca[ 0 ] = local_point_list[ thr * 15 + 3  ];
    my_Ca[ 1 ] = local_point_list[ thr * 15 + 4  ];
    my_Ca[ 2 ] = local_point_list[ thr * 15 + 5  ];
    
    my_C [ 0 ] = local_point_list[ thr * 15 + 6  ];
    my_C [ 1 ] = local_point_list[ thr * 15 + 7  ];
    my_C [ 2 ] = local_point_list[ thr * 15 + 8  ];
    
    my_O [ 0 ] = local_point_list[ thr * 15 + 9  ];
    my_O [ 1 ] = local_point_list[ thr * 15 + 10 ];
    my_O [ 2 ] = local_point_list[ thr * 15 + 11 ];
    
    my_H [ 0 ] = local_point_list[ thr * 15 + 12  ];
    my_H [ 1 ] = local_point_list[ thr * 15 + 13  ];
    my_H [ 2 ] = local_point_list[ thr * 15 + 14  ];
    
    if ( !g_atom_grid->query( my_N,  N  ) ) {
      attraction_points[ ((int) attraction_points[ 0 ]) * 3 + 1 + 0 ] = my_N [ 0 ];
      attraction_points[ ((int) attraction_points[ 0 ]) * 3 + 1 + 1 ] = my_N [ 1 ];
      attraction_points[ ((int) attraction_points[ 0 ]) * 3 + 1 + 2 ] = my_N [ 2 ];
      attraction_points[ 0 ]++;
    }
    
    if ( !g_atom_grid->query( my_Ca,  CA  ) ) {
      attraction_points[ ((int) attraction_points[ 0 ]) * 3 + 1 + 0 ] = my_Ca [ 0 ];
      attraction_points[ ((int) attraction_points[ 0 ]) * 3 + 1 + 1 ] = my_Ca [ 1 ];
      attraction_points[ ((int) attraction_points[ 0 ]) * 3 + 1 + 2 ] = my_Ca [ 2 ];
      attraction_points[ 0 ]++;
    }
    
    if ( !g_atom_grid->query( my_C,  CB  ) ) {
      attraction_points[ ((int) attraction_points[ 0 ]) * 3 + 1 + 0 ] = my_C [ 0 ];
      attraction_points[ ((int) attraction_points[ 0 ]) * 3 + 1 + 1 ] = my_C [ 1 ];
      attraction_points[ ((int) attraction_points[ 0 ]) * 3 + 1 + 2 ] = my_C [ 2 ];
      attraction_points[ 0 ]++;
    }
    
    if ( !g_atom_grid->query( my_O,  O  ) ) {
      attraction_points[ ((int) attraction_points[ 0 ]) * 3 + 1 + 0 ] = my_O [ 0 ];
      attraction_points[ ((int) attraction_points[ 0 ]) * 3 + 1 + 1 ] = my_O [ 1 ];
      attraction_points[ ((int) attraction_points[ 0 ]) * 3 + 1 + 2 ] = my_O [ 2 ];
      attraction_points[ 0 ]++;
    }
    
    if ( !g_atom_grid->query( my_H,  H  ) ) {
      attraction_points[ ((int) attraction_points[ 0 ]) * 3 + 1 + 0 ] = my_H [ 0 ];
      attraction_points[ ((int) attraction_points[ 0 ]) * 3 + 1 + 1 ] = my_H [ 1 ];
      attraction_points[ ((int) attraction_points[ 0 ]) * 3 + 1 + 2 ] = my_H [ 2 ];
      attraction_points[ 0 ]++;
    }
  }//thr
}//attraction_atom


void
check_atom_grd ( real * local_point_list, real attraction[3], real* check_success, int n_threads, int print_failed_var ) {
  
  /// N - Ca - C - O (- H)
  point my_N;
  point my_Ca;
  point my_C;
  point my_O;
  point my_H;
  point reference;
  
  reference[ 0 ] = attraction[ 0 ];
  reference[ 1 ] = attraction[ 1 ];
  reference[ 2 ] = attraction[ 2 ];
  
  real avg_dist = 0;
  for ( int thr = 0; thr < n_threads; thr++ ) {
    my_N [ 0 ] = local_point_list[ thr * 15 + 0  ];
    my_N [ 1 ] = local_point_list[ thr * 15 + 1  ];
    my_N [ 2 ] = local_point_list[ thr * 15 + 2  ];
    
    my_Ca[ 0 ] = local_point_list[ thr * 15 + 3  ];
    my_Ca[ 1 ] = local_point_list[ thr * 15 + 4  ];
    my_Ca[ 2 ] = local_point_list[ thr * 15 + 5  ];
    
    my_C [ 0 ] = local_point_list[ thr * 15 + 6  ];
    my_C [ 1 ] = local_point_list[ thr * 15 + 7  ];
    my_C [ 2 ] = local_point_list[ thr * 15 + 8  ];
    
    my_O [ 0 ] = local_point_list[ thr * 15 + 9  ];
    my_O [ 1 ] = local_point_list[ thr * 15 + 10 ];
    my_O [ 2 ] = local_point_list[ thr * 15 + 11 ];
    
    my_H [ 0 ] = local_point_list[ thr * 15 + 12  ];
    my_H [ 1 ] = local_point_list[ thr * 15 + 13  ];
    my_H [ 2 ] = local_point_list[ thr * 15 + 14  ];
    /*
    avg_dist += (g_atom_grid->query( my_N,  N  ) +
                 g_atom_grid->query( my_Ca, CA ) +
                 g_atom_grid->query( my_C,  CB ) +
                 g_atom_grid->query( my_O,  O  ) +
                 g_atom_grid->query( my_H,  H  ) );
     */
    avg_dist += ( Math::eucl_dist( my_N,  reference ) +
                  Math::eucl_dist( my_Ca, reference ) +
                  Math::eucl_dist( my_C,  reference ) +
                  Math::eucl_dist( my_O,  reference ) +
                  Math::eucl_dist( my_H,  reference ) );
    avg_dist /= 5.0;
    *check_success = (*check_success) + avg_dist;
  }//thr
}//check_atom_grd


void
check_atom_grd_cg ( real * local_point_list, real attraction[3], real* check_success, aminoacid * aa_seq,
                    int n_threads , int print_failed_var ) {
  int  CG_radius;
  real my_CG[ 3 ];
  real avg_dist = 0;
  point reference;
  
  reference[ 0 ] = attraction[ 0 ];
  reference[ 1 ] = attraction[ 1 ];
  reference[ 2 ] = attraction[ 2 ];
  for ( int thr = 0; thr < n_threads - 2; thr++ ) {
    Utilities::calculate_cg_atom( aa_seq [ thr + 1 ],
                                  &local_point_list [ (thr       * 5 + 1)*3 ],
                                  &local_point_list [ ((thr + 1) * 5 + 1)*3 ],
                                  &local_point_list [ ((thr + 2) * 5 + 1)*3 ],
                                  my_CG, &CG_radius );
    //avg_dist = g_atom_grid->query( my_CG, CB, -1, CG_radius );
    avg_dist = Math::eucl_dist( my_CG,  reference );
    *check_success = (*check_success) + avg_dist;
  }//thr
}//check_atom_grd_cg


