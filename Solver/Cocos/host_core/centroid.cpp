#include "centroid.h"
#include "utilities.h"
#include "mathematics.h"

//#define DEBUG_CENTROID

using namespace std;
using namespace Utilities;
using namespace Math;

/// @note: | V |    == blockDim.x
/// @note: | D_aa | == gridDim.x
void
centroid ( real* beam_str, real* validity_solutions, aminoacid * aa_seq, int v_id, int n_blocks, int n_threads, int n_bytes ) {
  for ( int blockIdx = 0; blockIdx < n_blocks; blockIdx++ ) {
    if ( !validity_solutions[ blockIdx ] ) continue;
    int check_success = 1;
    check_centroid ( &beam_str[ blockIdx * n_threads * 15 ], &check_success, n_threads, aa_seq );
    if ( !check_success ) {
      validity_solutions[ blockIdx ] = 0;
    }
  }
  
}//centroid

void
check_centroid ( real * local_point_list, int* check_success, int n_threads, aminoacid * aa_seq, int print_failed_var ) {
  ///CGs
  int first_cg_radius;
  real first_atom_cg[3];
  int second_cg_radius;
  real second_atom_cg[3];
  ///Atoms
  real atm_N [3];
  real atm_Ca[3];
  real atm_C [3];
  real atm_O [3];
  ///Limits
  real limit_n, limit_ca, limit_c, limit_o, limit_cg_cg;
  ///Atom radii
  atom_radii N_radii  = Utilities::get_atom_radii ( 0 );
  atom_radii Ca_radii = Utilities::get_atom_radii ( 1 );
  atom_radii C_radii  = Utilities::get_atom_radii ( 2 );
  atom_radii O_radii  = Utilities::get_atom_radii ( 3 );
  ///Delta
  int epsilon = 30;
  int n_res = n_threads;
  for ( int thr = 1; thr < n_res-2; thr++ ) {
    Utilities::calculate_cg_atom( aa_seq [ thr ],
                                  &local_point_list [ (thr - 1) * 15 + 3 ],
                                  &local_point_list [ (thr + 0) * 15 + 3 ],
                                  &local_point_list [ (thr + 1) * 15 + 3 ],
                                  first_atom_cg, &first_cg_radius );
    /// Check consistency with all the other atoms and cgs
    for ( int thr2 = thr + 8; thr2 < n_res-1; thr2++ ) {//+1
      Utilities::calculate_cg_atom( aa_seq [ thr2 ],
                                    &local_point_list [ (thr2 - 1) * 15 + 3 ],
                                    &local_point_list [ (thr2 + 0) * 15 + 3 ],
                                    &local_point_list [ (thr2 + 1) * 15 + 3 ],
                                    second_atom_cg, &second_cg_radius );
      
      ///Atoms' positions
      for ( int i = 0; i < 3; i++ ) {
        atm_N  [ i ] = local_point_list [ thr2 * 15 + i     ];
        atm_Ca [ i ] = local_point_list [ thr2 * 15 + 3 + i ];
        atm_C  [ i ] = local_point_list [ thr2 * 15 + 6 + i  ];
        atm_O  [ i ] = local_point_list [ thr2 * 15 + 9 + i  ];
      }
      
      limit_n     = ((first_cg_radius + N_radii)  - epsilon)/2;
      limit_ca    = ((first_cg_radius + Ca_radii) - epsilon)/2;
      limit_c     = ((first_cg_radius + C_radii)  - epsilon)/2;
      limit_o     = ((first_cg_radius + O_radii)  - epsilon)/2;
      limit_cg_cg = ((first_cg_radius + second_cg_radius) - epsilon);
      
      if ( ( (Math::eucl_dist( first_atom_cg, atm_N          ))*100 < limit_n     ) ||
           ( (Math::eucl_dist( first_atom_cg, atm_Ca         ))*100 < limit_ca    ) ||
           ( (Math::eucl_dist( first_atom_cg, atm_C          ))*100 < limit_c     ) ||
           ( (Math::eucl_dist( first_atom_cg, atm_O          ))*100 < limit_o     ) ||
           ( (Math::eucl_dist( first_atom_cg, second_atom_cg ))*100 < limit_cg_cg ) ) {
        if ( print_failed_var >= 0 ) {
          cout << "Failed " << thr << " <-> " << thr2 << "\n";
        }
        /// FAILED
        *check_success = 0;
        return;
      }
    }//thr2
  }//thr
}//check_centroid