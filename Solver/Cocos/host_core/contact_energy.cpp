#include "contact_energy.h"
#include "utilities.h"
#include "mathematics.h"
#include "atom_grid.h"

using namespace std;
using namespace Utilities;
using namespace Math;

ContactEnergy *
ContactEnergy::set_parameters ( aminoacid * aa_seq ) {
  _aa_seq = aa_seq;
  return this;
}//set_parameters

void
ContactEnergy::calculate_energy ( real* setOfStructures, real* setOfEnergies,
                                  real* validStructures, int n_res,
                                  int bb_start, int bb_end,
                                  int scope_start, int scope_end,
                                  int n_bytes, int n_blocks, int n_threads ) {
                                    
  int  CG_radius;
  real my_CG[ 3 ];
  real c_values[ n_res ];
  real * current_structure;
  /// Valid structure: calculate contacts
  for ( int blockIdx = 0; blockIdx < n_blocks; blockIdx++ ) {
    if ( validStructures[ blockIdx ] < MAX_ENERGY ) {
      memset ( c_values, 0, n_res*sizeof(real) );
      current_structure = &setOfStructures[ blockIdx * n_res * 15 ];
      /// Check contacts for each atom of the backbone --- Backbone
      for ( int threadIdx = 0; threadIdx < n_res * 5; threadIdx++ ) {
        if ( g_docking->query ( current_structure[ 3*threadIdx + 0 ],
                                current_structure[ 3*threadIdx + 1 ],
                                current_structure[ 3*threadIdx + 2 ],
                                get_atom_type ( threadIdx ) ) > 0 )  {
          /// Contact
          c_values[ threadIdx / 5 ] -= 1;
        }
      }//threadIdx
      /// Check contacts for each atom of the Sidechain --- Sidechain
      for ( int threadIdx = 0; threadIdx < n_res - 2; threadIdx++ ) {
        Utilities::calculate_cg_atom( _aa_seq [ threadIdx + 1 ],
                                      &current_structure [ (threadIdx       * 5 + 1)*3 ],
                                      &current_structure [ ((threadIdx + 1) * 5 + 1)*3 ],
                                      &current_structure [ ((threadIdx + 2) * 5 + 1)*3 ],
                                      my_CG, &CG_radius );

        if ( g_docking->query( my_CG, CB, -1, CG_radius ) > 0 ) {
          /// Contact
          c_values[ threadIdx ] -= 1;
        }
      }//thr
      
      
      real contact_component_value = 0;
      for ( int i = scope_start; i <= scope_end; i++ ) {
        contact_component_value  += c_values[ i ];
      }
      
      setOfEnergies[ blockIdx ] = contact_component_value;
      setOfEnergies[ blockIdx ] *= validStructures[ blockIdx ];
    }
    else {
      setOfEnergies[ blockIdx ] = MAX_ENERGY;
    }
  }//blockIdx
}//calculate_energy


