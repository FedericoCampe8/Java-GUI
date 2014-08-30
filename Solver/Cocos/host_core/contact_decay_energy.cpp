#include "contact_decay_energy.h"
#include "utilities.h"
#include "mathematics.h"
#include "atom_grid.h"

using namespace std;
using namespace Utilities;
using namespace Math;

ContactDecayEnergy *
ContactDecayEnergy::set_parameters ( aminoacid * aa_seq, point atom, int atom_idx ) {
  _aa_seq   = aa_seq;
  _atom_idx = atom_idx;
  for ( int i = 0; i < 3; i++ ) {
    _atom_coordinates [ i ] = atom [ i ];
  }
  return this;
}//set_parameters

void
ContactDecayEnergy::calculate_energy ( real* setOfStructures, real* setOfEnergies,
                                       real* validStructures, int n_res,
                                       int bb_start, int bb_end,
                                       int scope_start, int scope_end,
                                       int n_bytes, int n_blocks, int n_threads ) {
  int  CG_radius;
  real my_CG[ 3 ];
  real c_values[ n_res ];
  real * current_structure;
  real contact_component_value;
  
  // Points
  point pep_point;
  point dock_point;
  // Distances
  real min_bound, max_bound;
  real distance, avg_distance;
  // Dock atom
  int atom_idx;
  real weight = 10.0;//1000.0
  /// Valid structure: calculate contacts
  for ( int blockIdx = 0; blockIdx < n_blocks; blockIdx++ ) {
    if ( validStructures[ blockIdx ] < MAX_ENERGY ) {
      memset ( c_values, 0, n_res*sizeof(real) );
      current_structure = &setOfStructures[ blockIdx * n_res * 15 ];

      // Check contacts between a given pair of atoms
      contact_component_value = 0;
      // Check hard constraints
      distance     = 0;
      avg_distance = 0;
      for ( int idx = 0; idx < gh_params.force_contact.size(); idx++ ) {
        min_bound = gh_params.force_contact[ idx ][ 0 ];
        max_bound = gh_params.force_contact[ idx ][ 1 ];
        atom_idx  = (int) gh_params.force_contact[ idx ][ 2 ];
          
        pep_point[ 0 ]  =  current_structure[ (( atom_idx - 1) * 3) + 0 ];
        pep_point[ 1 ]  =  current_structure[ (( atom_idx - 1) * 3) + 1 ];
        pep_point[ 2 ]  =  current_structure[ (( atom_idx - 1) * 3) + 2 ];
        
        dock_point[ 0 ] = gh_params.force_contact[ idx ][ 3 ];
        dock_point[ 1 ] = gh_params.force_contact[ idx ][ 4 ];
        dock_point[ 2 ] = gh_params.force_contact[ idx ][ 5 ];
        
        distance = Math::eucl_dist( pep_point, dock_point );
        avg_distance += distance;
        //cout << "Distance " << distance << endl;
        if ( distance > max_bound ) {
          contact_component_value += ( (weight * max_bound) / (distance) ); // distance*distance
        }
        else  if ( distance >= min_bound ) {
          contact_component_value += weight * max_bound;
        }
        else {
          contact_component_value = gh_params.force_contact.size() * MAX_ENERGY;
        }
      }//idx
      
      avg_distance /= (gh_params.force_contact.size() * 1.0);

      //cout << "contact_component_value " << contact_component_value << endl; getchar();
      if ( contact_component_value < (gh_params.force_contact.size() * MAX_ENERGY) ) {
        if ( (validStructures[ blockIdx ] == 0) && (avg_distance < 5) ) {
          validStructures[ blockIdx ] = CLOSE_TO_ZERO_VAL;
        }
        real constraint_weight = 1.0/validStructures[ blockIdx ];
        if ( constraint_weight > 3 * contact_component_value ) {
          constraint_weight = 3 * contact_component_value;
        }
        
        real dynamic_weight = 0.4;//0.4
        if ( contact_component_value >= (weight * max_bound * gh_params.force_contact.size()) ) {
          dynamic_weight = 50;
          /*
          cout << "Energy val " << contact_component_value << " validity " <<
          validStructures[ blockIdx ] << " -> " << dynamic_weight * constraint_weight << endl;
          cout << "Total " << setOfEnergies[ blockIdx ] << endl; getchar();
           */
        }
        
        //cout << "Energy val " << contact_component_value << " validity " <<
        //validStructures[ blockIdx ] << " -> " << dynamic_weight * constraint_weight << endl;
        setOfEnergies[ blockIdx ] = -1.0 * ( contact_component_value + dynamic_weight * constraint_weight );
        //cout << "Total " << setOfEnergies[ blockIdx ] << endl; getchar();
      }
      else {
        setOfEnergies[ blockIdx ] = MAX_ENERGY;
      }
    }
    else {
      setOfEnergies[ blockIdx ] = MAX_ENERGY;
    }
  }//blockIdx
}//calculate_energy


