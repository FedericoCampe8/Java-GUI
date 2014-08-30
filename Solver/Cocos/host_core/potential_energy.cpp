#include "potential_energy.h"
#include "utilities.h"
#include "mathematics.h"

using namespace std;
using namespace Utilities;
using namespace Math;

PotentialEnergy *
PotentialEnergy::set_parameters ( ss_type * secondary_s_info,
                                  real * h_distances, real * h_angles,
                                  real * contact_params, aminoacid * aa_seq,
                                  real * tors, real * tors_corr,
                                  real hydrogen_w, real contact_w, real correlation_w ) {
  /// Energy tables and structure parameters
  _secondary_s_info = secondary_s_info;
  _h_distances      = h_distances;
  _h_angles         = h_angles;
  _contact_params   = contact_params;
  _tors             = tors;
  _tors_corr        = tors_corr;
  _aa_seq           = aa_seq;
  /// Energy weights
  _hydrogen_weight    = hydrogen_w;
  _contact_weight     = contact_w;
  _correlation_weight = correlation_w;
  
  return this;
}//set_parameters

void
PotentialEnergy::calculate_energy ( real* setOfStructures, real* setOfEnergies,
                                    real* validStructures, int n_res,
                                    int bb_start, int bb_end,
                                    int scope_start, int scope_end,
                                    int n_bytes, int n_blocks, int n_threads ) {              
  real h_values[ n_res ];
  real c_values[ n_res ];
  real correlation_component_value;
  /// Valid structure: calculate energy
  for ( int blockIdx = 0; blockIdx < n_blocks; blockIdx++ ) {
    
    if ( validStructures[ blockIdx ] < MAX_ENERGY ) {
      memset ( h_values, 0, n_res*sizeof(real) );
      memset ( c_values, 0, n_res*sizeof(real) );
      correlation_component_value = 0;
      /// Hydrogen energy
      for ( int threadIdx = 0; threadIdx < (n_threads-32)/2; threadIdx++ ) {
        hydrogen_energy ( &setOfStructures[ blockIdx * n_res * 15 ],
                          h_values, bb_start, bb_end, n_res, threadIdx );
      }
      /// Contact energy
      for ( int threadIdx = (n_threads-32)/2; threadIdx < (n_threads-32); threadIdx++ ) {
        int scaled_threadIdx = threadIdx - ((n_threads-32)/2);
        contact_energy ( &setOfStructures[ blockIdx * n_res * 15 ],
                         c_values, bb_start, bb_end, n_res, scaled_threadIdx );
      }
      /// Correlational energy
      correlation_energy ( &setOfStructures[ blockIdx * n_res * 15 ],
                           &correlation_component_value, bb_start, bb_end, n_res );
      
      
      real hydrogen_component_value = 0;
      real contact_component_value  = 0;
      for ( int i = scope_start; i <= scope_end; i++ ) {
        hydrogen_component_value += h_values[ i ];
        contact_component_value  += c_values[ i ];
      }
      
      setOfEnergies[ blockIdx ] =
      hydrogen_component_value    * _hydrogen_weight +
      contact_component_value     * _contact_weight  +
      correlation_component_value * _correlation_weight ;
      setOfEnergies[ blockIdx ] *= validStructures[ blockIdx ];
      
    }
    else {
      setOfEnergies[ blockIdx ] = MAX_ENERGY;
    }
  }//blockIdx
}//calculate_energy

/// Energy Pontentials
void
PotentialEnergy::hydrogen_energy ( real * structure, real * h_values,
                                   int bb_start, int bb_end, int n_res, int threadIdx ) {
  real h_energy = 0;
  if ( (threadIdx >= (bb_start/5)) && (threadIdx <= ((bb_end-4)/5)) ) {
    real H_O_dist;
    int col;
    int look_up_delta, look_up_theta, look_up_psi, look_up_chi;
    ss_type my_type = _secondary_s_info[ threadIdx ];
    /// Loop on O atoms
    for ( int i = bb_start+3; i < bb_end; i += 5 ) {
      if ( ( ((i - 3) / 5) == threadIdx ) || ( ((i - 3) / 5) == (threadIdx - 1) ) ) continue;
      ///@note: we use threadIdx here -> a threadIdx identifies the AA of a H atom
      ///@note: we use i to identify the O atoms (i goes from 5 to 5 starting from 3)
      H_O_dist = Math::eucl_dist ( &structure[ threadIdx * 15 + 12 ], &structure[ i * 3 ] );
      
#ifdef H_DEBUG
      cout << "H: " <<
      structure[ threadIdx * 15 + 12 ] << " " <<
      structure[ threadIdx * 15 + 12 + 1 ] << " " <<
      structure[ threadIdx * 15 + 12 + 2 ] << endl;
      cout << "O: " << structure[ i * 3 ] << " " <<
      structure[ i * 3 + 1 ] << " " << structure[ i * 3 + 2 ] << endl;
      printf("Thread %d Atom O %d Dist H-O %f \n", threadIdx, i, H_O_dist );
      getchar();
#endif
      
      if ( (H_O_dist >= 1.75) && (H_O_dist <= 2.6) ) {//2.6 - 3.2
        /// Distance H-O
        look_up_delta = get_h_distance_bin ( H_O_dist );
        /// Bond angle O-N-H
        look_up_theta = get_h_angle_bin ( Math::bond_angle ( &structure[ i * 3 ],
                                                             &structure[ threadIdx * 15  ],
                                                             &structure[ threadIdx * 15 + 12 ] ) );
        /// Bond angle H-C-O
        look_up_psi   = get_h_angle_bin ( Math::bond_angle ( &structure[ threadIdx * 15 + 12 ],
                                                             &structure[ (i-1) * 3 ],
                                                             &structure[ i * 3 ] ) );
        /// Torsion angle Ca-C-O-H
        look_up_chi   = get_h_angle_bin ( Math::torsion_angle ( &structure[ (i-2) * 3 ],
                                                                &structure[ (i-1) * 3 ],
                                                                &structure[ i * 3     ],
                                                                &structure[ threadIdx * 15 + 12 ] ) );
#ifdef H_DEBUG
        cout << "Distance H-O: " << H_O_dist << endl;
        cout << "Bond angle O-N-H: " << Math::bond_angle ( &structure[ i * 3 ],
                                                          &structure[ threadIdx * 15  ],
                                                          &structure[ threadIdx * 15 + 12 ] )
        << endl;
        cout << "Bond angle H-C-O: " << Math::bond_angle ( &structure[ threadIdx * 15 + 12 ],
                                                          &structure[ (i-1) * 3 ],
                                                          &structure[ i * 3 ] )
        << endl;
        cout << "Torsion angle Ca-C-O-H: " << Math::torsion_angle ( &structure[ (i-2) * 3 ],
                                                                   &structure[ (i-1) * 3 ],
                                                                   &structure[ i * 3     ],
                                                                   &structure[ threadIdx * 15 + 12 ] )
        << endl;
        
        printf ( "Thread %d delta %d, theta %d, psi %d, chi %d\n",
                threadIdx, look_up_delta, look_up_theta, look_up_psi, look_up_chi );
#endif
  
        if ( my_type == helix )      col = 0;
        else if ( my_type == sheet ) col = 1;
        else col = 2;
        if ( my_type != _secondary_s_info[ (i-3)/5 ] ) col = 2;
        
        h_energy += _h_distances [ 3 * look_up_delta + col ]  +
        _h_angles [ 9 * look_up_theta + (col * 3 + 2) ]  +
        _h_angles [ 9 * look_up_psi   + (col * 3 + 1) ]  +
        _h_angles [ 9 * look_up_chi   + (col * 3 + 0) ];
      }//H_O_dist
    }
  }// threadIdx.x
  
  if ( threadIdx < n_res ) { h_values [ threadIdx ] = h_energy; }
}//hydrogen_energy

void
PotentialEnergy::contact_energy ( real * structure, real * c_values,
                                  int bb_start, int bb_end, int n_res, int threadIdx ) {
  real contact_energy = 0;
  int my_aa_idx = threadIdx;
  if ( (my_aa_idx >= (bb_start/5))  &&
       (my_aa_idx < ((bb_end-4)/5)) &&
       ( my_aa_idx > 1 )            &&
       (my_aa_idx < n_res) ) {
    /// Last AA_idx before my CG centroid = my_aa_idx - 2
    for ( int i = bb_start/5; i < my_aa_idx - 3; i++ ) {
      contact_energy_cg( structure, i, my_aa_idx - 2, &contact_energy );
    }
#ifdef CN_DEBUG
    printf( "Contact Energy Thread %d AA_%d E Val %f \n",
           threadIdx, my_aa_idx, contact_energy );
#endif
    
  }
  if ( my_aa_idx < n_res ) c_values[ my_aa_idx ] = contact_energy;
}//contact_energy

void
PotentialEnergy::contact_energy_cg  ( real * structure, int first_cg_idx, int second_cg_idx,
                                      real* c_energy ) {
  int first_cg_radius, second_cg_radius;
  real first_atom_cg[3];
  real second_atom_cg[3];
  
  Utilities::calculate_cg_atom( _aa_seq [ first_cg_idx + 1 ],
                                &structure [ (first_cg_idx * 5 + 1)*3       ],
                                &structure [ ((first_cg_idx + 1) * 5 + 1)*3 ],
                                &structure [ ((first_cg_idx + 2) * 5 + 1)*3 ],
                                first_atom_cg, &first_cg_radius );
  
  Utilities::calculate_cg_atom( _aa_seq [ second_cg_idx + 1 ],
                                &structure [ (second_cg_idx * 5 + 1)*3       ],
                                &structure [ ((second_cg_idx + 1) * 5 + 1)*3 ],
                                &structure [ ((second_cg_idx + 2) * 5 + 1)*3 ],
                                second_atom_cg, &second_cg_radius );
  
  real e = 1.0;
  real threshold = ( first_cg_radius / 100.0 ) + ( second_cg_radius / 100.0 );
  real cg_cg_distance = Math::eucl_dist( first_atom_cg, second_atom_cg );
  //if (  cg_cg_distance > threshold ) e = (threshold * threshold) / (cg_cg_distance * cg_cg_distance);
  if (  cg_cg_distance > threshold ) e = ( threshold ) / (cg_cg_distance);
  
  e *= _contact_params[ Utilities::cv_class_to_n( _aa_seq[ first_cg_idx+1 ] ) * 20 +
                        Utilities::cv_class_to_n( _aa_seq[ second_cg_idx+1 ] ) ];
  
  *c_energy += e;
}//contact_energy_cg

void
PotentialEnergy::correlation_energy ( real * structure, real * corr_val,
                                      int bb_start, int bb_end, int n_res, int v_id ) {
  *corr_val = 0;
  int prev_corr_idx = 0, AA_curr, k, l;
  real t_angle;
  
  for ( int i = bb_start; i < bb_end; i += 5 ) {
    AA_curr = i / 5;
    if ( AA_curr >= (n_res - 3) ) break;
    
    t_angle = Math::torsion_angle( &structure[ (AA_curr * 5 + 1) * 3 ],
                                   &structure[ ((AA_curr + 1) * 5 + 1) * 3 ],
                                   &structure[ ((AA_curr + 2) * 5 + 1) * 3 ],
                                   &structure[ ((AA_curr + 3) * 5 + 1) * 3 ] );
    
    if ( t_angle > 179.99 ) t_angle = 180.0;
    
    k = get_corr_aa_type ( _aa_seq[ AA_curr + 1 ] );
    l = get_corr_aa_type ( _aa_seq[ AA_curr + 2 ] );
    
    /// Look for the interval
    for ( int j = 0; j < 18; j++ ) {
      if ( t_angle < _tors[(k*(20*18*3)) + (l*(18*3)) + (j*3) + 1] ) { /// tors[k][l][j][1]
        *corr_val += _tors[(k*(20*18*3)) + (l*(18*3)) + (j*3) + 2];
        break;
      }
    }
    
    if (i == bb_start) {
      for(int j = 0; j < 18; j++)
        if( t_angle < _tors_corr[(j*(18*5)) + 1] ) { prev_corr_idx = j; break; }
    }
    else {
      for(int j = 0; j < 18; j++)
        if( t_angle < _tors_corr[(j*(18*5)) + 1] ) {
          *corr_val = *corr_val + _tors_corr[(prev_corr_idx*(18*5)) + (j*5) + 4];
          prev_corr_idx = j;
          break;
        }
    }
  }//i
}//correlation_energy


/// Auxiliary function
int
PotentialEnergy::get_h_distance_bin ( real distance ) {
  int bin = 0;
  real bin_upper_bound = 1.4;
  while ( distance >= bin_upper_bound ) {
    bin_upper_bound += 0.05;
    ++bin;
  }
  
  if ( (bin-1) > 24 ) return 24;
  return --bin;
}//get_h_distance_bin

int
PotentialEnergy::get_h_angle_bin ( real angle ) {
  int bin = 0;
  real bin_upper_bound = -180;
  while ( angle >= bin_upper_bound ) {
    bin_upper_bound += 5;
    ++bin;
  }
  
  if ( (bin-1) > 72 ) return 72;
  return --bin;
}//get_h_angle_bin

int
PotentialEnergy::get_corr_aa_type ( aminoacid a ) {
  if (a==ala) return 8;
  if (a==arg) return 17;
  if (a==asn) return 12;
  if (a==asp) return 14;
  if (a==cys) return 0;
  if (a==gln) return 13;
  if (a==glu) return 15;
  if (a==gly) return 9;
  if (a==his) return 16;
  if (a==ile) return 3;
  if (a==leu) return 4;
  if (a==lys) return 18;
  if (a==met) return 1;
  if (a==phe) return 2;
  if (a==pro) return 19;
  if (a==ser) return 11;
  if (a==thr) return 10;
  if (a==trp) return 6;
  if (a==tyr) return 7;
  if (a==val) return 5;
  return -1;
}//get_corr_aa_type







