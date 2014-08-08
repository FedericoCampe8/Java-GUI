#include "distance.h"
#include "utilities.h"
#include "mathematics.h"

//#define DEBUG_DIST

using namespace std;
using namespace Utilities;
using namespace Math;

/// @note: | V |    == blockDim.x
void
distance ( real* beam_str, real* validity_solutions, int* distance_values, int len_prot, int n_blocks, int n_threads, int n_bytes ) {
  for ( int blockIdx = 0; blockIdx < n_blocks; blockIdx++ ) {
    real scaling_factor = 0;
    for ( int threadIdx = 0 ; threadIdx < n_threads; threadIdx++ ) {
      int first_aa     = distance_values[ 3*threadIdx + 0 ];
      int second_aa    = distance_values[ 3*threadIdx + 1 ];
      int distance_val = distance_values[ 3*threadIdx + 2 ];
      real first_Ca[ 3 ];
      real second_Ca[ 3 ];
      
      
      first_Ca[ 0 ] = beam_str[ blockIdx * len_prot * 15 + first_aa * 15 + 3  ];
      first_Ca[ 1 ] = beam_str[ blockIdx * len_prot * 15 + first_aa * 15 + 4  ];
      first_Ca[ 2 ] = beam_str[ blockIdx * len_prot * 15 + first_aa * 15 + 5  ];
      
      second_Ca[ 0 ] = beam_str[ blockIdx * len_prot * 15 + second_aa * 15 + 3  ];
      second_Ca[ 1 ] = beam_str[ blockIdx * len_prot * 15 + second_aa * 15 + 4  ];
      second_Ca[ 2 ] = beam_str[ blockIdx * len_prot * 15 + second_aa * 15 + 5  ];
      
      if ( ( (int)Math::eucl_dist( first_Ca,  second_Ca ) ) * 100 > distance_val ) {
        scaling_factor += distance_val / (Math::eucl_dist( first_Ca,  second_Ca ) * 1.0);
      }
    }//threadIdx
    if ( scaling_factor > 0 )
      validity_solutions[ blockIdx ] *= scaling_factor;
  }//blockIdx
}//all_distant

