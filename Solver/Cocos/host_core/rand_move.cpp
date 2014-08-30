#include "rand_move.h"
#include "utilities.h"

//#define RAND_MOVE_DEBUG

/// @note: | V |    == blockDim.x
/// @note: | D_aa | == gridDim.x

void
translate_rotate_rand ( real* beam_str, int len_prot, int n_blocks, int n_threads, int smbytes ) {
  // @note: len_prot = n_res
  R_MAT rot_m;
  int rand_num, pos_neg;
  real rand_angle, px, py, pz;
  real rand_step_x, rand_step_y, rand_step_z;
  double cos_val, sin_val;
  for ( int blockIdx = 0; blockIdx < n_blocks; blockIdx++ ) {
    // Random coordinates
    pos_neg = ((rand() % 100 + 1) > 50) ? 1 : -1;
    rand_step_x = ((rand() % 100 + 1) / 1000.0) * pos_neg;
    
    pos_neg = ((rand() % 100 + 1) > 50) ? 1 : -1;
    rand_step_y = ((rand() % 100 + 1) / 1000.0) * pos_neg;
    
    pos_neg = ((rand() % 100 + 1) > 50) ? 1 : -1;
    rand_step_z = ((rand() % 100 + 1) / 1000.0) * pos_neg;
    // Perform translation
    Utilities::translate_structure ( &beam_str[ blockIdx * n_threads * 15 ],
                                     0,
                                     beam_str[ blockIdx * n_threads * 15 + 0 ] + rand_step_x,
                                     beam_str[ blockIdx * n_threads * 15 + 1 ] + rand_step_y,
                                     beam_str[ blockIdx * n_threads * 15 + 2 ] + rand_step_z,
                                     len_prot * 5
                                    );
    // Random rotations: decide axis
    rand_num   = rand() % 100 + 1;
    rand_angle = rand() % 3   + 1;
    cos_val    = cos ( rand_angle * PI_VAL / 180.0 );
    sin_val    = sin ( rand_angle * PI_VAL / 180.0 );
    if ( rand_num <= 33 ) {
      // Rotation around z-axis
      rot_m[ 0 ][ 0 ] = cos_val;
      rot_m[ 0 ][ 1 ] = sin_val;
      rot_m[ 0 ][ 2 ] = 0;
      
      rot_m[ 1 ][ 0 ] = sin_val;
      rot_m[ 1 ][ 1 ] = cos_val;
      rot_m[ 1 ][ 2 ] = 0;
      
      rot_m[ 2 ][ 0 ] = 0;
      rot_m[ 2 ][ 1 ] = 0;
      rot_m[ 2 ][ 2 ] = 1;
    }
    else if ( (rand_num > 33)  && (rand_num <= 66) ) {
      // Rotation around x-axis
      rot_m[ 0 ][ 0 ] = 1;
      rot_m[ 0 ][ 1 ] = 0;
      rot_m[ 0 ][ 2 ] = 0;
      
      rot_m[ 1 ][ 0 ] = 0;
      rot_m[ 1 ][ 1 ] = cos_val;
      rot_m[ 1 ][ 2 ] = -sin_val;
      
      rot_m[ 2 ][ 0 ] = 0;
      rot_m[ 2 ][ 1 ] = sin_val;
      rot_m[ 2 ][ 2 ] = -cos_val;
    }
    else {
      // Rotation around y-axis
      rot_m[ 0 ][ 0 ] = cos_val;
      rot_m[ 0 ][ 1 ] = 0;
      rot_m[ 0 ][ 2 ] = sin_val;
      
      rot_m[ 1 ][ 0 ] = 0;
      rot_m[ 1 ][ 1 ] = 1;
      rot_m[ 1 ][ 2 ] = 0;
      
      rot_m[ 2 ][ 0 ] = -sin_val;
      rot_m[ 2 ][ 1 ] = 0;
      rot_m[ 2 ][ 2 ] = cos_val;
    }
    
    for ( int i = 0; i < len_prot * 5; i++ ) {
      px =
      rot_m[ 0 ][ 0 ] * beam_str[ blockIdx * n_threads * 15 + i*3 + 0 ] +
      rot_m[ 0 ][ 1 ] * beam_str[ blockIdx * n_threads * 15 + i*3 + 1 ] +
      rot_m[ 0 ][ 2 ] * beam_str[ blockIdx * n_threads * 15 + i*3 + 2 ];
      
      py =
      rot_m[ 1 ][ 0 ] * beam_str[ blockIdx * n_threads * 15 + i*3 + 0 ] +
      rot_m[ 1 ][ 1 ] * beam_str[ blockIdx * n_threads * 15 + i*3 + 1 ] +
      rot_m[ 1 ][ 2 ] * beam_str[ blockIdx * n_threads * 15 + i*3 + 2 ];
      
      pz =
      rot_m[ 2 ][ 0 ] * beam_str[ blockIdx * n_threads * 15 + i*3 + 0 ] +
      rot_m[ 2 ][ 1 ] * beam_str[ blockIdx * n_threads * 15 + i*3 + 1 ] +
      rot_m[ 2 ][ 2 ] * beam_str[ blockIdx * n_threads * 15 + i*3 + 2 ];
      
      if (std::abs((double)px) < CLOSE_TO_ZERO_VAL) px = 0;
      if (std::abs((double)py) < CLOSE_TO_ZERO_VAL) py = 0;
      if (std::abs((double)pz) < CLOSE_TO_ZERO_VAL) pz = 0;
      
      beam_str[ blockIdx * n_threads * 15 + i*3 + 0 ] = px;
      beam_str[ blockIdx * n_threads * 15 + i*3 + 1 ] = py;
      beam_str[ blockIdx * n_threads * 15 + i*3 + 2 ] = pz;
    }//i
  }//blockIdx
}//translate_rotate_rand

