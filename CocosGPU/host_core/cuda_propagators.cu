#include "cuda_propagators.h"
#include "cuda_utilities.h"
#include "cuda_math.h"

//#define DEBUG_ALL_DISTANT


/// @note: | V |    == blockDim.x
/// @note: | D_aa | == gridDim.x
__global__
void
cuda_all_distant ( real* beam_str, real* validity_solutions ) {
  
  extern __shared__ real local_curr_str[];
  __shared__ int check_success;
  
  copy_structure_from_to ( &beam_str[ blockIdx.x * blockDim.x * 15 ], local_curr_str );
  if ( threadIdx.x == 0 ) check_success = 1;
  
  __syncthreads();
  
  check_all_dist ( local_curr_str, &check_success );
  
  __syncthreads();
  
  if ( threadIdx.x == 0 ) {
    if ( !check_success ) {
      validity_solutions[ blockIdx.x ] = 0;
    }
  }
}//all_distant

__device__
void
check_all_dist ( real * local_point_list, int* check_success ) {
  /// N - Ca - C - O (- H)
  real my_N [ 3 ];
  real my_Ca[ 3 ];
  real my_C [ 3 ];
  real my_O [ 3 ];
  my_N [ 0 ] = local_point_list[ threadIdx.x * 15      ];
  my_Ca[ 0 ] = local_point_list[ threadIdx.x * 15 + 3  ];
  my_C [ 0 ] = local_point_list[ threadIdx.x * 15 + 6  ];
  my_O [ 0 ] = local_point_list[ threadIdx.x * 15 + 9  ];
  my_N [ 1 ] = local_point_list[ threadIdx.x * 15 + 1  ];
  my_Ca[ 1 ] = local_point_list[ threadIdx.x * 15 + 4  ];
  my_C [ 1 ] = local_point_list[ threadIdx.x * 15 + 7  ];
  my_O [ 1 ] = local_point_list[ threadIdx.x * 15 + 10 ];
  my_N [ 2 ] = local_point_list[ threadIdx.x * 15 + 2  ];
  my_Ca[ 2 ] = local_point_list[ threadIdx.x * 15 + 5  ];
  my_C [ 2 ] = local_point_list[ threadIdx.x * 15 + 8  ];
  my_O [ 2 ] = local_point_list[ threadIdx.x * 15 + 11 ];
  
  //int epsilon = 30;
  int N_radii  = get_atom_radii ( 0 );
  int Ca_radii = get_atom_radii ( 1 );
  int C_radii  = get_atom_radii ( 2 );
  int O_radii  = get_atom_radii ( 3 );
  /// Start checking from the second aa ahead, i.e., i--i+2, i--i+3, ...
  for ( int i = threadIdx.x + 2; i < blockDim.x; i++ ) {
    if ( *check_success == 0 ) break;
    if ( ( (eucl_dist( my_N,  &local_point_list[ i*15   ] ))*100 < (N_radii + N_radii  - 30) ) ||
         ( (eucl_dist( my_N,  &local_point_list[ i*15+3 ] ))*100 < (N_radii + Ca_radii - 30) ) ||
         ( (eucl_dist( my_N,  &local_point_list[ i*15+6 ] ))*100 < (N_radii + C_radii  - 30) ) ||
         ( (eucl_dist( my_N,  &local_point_list[ i*15+9 ] ))*100 < (N_radii + O_radii  - 30) ) ) {
      /// FAILED
      *check_success = 0;
      break;
    }
    if ( ( (eucl_dist( my_Ca,  &local_point_list[ i*15   ] ))*100 < (Ca_radii + N_radii  - 30) ) ||
         ( (eucl_dist( my_Ca,  &local_point_list[ i*15+3 ] ))*100 < (Ca_radii + Ca_radii - 30) ) ||
         ( (eucl_dist( my_Ca,  &local_point_list[ i*15+6 ] ))*100 < (Ca_radii + C_radii  - 30) ) ||
         ( (eucl_dist( my_Ca,  &local_point_list[ i*15+9 ] ))*100 < (Ca_radii + O_radii  - 30) ) ) {
      /// FAILED
      *check_success = 0;
      break;
    }
    if ( ( (eucl_dist( my_C,  &local_point_list[ i*15   ] ))*100 < (C_radii + N_radii  - 30) ) ||
         ( (eucl_dist( my_C,  &local_point_list[ i*15+3 ] ))*100 < (C_radii + Ca_radii - 30) ) ||
         ( (eucl_dist( my_C,  &local_point_list[ i*15+6 ] ))*100 < (C_radii + C_radii  - 30) ) ||
         ( (eucl_dist( my_C,  &local_point_list[ i*15+9 ] ))*100 < (C_radii + O_radii  - 30) ) ) {
      /// FAILED
      *check_success = 0;
      break;
    }
    if ( ( (eucl_dist( my_O,  &local_point_list[ i*15   ] ))*100 < (O_radii + N_radii  - 30) ) ||
         ( (eucl_dist( my_O,  &local_point_list[ i*15+3 ] ))*100 < (O_radii + Ca_radii - 30) ) ||
         ( (eucl_dist( my_O,  &local_point_list[ i*15+6 ] ))*100 < (O_radii + C_radii  - 30) ) ||
         ( (eucl_dist( my_O,  &local_point_list[ i*15+9 ] ))*100 < (O_radii + O_radii  - 30) ) ) {
      /// FAILED
      *check_success = 0;
      break;
    }
  }//i
}//check_all_dist


/// @note: | V |    == blockDim.x
/// @note: | D_aa | == gridDim.x
__global__
void
cuda_centroid ( real* beam_str, real* validity_solutions, aminoacid * aa_seq ) {
  if ( !validity_solutions[ blockIdx.x ] ) return;
  
  extern __shared__ real local_curr_str[];
  __shared__ int check_success;
  
  if ( threadIdx.x == 0 ) {
    for ( int threadIdx = 0; threadIdx < blockDim.x + 2; threadIdx++ ) {
      local_curr_str[ 15*threadIdx      ] = beam_str[ blockIdx.x * (blockDim.x + 2) * 15 + 15*threadIdx      ];
      local_curr_str[ 15*threadIdx + 1  ] = beam_str[ blockIdx.x * (blockDim.x + 2) * 15 + 15*threadIdx + 1  ];
      local_curr_str[ 15*threadIdx + 2  ] = beam_str[ blockIdx.x * (blockDim.x + 2) * 15 + 15*threadIdx + 2  ]; /// N
      local_curr_str[ 15*threadIdx + 3  ] = beam_str[ blockIdx.x * (blockDim.x + 2) * 15 + 15*threadIdx + 3  ];
      local_curr_str[ 15*threadIdx + 4  ] = beam_str[ blockIdx.x * (blockDim.x + 2) * 15 + 15*threadIdx + 4  ];
      local_curr_str[ 15*threadIdx + 5  ] = beam_str[ blockIdx.x * (blockDim.x + 2) * 15 + 15*threadIdx + 5  ]; /// Ca
      local_curr_str[ 15*threadIdx + 6  ] = beam_str[ blockIdx.x * (blockDim.x + 2) * 15 + 15*threadIdx + 6  ];
      local_curr_str[ 15*threadIdx + 7  ] = beam_str[ blockIdx.x * (blockDim.x + 2) * 15 + 15*threadIdx + 7  ];
      local_curr_str[ 15*threadIdx + 8  ] = beam_str[ blockIdx.x * (blockDim.x + 2) * 15 + 15*threadIdx + 8  ]; /// C
      local_curr_str[ 15*threadIdx + 9  ] = beam_str[ blockIdx.x * (blockDim.x + 2) * 15 + 15*threadIdx + 9  ];
      local_curr_str[ 15*threadIdx + 10 ] = beam_str[ blockIdx.x * (blockDim.x + 2) * 15 + 15*threadIdx + 10 ];
      local_curr_str[ 15*threadIdx + 11 ] = beam_str[ blockIdx.x * (blockDim.x + 2) * 15 + 15*threadIdx + 11 ]; /// O
      local_curr_str[ 15*threadIdx + 12 ] = beam_str[ blockIdx.x * (blockDim.x + 2) * 15 + 15*threadIdx + 12 ];
      local_curr_str[ 15*threadIdx + 13 ] = beam_str[ blockIdx.x * (blockDim.x + 2) * 15 + 15*threadIdx + 13 ];
      local_curr_str[ 15*threadIdx + 14 ] = beam_str[ blockIdx.x * (blockDim.x + 2) * 15 + 15*threadIdx + 14 ]; /// H
    }
    check_success = 1;
  }

  
//  copy_structure_from_to ( &beam_str[ blockIdx.x * blockDim.x * 15 ], local_curr_str );
//  if ( threadIdx.x == 0 ) check_success = 1;
  
  __syncthreads();
  
  check_centroid ( local_curr_str, &check_success, aa_seq );
  
  __syncthreads();
  
  if ( threadIdx.x == 0 ) {
    if ( !check_success ) {
      validity_solutions[ blockIdx.x ] = 0;
    }
  }
}//cuda_centroid

__device__
void
check_centroid ( real * local_point_list, int* check_success, aminoacid * aa_seq ) {
  ///CGs
  int first_cg_radius;
  real first_atom_cg[3];
  int second_cg_radius;
  real second_atom_cg[3];
  ///Limits
  real limit_n, limit_ca, limit_c, limit_o, limit_cg_cg;
  ///Atom radii
  atom_radii N_radii  = get_atom_radii ( 0 );
  atom_radii Ca_radii = get_atom_radii ( 1 );
  atom_radii C_radii  = get_atom_radii ( 2 );
  atom_radii O_radii  = get_atom_radii ( 3 );
  ///Delta
  //int epsilon = 30;
  int thr   = threadIdx.x + 1; 
  //int n_res = blockIdx.x + 2;
  
  calculate_cg_atom( aa_seq [ thr ],
                     &local_point_list [ (thr - 1) * 15 + 3 ],
                     &local_point_list [ (thr + 0) * 15 + 3 ],
                     &local_point_list [ (thr + 1) * 15 + 3 ],
                     first_atom_cg, &first_cg_radius );

  /// Check consistency with all the other atoms and cgs
  for ( int thr2 = thr + 8; thr2 < blockDim.x + 1; thr2++ ) {//+1
    if ( *check_success == 0 ) break;
    calculate_cg_atom( aa_seq [ thr2 ],
                       &local_point_list [ (thr2 - 1) * 15 + 3 ],
                       &local_point_list [ (thr2 + 0) * 15 + 3 ],
                       &local_point_list [ (thr2 + 1) * 15 + 3 ],
                       second_atom_cg, &second_cg_radius );
    
    limit_n     = ((first_cg_radius + N_radii)  - 30)/2;
    limit_ca    = ((first_cg_radius + Ca_radii) - 30)/2;
    limit_c     = ((first_cg_radius + C_radii)  - 30)/2;
    limit_o     = ((first_cg_radius + O_radii)  - 30)/2;
    limit_cg_cg = ((first_cg_radius + second_cg_radius) - 30);
    
    if ( ( (eucl_dist( first_atom_cg, &local_point_list [ thr2 * 15     ] ))*100 < limit_n     ) ||
         ( (eucl_dist( first_atom_cg, &local_point_list [ thr2 * 15 + 3 ] ))*100 < limit_ca    ) ||
         ( (eucl_dist( first_atom_cg, &local_point_list [ thr2 * 15 + 6 ] ))*100 < limit_c     ) ||
         ( (eucl_dist( first_atom_cg, &local_point_list [ thr2 * 15 + 9 ] ))*100 < limit_o     ) ||
         ( (eucl_dist( first_atom_cg, second_atom_cg ))*100 < limit_cg_cg ) ) {
      /// FAILED
      *check_success = 0;
      break;
    }
  }//thr2
}//check_centroid


__global__
void
cuda_check_failure ( real* state, int* events ) {
  __shared__ int failed;
  if ( threadIdx.x == 0 ) failed = 1;
  if ( state[ threadIdx.x ] > 0 ) failed = 0;
  __syncthreads();
  
  if ( threadIdx.x == 0 )
    failed ? events[ 0 ] = failed_event : events[ 0 ] = events_size;
}//check_failure


/// @note: | V |    == blockDim.x
/// @note: | D_aa | == gridDim.x
__global__
void
cuda_k_angle_shuffle ( int* vars_to_shuffle,
                      real* all_domains, int* all_domains_idx,
                      real* current_str, real* beam_str,
                      curandState *random_vals,
                      int n_vars_to_shuffle, int len_prot ) {
  /*
   * @note: here all structures are valid
   * int warp = WHICHWARP( blockIdx.x );
   * if ( !(domain_states[ MAX_DIM * v_id + warp ] & ((uint) 1<<(blockIdx.x))) ) return;
   */
  extern __shared__ real local_curr_str[];
  
  if ( threadIdx.x == 0 ) {
    for ( int threadIdx = 0; threadIdx < len_prot; threadIdx++ ) {
      local_curr_str[ 15*threadIdx      ] = current_str[ 15*threadIdx      ];
      local_curr_str[ 15*threadIdx + 1  ] = current_str[ 15*threadIdx + 1  ];
      local_curr_str[ 15*threadIdx + 2  ] = current_str[ 15*threadIdx + 2  ]; /// N
      local_curr_str[ 15*threadIdx + 3  ] = current_str[ 15*threadIdx + 3  ];
      local_curr_str[ 15*threadIdx + 4  ] = current_str[ 15*threadIdx + 4  ];
      local_curr_str[ 15*threadIdx + 5  ] = current_str[ 15*threadIdx + 5  ]; /// Ca
      local_curr_str[ 15*threadIdx + 6  ] = current_str[ 15*threadIdx + 6  ];
      local_curr_str[ 15*threadIdx + 7  ] = current_str[ 15*threadIdx + 7  ];
      local_curr_str[ 15*threadIdx + 8  ] = current_str[ 15*threadIdx + 8  ]; /// C
      local_curr_str[ 15*threadIdx + 9  ] = current_str[ 15*threadIdx + 9  ];
      local_curr_str[ 15*threadIdx + 10 ] = current_str[ 15*threadIdx + 10 ];
      local_curr_str[ 15*threadIdx + 11 ] = current_str[ 15*threadIdx + 11 ]; /// O
      local_curr_str[ 15*threadIdx + 12 ] = current_str[ 15*threadIdx + 12 ];
      local_curr_str[ 15*threadIdx + 13 ] = current_str[ 15*threadIdx + 13 ];
      local_curr_str[ 15*threadIdx + 14 ] = current_str[ 15*threadIdx + 14 ]; /// H
    }
  }
  
  /*
   if ( threadIdx.x < len_prot ) {
   copy_structure_from_to ( current_str, local_curr_str );
   }
   __syncthreads();
   */
  
  __shared__ int v_id;
  __shared__ real phi;
  __shared__ real psi;
  
  for ( int i = 0; i < n_vars_to_shuffle; i++ ) {
    if (threadIdx.x == 0) {
      v_id = vars_to_shuffle[ i ];
      int random_idx = all_domains_idx[ v_id ] +
      ( ((uint)curand( &random_vals[ (blockIdx.x + i)%gridDim.x ] )) %
       ((uint) all_domains[ all_domains_idx[ v_id ] ]) ) * 2;
      
      phi = (all_domains[ random_idx + 1 ] * PI_VAL)/180.0;
      psi = (all_domains[ random_idx + 2 ] * PI_VAL)/180.0;
    }
    __syncthreads();
    if ( threadIdx.x == 0 ) {
      for ( int j = v_id; j >= 0; j-- )
        move_phi( local_curr_str, phi, v_id, ( v_id * 5 + 1 ) * 3, 0, j );
    }
    else if ( threadIdx.x == 32 ) {
      for ( int j = v_id; j < len_prot; j++ )
        move_psi( local_curr_str, psi, v_id, ( v_id * 5 + 1 ) * 3, len_prot, j );
    }
    __syncthreads();
  }//i
  
  /// Copy back the rotated structure
  if ( threadIdx.x == 0 ) {
    for ( int threadIdx = 0; threadIdx < len_prot; threadIdx++ ) {
      beam_str[ blockIdx.x * len_prot * 15 + 15*threadIdx      ] = local_curr_str[ 15*threadIdx      ];
      beam_str[ blockIdx.x * len_prot * 15 + 15*threadIdx + 1  ] = local_curr_str[ 15*threadIdx + 1  ];
      beam_str[ blockIdx.x * len_prot * 15 + 15*threadIdx + 2  ] = local_curr_str[ 15*threadIdx + 2  ]; /// N
      beam_str[ blockIdx.x * len_prot * 15 + 15*threadIdx + 3  ] = local_curr_str[ 15*threadIdx + 3  ];
      beam_str[ blockIdx.x * len_prot * 15 + 15*threadIdx + 4  ] = local_curr_str[ 15*threadIdx + 4  ];
      beam_str[ blockIdx.x * len_prot * 15 + 15*threadIdx + 5  ] = local_curr_str[ 15*threadIdx + 5  ]; /// Ca
      beam_str[ blockIdx.x * len_prot * 15 + 15*threadIdx + 6  ] = local_curr_str[ 15*threadIdx + 6  ];
      beam_str[ blockIdx.x * len_prot * 15 + 15*threadIdx + 7  ] = local_curr_str[ 15*threadIdx + 7  ];
      beam_str[ blockIdx.x * len_prot * 15 + 15*threadIdx + 8  ] = local_curr_str[ 15*threadIdx + 8  ]; /// C
      beam_str[ blockIdx.x * len_prot * 15 + 15*threadIdx + 9  ] = local_curr_str[ 15*threadIdx + 9  ];
      beam_str[ blockIdx.x * len_prot * 15 + 15*threadIdx + 10 ] = local_curr_str[ 15*threadIdx + 10 ];
      beam_str[ blockIdx.x * len_prot * 15 + 15*threadIdx + 11 ] = local_curr_str[ 15*threadIdx + 11 ]; /// O
      beam_str[ blockIdx.x * len_prot * 15 + 15*threadIdx + 12 ] = local_curr_str[ 15*threadIdx + 12 ];
      beam_str[ blockIdx.x * len_prot * 15 + 15*threadIdx + 13 ] = local_curr_str[ 15*threadIdx + 13 ];
      beam_str[ blockIdx.x * len_prot * 15 + 15*threadIdx + 14 ] = local_curr_str[ 15*threadIdx + 14 ]; /// H
    }
  }
  
  /*
   if ( threadIdx.x < len_prot ) {
   copy_structure_from_to ( local_curr_str, &beam_str[ blockIdx.x * len_prot * 15 ] );
   }
   */
}//cuda_k_angle_shuffle

__global__
void
cuda_k_rang ( int v_id,
              real* all_domains, int* all_domains_idx,
              real* beam_str, real* beam_str_upd,
              curandState * random_vals ) {
  extern __shared__ real local_curr_str[];
  copy_structure_from_to ( &beam_str[ blockIdx.x * blockDim.x * 15 ], local_curr_str );
  /*
   if ( threadIdx.x == 0 ) {
   for ( int threadIdx = 0; threadIdx < len_prot; threadIdx++ ) {
   local_curr_str[ 15*threadIdx      ] = current_str[ 15*threadIdx      ];
   local_curr_str[ 15*threadIdx + 1  ] = current_str[ 15*threadIdx + 1  ];
   local_curr_str[ 15*threadIdx + 2  ] = current_str[ 15*threadIdx + 2  ]; /// N
   local_curr_str[ 15*threadIdx + 3  ] = current_str[ 15*threadIdx + 3  ];
   local_curr_str[ 15*threadIdx + 4  ] = current_str[ 15*threadIdx + 4  ];
   local_curr_str[ 15*threadIdx + 5  ] = current_str[ 15*threadIdx + 5  ]; /// Ca
   local_curr_str[ 15*threadIdx + 6  ] = current_str[ 15*threadIdx + 6  ];
   local_curr_str[ 15*threadIdx + 7  ] = current_str[ 15*threadIdx + 7  ];
   local_curr_str[ 15*threadIdx + 8  ] = current_str[ 15*threadIdx + 8  ]; /// C
   local_curr_str[ 15*threadIdx + 9  ] = current_str[ 15*threadIdx + 9  ];
   local_curr_str[ 15*threadIdx + 10 ] = current_str[ 15*threadIdx + 10 ];
   local_curr_str[ 15*threadIdx + 11 ] = current_str[ 15*threadIdx + 11 ]; /// O
   local_curr_str[ 15*threadIdx + 12 ] = current_str[ 15*threadIdx + 12 ];
   local_curr_str[ 15*threadIdx + 13 ] = current_str[ 15*threadIdx + 13 ];
   local_curr_str[ 15*threadIdx + 14 ] = current_str[ 15*threadIdx + 14 ]; /// H
   }
   }
   */
  
  __shared__ real phi;
  __shared__ real psi;
  __shared__ int selected_var_ca_pos;
  
  if ( threadIdx.x == 0 ) {
    int random_idx = all_domains_idx[ v_id ] +
    ( ((uint)curand( &random_vals[ blockIdx.x % gridDim.x ] )) %
     ((uint) all_domains[ all_domains_idx[ v_id ] ]) ) * 2;
    
    phi = (all_domains[ random_idx + 1 ] * PI_VAL)/180.0;
    psi = (all_domains[ random_idx + 2 ] * PI_VAL)/180.0;
    selected_var_ca_pos = ( v_id * 5 + 1 ) * 3;
  }
  
  __syncthreads();
  
  /// Move PHI angle
  if ( threadIdx.x <= v_id ) {
    move_phi( local_curr_str, phi, v_id, selected_var_ca_pos, 0 );
  }
  
  /// Move PSI angle
  if ( threadIdx.x >= v_id ) {
    move_psi( local_curr_str, psi, v_id, selected_var_ca_pos, blockDim.x );
  }
  __syncthreads();
  
  
  /// Copy back the rotated structure
  /*
   if ( threadIdx.x == 0 ) {
   for ( int threadIdx = 0; threadIdx < len_prot; threadIdx++ ) {
   beam_str[ blockIdx.x * len_prot * 15 + 15*threadIdx      ] = local_curr_str[ 15*threadIdx      ];
   beam_str[ blockIdx.x * len_prot * 15 + 15*threadIdx + 1  ] = local_curr_str[ 15*threadIdx + 1  ];
   beam_str[ blockIdx.x * len_prot * 15 + 15*threadIdx + 2  ] = local_curr_str[ 15*threadIdx + 2  ]; /// N
   beam_str[ blockIdx.x * len_prot * 15 + 15*threadIdx + 3  ] = local_curr_str[ 15*threadIdx + 3  ];
   beam_str[ blockIdx.x * len_prot * 15 + 15*threadIdx + 4  ] = local_curr_str[ 15*threadIdx + 4  ];
   beam_str[ blockIdx.x * len_prot * 15 + 15*threadIdx + 5  ] = local_curr_str[ 15*threadIdx + 5  ]; /// Ca
   beam_str[ blockIdx.x * len_prot * 15 + 15*threadIdx + 6  ] = local_curr_str[ 15*threadIdx + 6  ];
   beam_str[ blockIdx.x * len_prot * 15 + 15*threadIdx + 7  ] = local_curr_str[ 15*threadIdx + 7  ];
   beam_str[ blockIdx.x * len_prot * 15 + 15*threadIdx + 8  ] = local_curr_str[ 15*threadIdx + 8  ]; /// C
   beam_str[ blockIdx.x * len_prot * 15 + 15*threadIdx + 9  ] = local_curr_str[ 15*threadIdx + 9  ];
   beam_str[ blockIdx.x * len_prot * 15 + 15*threadIdx + 10 ] = local_curr_str[ 15*threadIdx + 10 ];
   beam_str[ blockIdx.x * len_prot * 15 + 15*threadIdx + 11 ] = local_curr_str[ 15*threadIdx + 11 ]; /// O
   beam_str[ blockIdx.x * len_prot * 15 + 15*threadIdx + 12 ] = local_curr_str[ 15*threadIdx + 12 ];
   beam_str[ blockIdx.x * len_prot * 15 + 15*threadIdx + 13 ] = local_curr_str[ 15*threadIdx + 13 ];
   beam_str[ blockIdx.x * len_prot * 15 + 15*threadIdx + 14 ] = local_curr_str[ 15*threadIdx + 14 ]; /// H
   }
   }
   */
  /// Copy back the rotated structure
  copy_structure_from_to ( local_curr_str, &beam_str_upd[ blockIdx.x * blockDim.x * 15 ] );
}//cuda_k_rang

__global__
void
cuda_c_sang ( real* current_str, real* beam_str,
              real* all_domains, int* all_domains_idx, int v_id ) {
  /*
   * @note: here all structures are valid
   * int warp = WHICHWARP( blockIdx.x );
   * if ( !(domain_states[ MAX_DIM * v_id + warp ] & ((uint) 1<<(blockIdx.x))) ) return;
   */
  extern __shared__ real local_curr_str[];
  
  __shared__ real phi;
  __shared__ real psi;
  __shared__ int selected_var_ca_pos;
  
  copy_structure_from_to ( current_str, local_curr_str );
  //memcpy(local_curr_str, current_str, blockDim.x*15*sizeof(real));
  
  if ( threadIdx.x == 0 ) {
    phi = (all_domains[ all_domains_idx[ v_id ] + blockIdx.x * 2 + 1 ] * PI_VAL)/180.0;
    psi = (all_domains[ all_domains_idx[ v_id ] + blockIdx.x * 2 + 2 ] * PI_VAL)/180.0;
    
    selected_var_ca_pos = ( v_id * 5 + 1 ) * 3;
    
#ifdef CUDA_SANG_DEBUG
    if (blockIdx.x == 0) {
      printf ( "V_%d pos %d phi %f psi %f idx %d gridDim %d \n",
              v_id, selected_var_ca_pos, phi, psi, all_domains_idx[ v_id ], gridDim.x );
    }
#endif
    
  }
  
  __syncthreads();
  
  /// Move PHI angle
  if ( threadIdx.x <= v_id ) {
    move_phi( local_curr_str, phi, v_id, selected_var_ca_pos, 0 );
  }
  
  /// Move PSI angle
  if ( threadIdx.x >= v_id ) {
    move_psi( local_curr_str, psi, v_id, selected_var_ca_pos, blockDim.x );
  }
  __syncthreads();
  
  /// Copy back the rotated structure
  copy_structure_from_to ( local_curr_str, &beam_str[ blockIdx.x * blockDim.x * 15 ] );
  //memcpy(&beam_str[ blockIdx.x * blockDim.x * 15 ], local_curr_str, blockDim.x*15*sizeof(real));
}//cuda_c_sang

__global__
void
cuda_c_mang ( real* current_str, real* beam_str,
              real* all_domains, int* all_domains_idx,
              int* vars_to_shuffle, int *random_array,
              int n_vars_to_shuffle, int v_id, int domain_size, int len_prot ) {
  extern __shared__ real local_curr_str[];
  __shared__ real phi;
  __shared__ real psi;
  __shared__ int v_id_aux;
  __shared__ int selected_var_ca_pos;
  
  if ( threadIdx.x == 0 ) {
    for ( int threadIdx = 0; threadIdx < len_prot; threadIdx++ ) {
      local_curr_str[ 15*threadIdx      ] = current_str[ 15*threadIdx      ];
      local_curr_str[ 15*threadIdx + 1  ] = current_str[ 15*threadIdx + 1  ];
      local_curr_str[ 15*threadIdx + 2  ] = current_str[ 15*threadIdx + 2  ]; /// N
      local_curr_str[ 15*threadIdx + 3  ] = current_str[ 15*threadIdx + 3  ];
      local_curr_str[ 15*threadIdx + 4  ] = current_str[ 15*threadIdx + 4  ];
      local_curr_str[ 15*threadIdx + 5  ] = current_str[ 15*threadIdx + 5  ]; /// Ca
      local_curr_str[ 15*threadIdx + 6  ] = current_str[ 15*threadIdx + 6  ];
      local_curr_str[ 15*threadIdx + 7  ] = current_str[ 15*threadIdx + 7  ];
      local_curr_str[ 15*threadIdx + 8  ] = current_str[ 15*threadIdx + 8  ]; /// C
      local_curr_str[ 15*threadIdx + 9  ] = current_str[ 15*threadIdx + 9  ];
      local_curr_str[ 15*threadIdx + 10 ] = current_str[ 15*threadIdx + 10 ];
      local_curr_str[ 15*threadIdx + 11 ] = current_str[ 15*threadIdx + 11 ]; /// O
      local_curr_str[ 15*threadIdx + 12 ] = current_str[ 15*threadIdx + 12 ];
      local_curr_str[ 15*threadIdx + 13 ] = current_str[ 15*threadIdx + 13 ];
      local_curr_str[ 15*threadIdx + 14 ] = current_str[ 15*threadIdx + 14 ]; /// H
    }
    /// Init values
    phi = (all_domains[ all_domains_idx[ v_id ] + (blockIdx.x % domain_size) * 2 + 1 ] * PI_VAL)/180.0;
    psi = (all_domains[ all_domains_idx[ v_id ] + (blockIdx.x % domain_size) * 2 + 2 ] * PI_VAL)/180.0;
    selected_var_ca_pos = ( v_id * 5 + 1 ) * 3;
  }//threadIdx
  
  __syncthreads();
  
  /// Move angles as usual
  if ( threadIdx.x == 0 ) {
    for ( int j = v_id; j >= 0; j-- )
      move_phi( local_curr_str, phi, v_id, selected_var_ca_pos, 0, j );
  }
  else if ( threadIdx.x == 32 ) {
    for ( int j = v_id; j < len_prot; j++ )
      move_psi( local_curr_str, psi, v_id, selected_var_ca_pos, len_prot, j );
  }
  
  __syncthreads();
  
  /// Move random angles
  /*
  if ( blockIdx.x >= domain_size ) {
    for ( int i = 0; i < n_vars_to_shuffle; i++ ) {
      if (threadIdx.x == 0) {
        int random_idx;
        v_id_aux = vars_to_shuffle[ i ];
        if ( v_id_aux == v_id ) continue;
        selected_var_ca_pos = ( v_id_aux * 5 + 1 ) * 3;
        /// Random value
        random_idx = all_domains_idx[ v_id_aux ] +
        (
          (random_array[ blockIdx.x - domain_size + i ]) %
          ((uint) all_domains[ all_domains_idx[ v_id_aux ] ])
         ) * 2;
        
        phi = (all_domains[ random_idx + 1 ] * PI_VAL)/180.0;
        psi = (all_domains[ random_idx + 2 ] * PI_VAL)/180.0;
      }
      __syncthreads();
      if ( threadIdx.x == 0 ) {
        for ( int j = v_id_aux; j >= 0; j-- )
          move_phi( local_curr_str, phi, v_id_aux, selected_var_ca_pos, 0, j );
      }
      else if ( threadIdx.x == 32 ) {
        for ( int j = v_id_aux; j < len_prot; j++ )
          move_psi( local_curr_str, psi, v_id_aux, selected_var_ca_pos, len_prot, j );
      }
      __syncthreads();
    }//i
  }//blockIdx
   */
  
  if ( blockIdx.x >= domain_size ) {
    for ( int i = 0; i < n_vars_to_shuffle; i++ ) {
      if (threadIdx.x == 0) {
        int random_idx;
        v_id_aux = vars_to_shuffle[ i ];
        if ( v_id_aux == v_id ) continue;
        selected_var_ca_pos = ( v_id_aux * 5 + 1 ) * 3;
        /// Random value
        random_idx = all_domains_idx[ v_id_aux ] +
        (
         (random_array[ blockIdx.x - domain_size + i ]) %
         ((uint) all_domains[ all_domains_idx[ v_id_aux ] ])
         ) * 2;
        
        phi = (all_domains[ random_idx + 1 ] * PI_VAL)/180.0;
        psi = (all_domains[ random_idx + 2 ] * PI_VAL)/180.0;
      }
      __syncthreads();
      if ( threadIdx.x == 0 ) {
        for ( int j = v_id_aux; j >= 0; j-- )
          move_phi( local_curr_str, phi, v_id_aux, selected_var_ca_pos, 0, j );
      }
      else if ( threadIdx.x == 32 ) {
        for ( int j = v_id_aux; j < len_prot; j++ )
          move_psi( local_curr_str, psi, v_id_aux, selected_var_ca_pos, len_prot, j );
      }
      __syncthreads();
    }//i
  }//blockIdx
  
  /// Copy back the rotated structure
  if ( threadIdx.x == 0 ) {
    for ( int threadIdx = 0; threadIdx < len_prot; threadIdx++ ) {
      beam_str[ blockIdx.x * len_prot * 15 + 15*threadIdx      ] = local_curr_str[ 15*threadIdx      ];
      beam_str[ blockIdx.x * len_prot * 15 + 15*threadIdx + 1  ] = local_curr_str[ 15*threadIdx + 1  ];
      beam_str[ blockIdx.x * len_prot * 15 + 15*threadIdx + 2  ] = local_curr_str[ 15*threadIdx + 2  ]; /// N
      beam_str[ blockIdx.x * len_prot * 15 + 15*threadIdx + 3  ] = local_curr_str[ 15*threadIdx + 3  ];
      beam_str[ blockIdx.x * len_prot * 15 + 15*threadIdx + 4  ] = local_curr_str[ 15*threadIdx + 4  ];
      beam_str[ blockIdx.x * len_prot * 15 + 15*threadIdx + 5  ] = local_curr_str[ 15*threadIdx + 5  ]; /// Ca
      beam_str[ blockIdx.x * len_prot * 15 + 15*threadIdx + 6  ] = local_curr_str[ 15*threadIdx + 6  ];
      beam_str[ blockIdx.x * len_prot * 15 + 15*threadIdx + 7  ] = local_curr_str[ 15*threadIdx + 7  ];
      beam_str[ blockIdx.x * len_prot * 15 + 15*threadIdx + 8  ] = local_curr_str[ 15*threadIdx + 8  ]; /// C
      beam_str[ blockIdx.x * len_prot * 15 + 15*threadIdx + 9  ] = local_curr_str[ 15*threadIdx + 9  ];
      beam_str[ blockIdx.x * len_prot * 15 + 15*threadIdx + 10 ] = local_curr_str[ 15*threadIdx + 10 ];
      beam_str[ blockIdx.x * len_prot * 15 + 15*threadIdx + 11 ] = local_curr_str[ 15*threadIdx + 11 ]; /// O
      beam_str[ blockIdx.x * len_prot * 15 + 15*threadIdx + 12 ] = local_curr_str[ 15*threadIdx + 12 ];
      beam_str[ blockIdx.x * len_prot * 15 + 15*threadIdx + 13 ] = local_curr_str[ 15*threadIdx + 13 ];
      beam_str[ blockIdx.x * len_prot * 15 + 15*threadIdx + 14 ] = local_curr_str[ 15*threadIdx + 14 ]; /// H
    }
  }//threadIdx
}//cuda_c_mang


__global__
void
cuda_distance ( real* beam_str, real* validity_solutions, int* cons_descriptions, int len_prot ) {
  real scaling_factor = 0;
  
  int first_aa     = cons_descriptions[ 3*threadIdx.x + 0 ];
  int second_aa    = cons_descriptions[ 3*threadIdx.x + 1 ];
  int distance_val = cons_descriptions[ 3*threadIdx.x + 2 ];
  real first_Ca[ 3 ];
  real second_Ca[ 3 ];
  
  
  first_Ca[ 0 ] = beam_str[ blockIdx.x * len_prot * 15 + first_aa * 15 + 3  ];
  first_Ca[ 1 ] = beam_str[ blockIdx.x * len_prot * 15 + first_aa * 15 + 4  ];
  first_Ca[ 2 ] = beam_str[ blockIdx.x * len_prot * 15 + first_aa * 15 + 5  ];
  
  second_Ca[ 0 ] = beam_str[ blockIdx.x * len_prot * 15 + second_aa * 15 + 3  ];
  second_Ca[ 1 ] = beam_str[ blockIdx.x * len_prot * 15 + second_aa * 15 + 4  ];
  second_Ca[ 2 ] = beam_str[ blockIdx.x * len_prot * 15 + second_aa * 15 + 5  ];
  
  if ( ( (int)eucl_dist( first_Ca,  second_Ca ) ) * 100 > distance_val ) {
    scaling_factor += distance_val / (eucl_dist( first_Ca,  second_Ca ) * 1.0);
  }
  if ( scaling_factor > 0 )
    validity_solutions[ blockIdx.x ] *= scaling_factor;
}//cuda_distance

/*********
 *       *
 * OTHER *
 *       *
 *********/
__global__
void
init_random( curandState *state, unsigned long seed ) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  curand_init( seed, idx, 0, &state[ idx ] );
}//init_random

__global__
void
cuda_prepare_init_set ( real* current_str,
                        real* beam_str, real* validity_solutions,
                        int best_label ) {
  if ( validity_solutions[ blockIdx.x ] == 0 ) {
    copy_structure_from_to ( &beam_str[ best_label * blockDim.x * 15 ], &beam_str[ blockIdx.x * blockDim.x * 15 ] );
  }
  
  if (blockIdx.x == 0) {
    copy_structure_from_to ( &beam_str[ best_label * blockDim.x * 15 ], current_str );
  }
}//cuda_prepare_init_set

__global__
void
cuda_update_set ( real* beam_str, real* beam_str_upd, real* validity_solutions ) {
  if ( validity_solutions[ blockIdx.x ] > 0 ) {
    copy_structure_from_to ( &beam_str[ blockIdx.x * blockDim.x * 15 ],
                             &beam_str_upd[ blockIdx.x * blockDim.x * 15 ] );
  }
}//cuda_update_set





