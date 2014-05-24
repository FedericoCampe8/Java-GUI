/***************************************
 *              Cuda RMSD              *
 ***************************************/
#ifndef COCOS_CUDA_RMSD__
#define COCOS_CUDA_RMSD__

#include "globals.h"

__global__ void cuda_rmsd( real* current_str, real* beam_str,
                           real* beam_rmsd, uint* domain_states,
                           int v_id, int bb_start, int n_res );

__device__ real fast_rmsd( real* ref_xlist, real* mov_xlist, int n_list );

__device__ void setup_rotation( real* ref_xlist, real* mov_xlist, int n_list,
                                real mov_com[3], real mov_to_ref[3],
                                real R[3][3], real* E0 );

#endif


