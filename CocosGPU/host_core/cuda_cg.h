/*********************************************************************
 * Authors: Federico Campeotto (campe8@nmsu.edu)                     *
 *                                                                   *
 * (C) Copyright 2012-2013                                           *
 *                                                                   *
 * This file is part of COCOS (COncurrent system with COnstraints    *
 * for protein Structure prediction).                                *
 *                                                                   *
 * COCOS is free software; you can redistribute it and/or            *
 * modify it under the terms of the GNU General Public License       *
 * as published by the Free Software Foundation;                     *
 *                                                                   *
 * COCOS is distributed WITHOUT ANY WARRANTY; without even the       *
 * implied  warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR  *
 * PURPOSE. See the GNU General Public License for more details.     *
 *                                                                   *
 * You should have received a copy of the GNU General Public License *
 * along with this program; if not, see http://www.gnu.org/licenses. *
 *                                                                   *
 *********************************************************************/
#ifndef COCOS_CUDA_CG__
#define COCOS_CUDA_CG__

#include <stdlib.h>
#include <iostream>
#include <math.h>
#include <algorithm>
#include <cassert>

#include "typedefs.h"
#include "cuda_propagation.h"

/***************************************
 *          Cuda CG Constraint         *
 ***************************************/
__global__ void cuda_cg_consistency(real * allstrs, bool * no_good_strs, aminoacid * aa_seq, fragment_type * type_seq,
                                    int bb_start, int bb_end, int n_points);

__device__ void check_consistencyCG_fast(point * local_point_list, int bb_start, int bb_end, aminoacid * aa_seq,
                                         fragment_type * type_seq,
                                         short int * accept_structure);

__device__ void calculate_centroid_atom(aminoacid a, point &ca1, point &ca2, point &ca3, real * cg, int * radius);

#endif


