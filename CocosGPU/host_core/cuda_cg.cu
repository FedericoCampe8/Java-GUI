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

#include "cuda_cg.h"
#include "cuda_math.h"
#include "cuda_utilities.h"

__global__ 
void
cuda_cg_consistency(real * allstrs, bool * no_good_strs, aminoacid * aa_seq, fragment_type * type_seq,
                    int bb_start, int bb_end, int n_points) {
  
  extern __shared__ point local_point_list[];
  int frg_idx = blockIdx.x;
  
  if (no_good_strs[frg_idx]) {
    __shared__ short int accept_structure[MAX_TARGET_SIZE];
  
    int n_pt = n_points*3, z = 0;   //g_var_point_list.size() * 3
    int start_my_points = frg_idx * n_pt;
    for (int i = 0; i < n_points; i++) {
      for (int j = 0; j < 3; j++) {
        local_point_list[i][j] = allstrs[start_my_points + z];
        z++;
      }
    }
  
    for (int i = 0; i < blockDim.x; i++) accept_structure[i] = -1;

    check_consistencyCG_fast(local_point_list, bb_start, bb_end, aa_seq, type_seq, accept_structure);
  
    __syncthreads();
  
    bool good_structure = true;
    if (threadIdx.x==0) {
      for (int i = 0; i < blockDim.x; i++)
        if (accept_structure[i] >= 0) {
          good_structure = false;
          break;
        }
      no_good_strs[frg_idx] = good_structure;
    }
  }//no_good_strs[frg_idx]
}//cuda_all_dist_consistency

__device__
void
check_consistencyCG_fast(point * local_point_list, int bb_start, int bb_end, aminoacid * aa_seq,
                         fragment_type * type_seq, short int * accept_structure) {
  int epsilon = 30;
  int n_res = bb_end/4;
  int ca_1, ca_2, ca_3;
  int ca_s1, ca_s2, ca_s3;
  int radius1, radius2;
  point cg1, cg2;
  point pt_j_1, pt_j_2, pt_j_3, pt_j_4;
  
  atom_radii rj_1, rj_2, rj_3, rj_4;
  real dist_n, dist_ca, dist_c, dist_o;
  int limit_n, limit_ca, limit_c, limit_o, limit_cg_cg, dist_cg_cg;

  ca_1 = 4*threadIdx.x + 1;
  ca_2 = 4*threadIdx.x + 5;
  ca_3 = 4*threadIdx.x + 9;
  
  calculate_centroid_atom(aa_seq[(threadIdx.x)+1],
                          local_point_list[ca_1],
                          local_point_list[ca_2],
                          local_point_list[ca_3],
                          cg1, &radius1);
  
  for (int i = 0; i < n_res-2; i++) {
    ca_s1 = 4*i + 1;
    ca_s2 = 4*i + 5;
    ca_s3 = 4*i + 9;
    
    // BB atoms
    for (int ii = 0; ii < 3; ii++) {
      pt_j_1[ii] = local_point_list[4*i+0][ii];
      pt_j_2[ii] = local_point_list[4*i+1][ii];
      pt_j_3[ii] = local_point_list[4*i+2][ii];
      pt_j_4[ii] = local_point_list[4*i+3][ii];
    }
    
    // Centroid
    calculate_centroid_atom(aa_seq[i+1],
                            local_point_list[ca_s1],
                            local_point_list[ca_s2],
                            local_point_list[ca_s3],
                            cg2, &radius2);
    
    rj_1 = get_atom_radii(4*i+0);
    rj_2 = get_atom_radii(4*i+1);
    rj_3 = get_atom_radii(4*i+2);
    rj_4 = get_atom_radii(4*i+3);
    
    dist_n  = eucl_dist(cg1, pt_j_1);
    dist_ca = eucl_dist(cg1, pt_j_2);
    dist_c  = eucl_dist(cg1, pt_j_3);
    dist_o  = eucl_dist(cg1, pt_j_4);
    
    limit_n  = ((radius1 + rj_1) - epsilon)/2;
    limit_ca = ((radius1 + rj_2) - epsilon)/2;
    limit_c  = ((radius1 + rj_3) - epsilon)/2;
    limit_o  = ((radius1 + rj_4) - epsilon)/2;
    
    dist_cg_cg  = eucl_dist(cg1, cg2);
    limit_cg_cg = ((radius1 + radius2) - epsilon)/2;
    
    if ((abs((int)(threadIdx.x+1) - i) < 2) || (type_seq[i] == sheet)) { //(type_seq[threadIdx.x+1] == sheet)
      dist_n = 10000;
      dist_ca = 10000;
      dist_c = 10000;
      dist_o = 10000;
      dist_cg_cg = 10000;
    }
    
    if ((((int)(dist_n*100))  < limit_n)  ||
        (((int)(dist_ca*100)) < limit_ca) ||
        (((int)(dist_c*100))  < limit_c)  ||
        (((int)(dist_o*100))  < limit_o)  ||
        (((int)(dist_cg_cg*100)) < limit_cg_cg)) {
      if(accept_structure[threadIdx.x] < 0 ) {
        accept_structure[threadIdx.x] = threadIdx.x;
        
      /*
        printf("b %d, fail thr %d, i %d, n %d, ca %d, c %d, o %d, cg %d, ln %d, lca %d, lc %d, lo %d, lcg %d\n",
               blockIdx.x, threadIdx.x+1, i, (int)(dist_n*100), (int)(dist_ca*100), (int)(dist_c*100), (int)(dist_o*100),
               (int)(dist_cg_cg*100), limit_n, limit_ca, limit_c, limit_o, limit_cg_cg);
       */
        
        
      }
    }
    
  }//i
}//check_consistency_fast

__device__
void
calculate_centroid_atom(aminoacid a, point &ca1, point &ca2, point &ca3, real * cg, int * radius) {
  vec3 v1,v2,v3,v,b;
  R_MAT R;
  real chi2, tors, dist, x;
  int i;
  real D, Dx, Dy, Dz;
  
  // Placement of the centroid using dist, chi2, e tors
  chi2 = centroid_chi2 (a);
  tors = centroid_torsional_angle (a);
  dist = centroid_distance(a);
  
  // v1 is the normalized vector w.r.t. ca1, ca2
  vsub (ca2, ca1, v1);
  vnorm (v1);
  
  // v2 is the normalized vector w.r.t. ca2, ca3
  vsub (ca3, ca2, v2);
  vnorm (v2);
  
  // Compute v1 (subtracting the component along v2) so that to
  // obtain v1 and v2 orthogonal each other
  x = vdot (v1, v2);
  for (i = 0; i < 3; i++)
    v1[i] = v1[i] - x*v2[i];
  vnorm (v1);
  
  // compute v3 orthogonal to v1 and v2
  vcross (v1, v2, v3);
  
  // Using Cramer method
  b[0] = cos(chi2 * M_PI/180);
  b[1] = sin(tors * M_PI/180) * sqrt(1 - b[0] * b[0]) ;
  b[2] = cos(tors * M_PI/180) * sqrt(1 - b[0] * b[0]) ;
  
  R[0][0] = v2[0];
  R[0][1] = v2[1];
  R[0][2] = v2[2];
  
  R[1][0] = v3[0];
  R[1][1] = v3[1];
  R[1][2] = v3[2];
  
  R[2][0] = -v1[0];
  R[2][1] = -v1[1];
  R[2][2] = -v1[2];
  D =
  R[0][0] * R[1][1] * R[2][2] +
  R[0][1] * R[1][2] * R[2][0] +
  R[0][2] * R[1][0] * R[2][1] -
  R[0][2] * R[1][1] * R[2][0] -
  R[0][1] * R[1][0] * R[2][2] -
  R[0][0] * R[1][2] * R[2][1];
  Dx =
  b[0] * (R[1][1] * R[2][2] - R[1][2] * R[2][1]) +
  b[1] * (R[2][1] * R[0][2] - R[2][2] * R[0][1]) +
  b[2] * (R[0][1] * R[1][2] - R[0][2] * R[1][1]) ;
  Dy =
  b[0] * (R[1][2] * R[2][0] - R[1][0] * R[2][2]) +
  b[1] * (R[2][2] * R[0][0] - R[2][0] * R[0][2]) +
  b[2] * (R[0][2] * R[1][0] - R[0][0] * R[1][2]) ;
  Dz =
  b[0] * (R[1][0] * R[2][1] - R[1][1] * R[2][0]) +
  b[1] * (R[2][0] * R[0][1] - R[2][1] * R[0][0]) +
  b[2] * (R[0][0] * R[1][1] - R[0][1] * R[1][0]) ;
  v[0] = Dx/D;
  v[1] = Dy/D;
  v[2] = Dz/D;
  
  // Now compute the centroids coordinates
  for(i = 0; i < 3; i++) v[i] = dist * v[i];
  
  // Update the output
  vadd (v, ca2, cg);
  *radius = centroid_radius(a);
}//calculate_centroid_atom
