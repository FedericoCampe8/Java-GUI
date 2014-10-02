#include "cuda_rmsd.h"
#include "cuda_utilities.h"
#include "cuda_math.h"

/*
 cuda_rmsd( real* current_str, real* beam_str,
 real* beam_rmsd, uint* domain_states,
 int v_id, int bb_start, int n_res ) {
 */

__global__
void
cuda_rmsd( real* beam_str, real* beam_energies,
           real* validity_solutions,
           real* known_prot, int n_res, int len_prot,
           int scope_first, int scope_second,
           bool h_set_on_known_prot ) {
  
  /// Valid structure: calculate RMSD
  if ( validity_solutions[ blockIdx.x ] > 0 ) {
    extern __shared__ real local_curr_str[];
    __shared__ real* known_str;
    known_str = &local_curr_str[ ( 4 * n_res ) * 3 ];
    
    int t = 0;
    int offset_know_prot = 15;
    if ( !h_set_on_known_prot ) offset_know_prot = 12;
    for ( int i = scope_first; i <= scope_second; i++ ) {
      for ( int j = 0; j < 4; j++ ) {
        known_str[ t*12 + 3*j + 0 ] = known_prot[ i * offset_know_prot + j*3 + 0 ];
        known_str[ t*12 + 3*j + 1 ] = known_prot[ i * offset_know_prot + j*3 + 1 ];
        known_str[ t*12 + 3*j + 2 ] = known_prot[ i * offset_know_prot + j*3 + 2 ];
        
        local_curr_str[ t*12 + 3*j + 0 ] = beam_str[ blockIdx.x * 15 * len_prot + (i*15 + j*3) + 0 ];
        local_curr_str[ t*12 + 3*j + 1 ] = beam_str[ blockIdx.x * 15 * len_prot + (i*15 + j*3) + 1 ];
        local_curr_str[ t*12 + 3*j + 2 ] = beam_str[ blockIdx.x * 15 * len_prot + (i*15 + j*3) + 2 ];
        /*
        if (blockIdx.x == 0) {
          printf ("FOLD: %f %f %f\n",
                  beam_str[ blockIdx.x * 15 * len_prot + (i*15 + j*3) + 0 ],
                  beam_str[ blockIdx.x * 15 * len_prot + (i*15 + j*3) + 1 ],
                  beam_str[ blockIdx.x * 15 * len_prot + (i*15 + j*3) + 2 ] );
        }
        */
      }//j
      t++;
    }//i
    
    beam_energies[ blockIdx.x ] = fast_rmsd ( known_str, local_curr_str, 4 * n_res );
    /*
    if (blockIdx.x == 0)
    printf("block %d: %f\n", blockIdx.x, beam_energies[ blockIdx.x ]);
     */
  }
  else {
    beam_energies[ blockIdx.x ] = MAX_ENERGY;
  }
}//cuda_rmsd

__device__
real
fast_rmsd ( real* ref_xlist, real* mov_xlist, int n_list ) {
  real R[3][3];
  real d0,d1,d2,e0,e1,f0;
  real omega;
  real mov_com[3];
  real mov_to_ref[3];
  
  /* cubic roots */
  real r1,r2,r3;
  real rlow;
  
  real v[3];
  real Eo, residual;
  
  setup_rotation( ref_xlist, mov_xlist, n_list,
                  mov_com, mov_to_ref, R, &Eo );
  
  /*
   * check if the determinant is greater than 0 to
   * see if R produces a right-handed or left-handed
   * co-ordinate system.
   */
  vcross( &R[1][0], &R[2][0], v );
  if ( vdot( &R[0][0], v ) > 0.0 )
    omega = 1.0;
  else
    omega = -1.0;
  
  /*
   * get elements we need from tran(R) x R
   *  (funky matrix naming relic of first attempt using pivots)
   *          matrix = d0 e0 f0
   *                      d1 e1
   *                         d2
   * divide matrix by d0, so that cubic root algorithm can handle it
   */
  
  d0 =  R[0][0]*R[0][0] + R[1][0]*R[1][0] + R[2][0]*R[2][0];
  
  d1 = (R[0][1]*R[0][1] + R[1][1]*R[1][1] + R[2][1]*R[2][1])/d0;
  d2 = (R[0][2]*R[0][2] + R[1][2]*R[1][2] + R[2][2]*R[2][2])/d0;
  
  e0 = (R[0][0]*R[0][1] + R[1][0]*R[1][1] + R[2][0]*R[2][1])/d0;
  e1 = (R[0][1]*R[0][2] + R[1][1]*R[1][2] + R[2][1]*R[2][2])/d0;
  
  f0 = (R[0][0]*R[0][2] + R[1][0]*R[1][2] + R[2][0]*R[2][2])/d0;
  
  /* cubic roots */
  {
    double B, C, D, q, q3, r, theta;
    /*
     * solving for eigenvalues as det(A-I*lambda) = 0
     * yeilds the values below corresponding to:
     * lambda**3 + B*lambda**2 + C*lambda + D = 0
     *   (given that d0=1.)
     */
    B = -1.0 - d1 - d2;
    C = d1 + d2 + d1*d2 - e0*e0 - f0*f0 - e1*e1;
    D = e0*e0*d2 + e1*e1 + f0*f0*d1 - d1*d2 - 2*e0*f0*e1;
    
    /* cubic root method of Viete with all safety belts off */
    q = (B*B - 3.0*C) / 9.0;
    q3 = q*q*q;
    r = (2.0*B*B*B - 9.0*B*C + 27.0*D) / 54.0;
    if ( (r/sqrt(q3)) <= -1 ) {
      theta = PI_VAL;
    }
//    else if ( (r/sqrt(q3)) >= 1 ) {
//      theta = 0.0;
//    }
    else {
      theta = acos(r/sqrt(q3));
    }
    
    r1 = r2 = r3 = -2.0 * sqrt(q);
    /*
    if (blockIdx.x == 0)
      printf("B %d r %f, q3 %f, theta %f e %f %f\n", blockIdx.x, r, q3, theta, sqrt(q3), r/sqrt(q3));
    */
    
    r1 *= cos(theta/3.0);
    r2 *= cos((theta + 2.0*PI_VAL) / 3.0);
    r3 *= cos((theta - 2.0*PI_VAL) / 3.0);
    
    r1 -= B / 3.0;
    r2 -= B / 3.0;
    r3 -= B / 3.0;
    
    if ( (r1 < 0.00) || (0.00 < r1 < 0.009) ) {
      r1 = 0.0;
    }
    if ( (r2 < 0.00) || (0.00 < r2 < 0.009) ) {
      r2 = 0.0;
    }
    if ( (r3 < 0.00) || (0.00 < r3 < 0.009) ) {
      r3 = 0.0;
    }
    /*
    if ( true ) {
      printf("B %d r1 %f, r2 %f r3 %f B %f d0 %f\n",
             blockIdx.x, r1, r2, r3, B, d0 );
    }
     */
  }
  
  /* undo the d0 norm to get eigenvalues */
  r1 = r1*d0;
  r2 = r2*d0;
  r3 = r3*d0;
  
  /* set rlow to lowest eigenval; set other two to r1,r2 */
  if ( r3<r1 && r3<r2 ) {
    rlow = r3;
  }
  else if (r2<r1 && r2<r3) {
    rlow = r2;
    r2 = r3;
  }
  else {
    rlow = r1;
    r1 = r3;
  }
  
  residual = Eo - sqrt(r1) - sqrt(r2) - omega*sqrt(fabs(rlow));
  
  /*
  if ( blockIdx.x == 0 || true ) {
  printf("B %d E0 %f, sr1 %f, sr2 %f omega %f, sqrt(rlow) %f rlow %f\n",
         blockIdx.x, Eo, r1, r2, omega, sqrt(rlow), rlow );
  }
  */
  return sqrt( (real) residual*2.0 / ((real) n_list) );
}

__device__
void
setup_rotation( real* ref_xlist,
                real* mov_xlist,
                int n_list,
                real mov_com[3],
                real mov_to_ref[3],
                real R[3][3],
                real* E0 ) {
  int i, j, n;
  real ref_com[3];
  
  /* calculate the centre of mass */
  for ( i = 0; i < 3; i++ ) {
    mov_com[i] = 0.0;
    ref_com[i] = 0.0;
  }
  
  for ( n = 0; n < n_list; n++ ) {
    for ( i = 0; i < 3; i++ ) {
      mov_com[i] += mov_xlist[ n * 3 + i ];
      ref_com[i] += ref_xlist[ n * 3 + i ];
      
//      if (blockIdx.x == 0)
//        printf("%f %d \n", mov_xlist[ n * 3 + i ], n * 3 + i);
    }
  }
  
  for (i=0; i<3; i++)
  {
    mov_com[i] /= n_list;
    ref_com[i] /= n_list;
    mov_to_ref[i] = ref_com[i] - mov_com[i];
  }
  
  /* shift mov_xlist and ref_xlist to centre of mass */
  for (n=0; n<n_list; n++)
    for (i=0; i<3; i++)
    {
      mov_xlist[n*3+i] -= mov_com[i];
      ref_xlist[n*3+i] -= ref_com[i];
    }
  
  /* initialize */
  for (i=0; i<3; i++)
    for (j=0; j<3; j++)
      R[i][j] = 0.0;
  *E0 = 0.0;
  
  for ( n = 0; n < n_list; n++ ) {
    /*
     * E0 = 1/2 * sum(over n): y(n)*y(n) + x(n)*x(n)
     */
    for ( i = 0; i < 3; i++ ) {
      *E0 +=  mov_xlist[n*3+i] * mov_xlist[n*3+i] + ref_xlist[n*3+i] * ref_xlist[n*3+i];
    }
    /*
     * correlation matrix R:
     *   R[i,j) = sum(over n): y(n,i) * x(n,j)
     *   where x(n) and y(n) are two vector sets
     */
    for (i=0; i<3; i++)
    {
      for (j=0; j<3; j++)
        R[i][j] += mov_xlist[n*3+i] * ref_xlist[n*3+j];
    }
  }
  
  *E0 *= 0.5;
}//setup_rotation
