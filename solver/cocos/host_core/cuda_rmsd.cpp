#include "cuda_rmsd.h"
#include "cuda_utilities.h"
#include "cuda_math.h"

__global__
void
cuda_rmsd( real* current_str, real* beam_str,
           real* beam_rmsd, uint* domain_states,
           int v_id, int bb_start, int n_res ) {
  
  /// Valid structure: calculate RMSD
  if ( domain_states[ MAX_DIM * v_id + WHICHWARP( blockIdx.x ) ] & ((uint) 1<<(blockIdx.x) ) ) {
    
    extern __shared__ real local_curr_str[];
    __shared__ real* my_local_str;
    
    my_local_str = &local_curr_str[ ( 5 * n_res ) * 3 ];
    memcpy( local_curr_str, &current_str[bb_start*3], ( 5 * n_res ) * 3 * sizeof(real) );
    memcpy( my_local_str, &beam_str[ blockIdx.x * ( 5 * n_res ) * 3 + (bb_start*3)], ( 5 * n_res ) * 3 * sizeof(real) );
    
    real rmsd_val = fast_rmsd ( local_curr_str, my_local_str, 5 * n_res );
    beam_rmsd[ blockIdx.x ] = rmsd_val;
  }
  else {
    beam_rmsd[ blockIdx.x ] = MAX_RMSD;
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
  if ( vdot( &R[0][0], v ) > 0.0 ) omega = 1.0;
  else omega = -1.0;
  
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
    theta = acos(r/sqrt(q3));
    r1 = r2 = r3 = -2.0*sqrt(q);
    r1 *= cos(theta/3.0);
    r2 *= cos((theta + 2.0*PI_VAL) / 3.0);
    r3 *= cos((theta - 2.0*PI_VAL) / 3.0);
    r1 -= B / 3.0;
    r2 -= B / 3.0;
    r3 -= B / 3.0;
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
  
  residual = Eo - sqrt(r1) - sqrt(r2) - omega*sqrt(abs(rlow));
  return sqrt( (double) residual*2.0 / ((double) n_list) );
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
  double ref_com[3];
  
  /* calculate the centre of mass */
  for (i=0; i<3; i++)
  {
    mov_com[i] = 0.0;
    ref_com[i] = 0.0;
  }
  
  for (n=0; n<n_list; n++)
    for (i=0; i<3; i++)
    {
      mov_com[i] += mov_xlist[n*3+i];
      ref_com[i] += ref_xlist[n*3+i];
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
  
  for (n=0; n<n_list; n++)
  {
    /*
     * E0 = 1/2 * sum(over n): y(n)*y(n) + x(n)*x(n)
     */
    for (i=0; i<3; i++)
      *E0 +=  mov_xlist[n*3+i] * mov_xlist[n*3+i] + ref_xlist[n*3+i] * ref_xlist[n*3+i];
    
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
