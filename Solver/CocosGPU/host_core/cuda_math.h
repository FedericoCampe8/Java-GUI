/***************************************
 *              Cuda Math              *
 ***************************************/
#ifndef COCOS_CUDA_MATH__
#define COCOS_CUDA_MATH__

#include "globals.h"

__forceinline__ __device__
void
vadd ( real* x, real* y, real* z ) {
  z[0] = x[0] + y[0];
  z[1] = x[1] + y[1];
  z[2] = x[2] + y[2];
}//vadd

__forceinline__ __device__
void
vsub ( real* x, real* y, real* z ) {
  z[0] = x[0] - y[0];
  z[1] = x[1] - y[1];
  z[2] = x[2] - y[2];
}//vsub

__forceinline__ __device__
real
vdot (real* x, real* y) {
  return ( x[0]*y[0] + x[1]*y[1] + x[2]*y[2] );
}//vdot

__forceinline__ __device__
void
vcross ( real* a, real* b, real* n ) {
  n[0] = a[1] * b[2] - a[2] * b[1];
  n[1] = a[2] * b[0] - a[0] * b[2];
  n[2] = a[0] * b[1] - a[1] * b[0];
}//vcross

__forceinline__ __device__
void
vnorm ( real* x ) {
  real scale = sqrt( x[0]*x[0] + x[1]*x[1] + x[2]*x[2] );
  if ( scale > CLOSE_TO_ZERO_VAL ){
    x[ 0 ] /= scale;
    x[ 1 ] /= scale;
    x[ 2 ] /= scale;
  }
  else
    x[ 0 ] = x[ 1 ] = x[ 2 ] = 0;
}//vnorm

__forceinline__ __device__
real
eucl_dist ( real* v1, real* v2 ) {
  real dist = sqrt ( ((v1[0] - v2[0]) * (v1[0] - v2[0])) +
                     ((v1[1] - v2[1]) * (v1[1] - v2[1])) +
                     ((v1[2] - v2[2]) * (v1[2] - v2[2])) );
  /*
  real dist = sqrt ( pow( v1[0] - v2[0], 2) +
                     pow( v1[1] - v2[1], 2) +
                     pow( v1[2] - v2[2], 2) );
  */
  if ( dist > CLOSE_TO_ZERO_VAL ) return dist;
  return 0;
}//eucl_dist

__forceinline__ __device__
real bond_angle ( real* a, real* b, real* c ) {
  real a_vec[3];
  real b_vec[3];  
  vsub( a, c, a_vec);
  vsub( b, c, b_vec);
  
  real div = vdot ( a_vec, b_vec ) / ( eucl_dist(a, c) * eucl_dist(b, c) );
  if ( div < -1 ) return 180.0;
  if ( div > 1 )  return 0;
  return acos ( div ) * 180.0 / PI_VAL;
}//bond_angle


__forceinline__ __device__
real
torsion_angle ( real* a, real* b, real* c, real* d ) {
  real ab[3];
  real cb[3];
  real bc[3];
  real dc[3];
  real abc[3];
  real bcd[3];
  vsub( a, b, ab );
  vsub( c, b, cb );
  vsub( b, c, bc );
  vsub( d, c, dc );
  
  vcross ( ab, cb, abc );
  vcross ( bc, dc, bcd );
  
  real angle = vdot( abc, bcd ) / ( sqrt( vdot( abc, abc ) ) *
                                    sqrt( vdot( bcd, bcd ) ) );
  
  if ( angle < -1 ) angle = -1;
  if ( angle > 1 )  angle = 1;
  
  if (angle == -1) angle = 180.0;
  else angle = ( acos ( angle ) ) * 180.0 / PI_VAL;

  /// --- Here it is possible to return angle without sign ---

  // Calc the sign
  real vec_prod[3];
  vcross ( abc, bcd, vec_prod );
  real val = vdot( cb, vec_prod );
  if ( fabsf(val < CLOSE_TO_ZERO_VAL) ) val = 0.0;//abs( val < CLOSE_TO_ZERO_VAL )
  if ( val < 0.0 ) angle *= -1;
  return angle;
}//torsion_angle

#endif


