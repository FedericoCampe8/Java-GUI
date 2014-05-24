#ifndef COCOS_MATH_
#define COCOS_MATH_

#include "typedefs.h"

namespace Math{
  
  real min (real a, real b);
  real max (real a, real b);
  
  /* Compute the distance from one point to others */
  void distance(real *X, real *distance, const int pos);
  real amin(const int N, const real *distance);
  real amax(const int N, const real *distance);
  real abs(const real x);
  
  /* Compute the exponential moving average */
  real exponential_moving_avg(real alpha,real current_val, real prev_avg);
  
  void set_identity(R_MAT &m);
  void set_identity(vec3 &v);
  void translate(point& p, const vec3& v);
  void rotate(point& p, const R_MAT& rot_m);
  void rotate_inverse(point& p, const R_MAT& rot_m);
  
  /* 3x1 Vectors routines */
  real eucl_dist (point v1, point v2);
  real eucl_dist2 (point v1, point v2);
  
  real vnorm2(real *x);
  int  vnorm(real *x);
  
  void vabs (real* v);
  void vadd(const real *x, const real *y, real *z);
  void vadd(const real *x, const real y, real *z);
  void vsub(const real *x, const real *y, real *z);
  void vsub(const real *x, const real y, real *z);
  void vsub(const real x, const real* y, real *z);
  void vdiv(real* u1, real* u2, real* v);
  void vdiv(real* u1, real u2, real* v);
  
  real vdot(real *x, real *y);
  void vcross(real *x, real *y, real *n);
  
  real truncate_number ( real n, int n_of_decimals=5 );
  
  real bond_angle ( real* a, real* b, real* c );
  real torsion_angle ( real* a, real* b, real* c, real* d );
}//Math

#endif
