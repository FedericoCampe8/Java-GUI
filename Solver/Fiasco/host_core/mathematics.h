#ifndef FIASCO_MATH_
#define FIASCO_MATH_

#include "typedefs.h"

typedef struct {
  R_MAT orthonormal_base;
  vec3 normal_xy;
  vec3 normal_xz;
  vec3 normal_yz;

} ReferenceSystem;


namespace Math{
  // constants
  /* real EPS = 1.0E-7F; */
  /* real MINLEN = 1.0E-7F; */
  /* real MAX_INV_LEN  = 1.0E+7; */
  const real CONST_RAD_TO_DEG = 57.2957795131;
  const real CONST_DEG_TO_RAD = 0.01745329251;
  const real CONST_PI = 3.14159265359;
  const real CONST_2PI = 6.28318530718;
  // basic
  real min (real a, real b);
  real max (real a, real b);

  // Angles and conversions
  real rad_to_deg (real angle);
  real deg_to_rad (real angle);

  void set_identity(R_MAT &m);
  void set_identity(vec3 &v);
  void translate(point& p, const vec3& v);
  void rotate(point& p, const R_MAT& rot_m);
  void rotate_inverse(point& p, const R_MAT& rot_m);

  // 3x1 Vectors routines
  void vcross(real *x, real *y, real *n);
  void vcross2(const vec3& a, const vec3 & b, vec3& n);
  real vdot(real *x, real *y);
  real vdot2(const vec3& x, const vec3& y);

  real eucl_dist (point v1, point v2);
  real eucl_dist2 (point v1, point v2);
  
  real vnorm2(real *x);
  real vnorm22(const vec3& x);
  int vnorm(real *x);
  void vabs (real* v);
  void vadd(const real *x, const real *y, real *z);
  void vadd(const real *x, const real y, real *z);
  void vsub(const real *x, const real *y, real *z);
  void vsub(const real *x, const real y, real *z);
  void vsub(const real x, const real* y, real *z);
  void vdiv(real* u1, real* u2, real* v);
  void vdiv(real* u1, real u2, real* v);
  real vdist(real *r1, real *r2);
  void middle_point (const point& a, const point& b, point& c);

  void dilatate (point& box_l, point& box_u, real d);
  void dilatate (std::pair<point, point>& box, real d);
  bool does_intersect (const std::pair<point, point>& boxA, 
		       const std::pair<point, point>& boxB);
  bool is_contained  (const std::pair<point, point>& boxA, 
		      const std::pair<point, point>& boxB);
  bool in_range (real p, real min, real max);
  bool in_range (const point p, const point min, const point max, real eps=0.0);
  bool in_range (const point p, const std::pair<point, point> bound, real eps=0.0);
  bool in_range (const point p, const point q, real eps=0.0);
  void round (point &p);
  
  // Compute the distance from one point to others
  void distance(real *X, real *distance, const int pos); 
  real amax(const int N, const real *distance); 
  real amin(const int N, const real *distance); 
  real abs(const real x);
  
  /* Compute the exponential moving average */
  real exponential_moving_avg(real alpha,real current_val, real prev_avg);

  ReferenceSystem set_reference_system (point x, point y, point z);
  
  real bond_angle ( real* a, real* b, real* c );
  real torsion_angle ( real* a, real* b, real* c, real* d );
  void calculate_h_atom (const real *c, const real *n, const real *ca, real *h);
  void calculate_cg_atom ( aminoacid a,
                           real* ca1, real* ca2, real* ca3,
                           real* cg, int* radius );
  real centroid_torsional_angle ( aminoacid a );
  real centroid_chi2 ( aminoacid a );
  real centroid_distance ( aminoacid a );
  int centroid_radius ( aminoacid a );
  
}

#endif
