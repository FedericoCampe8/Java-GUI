#include "typedefs.h"
#include "mathematics.h"

#include <cmath>
#include <string.h>
#include <utility>

real 
Math::min (real a, real b) {
  return (a<b)? a : b;
}//-

 real 
Math::max (real a, real b) {
  return (a>b)? a : b;
 }//-

real 
Math::rad_to_deg (real angle) {
  return angle * CONST_RAD_TO_DEG; 
}//-

real 
Math::deg_to_rad (real angle) {
  return angle * CONST_DEG_TO_RAD; 
}//- 


// Compute the distance from the current point to previous point (left hand size points)
 void 
Math::distance(real *X, real *distance, const int pos){
  register int CurrentCol;
  register int CurrentPos;
  register real dx,dy,dz;
  register int N;
  N = 3 * pos;
  for (CurrentCol = 0 ; CurrentCol < pos ; CurrentCol ++){
    CurrentPos = CurrentCol * 3;
    dx = X[CurrentPos] - X[N];
    dy = X[CurrentPos + 1] - X[N + 1];
    dz = X[CurrentPos + 2] - X[N + 2];
    distance[CurrentCol] = dx * dx + dy * dy + dz * dz;
  }
}//-

// Find the minimum distance from a point to other points
real 
Math::amin(const int N, const real *distance){
  register real mindis = distance[0];
  register int ii;
  for (ii = 1 ; ii < N ;){
    (mindis > distance[ii]) ? (mindis = distance[ii] , ii++) : (ii++);
  }
  return mindis;
}//-

// the maximum value of distance
real 
Math::amax(const int N, const real *distance) {
  register real maxdis = distance[0];
  register int ii;
  for (ii = 1 ; ii < N ;){
    (maxdis < distance[ii]) ? ( maxdis = distance[ii] , ii++) : (ii++);
  }
  return maxdis;
}//-

//Abs for realS
real 
Math::abs (const real x) {
  return (x >= 0)? x : -x;
}//-

// Compute the exponential moving average
real 
Math::exponential_moving_avg (real alpha,real current_val, real prev_avg){
  return (alpha*current_val) + ((1 - alpha)*prev_avg);
}//-

// // compute the centroid of a box given an upper and lower bound
// // representation -- i.e. the middle point of the diagonal
// real
// Math::box_centroid (point l, point u) {
//   register real a = u[0]-l[0];
//   register real b = u[1]-l[1];
//   register real c = u[2]-l[2];
//   register real d2 = (a*a + b*b);
//   return sqrt(c*c + d)/2;
// }

/***************************************************************************/
/* Matrix/Vectors Operation */
/***************************************************************************/
void 
Math::set_identity(R_MAT &m){
  m[0][0] = m[1][1] = m[2][2] = 1;
  m[0][1] = m[0][2] = m[1][0] = m[1][2] = m[2][0] = m[2][1] = 0;
}

void 
Math::set_identity(vec3 &v){
  v[0] = v[1] = v[2] = 0;
}

void
Math::translate(point& p, const vec3& v) {
  p[0]+=v[0]; p[1]+=v[1]; p[2]+=v[2];
}//-

void
Math::rotate(point& p, const R_MAT& rot_m) {
  point _p;
  memcpy(&_p, &p, sizeof(point));
  p[0] = rot_m[0][0] * _p[0] + rot_m[0][1] * _p[1] + rot_m[0][2] * _p[2];
  p[1] = rot_m[1][0] * _p[0] + rot_m[1][1] * _p[1] + rot_m[1][2] * _p[2];   
  p[2] = rot_m[2][0] * _p[0] + rot_m[2][1] * _p[1] + rot_m[2][2] * _p[2]; 
}//-

void
Math::rotate_inverse(point& p, const R_MAT& rot_m) {
  point _p;
  memcpy(&_p, &p, sizeof(point));
  p[0] = rot_m[0][0] * _p[0] + rot_m[1][0] * _p[1] + rot_m[2][0] * _p[2];
  p[1] = rot_m[0][1] * _p[0] + rot_m[1][1] * _p[1] + rot_m[2][1] * _p[2];   
  p[2] = rot_m[0][2] * _p[0] + rot_m[1][2] * _p[1] + rot_m[2][2] * _p[2]; 
}//-

//Calculate the eucleudian distance between two vectors
real 
Math::eucl_dist (point v1, point v2) {
  register real x = v1[0] - v2[0];
  register real y = v1[1] - v2[1];
  register real z = v1[2] - v2[2];
  return sqrt (x*x + y*y + z*z);
}//-

// Squared eucleudian distance 
real 
Math::eucl_dist2 (point v1, point v2) {  
  register real x = v1[0] - v2[0];
  register real y = v1[1] - v2[1];
  register real z = v1[2] - v2[2];
  return (x*x + y*y + z*z);
}//-

// Calculate the nrm2 of a given vector -- magnitude of a vector
real 
Math::vnorm2(real *x) {
  return sqrt(x[0]*x[0] + x[1]*x[1] + x[2]*x[2]);
}//-

real 
Math::vnorm22(const vec3& x) {
  return sqrt(x[0]*x[0] + x[1]*x[1] + x[2]*x[2]);
}//-

// Normalize a given vector by its Frobenius norm
int 
Math::vnorm(real *x) {
  register real scale = sqrt(x[0]*x[0] + x[1]*x[1] + x[2]*x[2]);
  if (scale > 0){
    x[0] /= scale; // x[0] = (x[0] < 0.00001 && x[0] > 0.0001) ? 0.0 : x[0];
    x[1] /= scale; // x[0] = (x[1] < 0.00001 && x[1] > 0.0001) ? 0.0 : x[1];
    x[2] /= scale; // x[0] = (x[2] < 0.00001 && x[2] > 0.0001) ? 0.0 : x[2];
  } 
  else {
    x[0]=x[1]=x[2]=0.0;
    return 0;
  }
  return 1;
}//-

// |v| 
void
Math::vabs (real* v) {
  v[0] = fabs(v[0]);
  v[1] = fabs(v[1]);
  v[2] = fabs(v[2]);
}//-

// z = x + y.
void 
Math::vadd(const real *x, const real *y, real *z) {
  z[0] = x[0] + y[0];
  z[1] = x[1] + y[1];
  z[2] = x[2] + y[2];
}//-

// z = x + y
void 
Math::vadd(const real *x, const real y, real *z) {
  z[0] = x[0] + y;
  z[1] = x[1] + y;
  z[2] = x[2] + y;
}//-

// z = x - y
 void 
Math::vsub(const real *x, const real *y, real *z) {
  z[0] = x[0] - y[0];
  z[1] = x[1] - y[1];
  z[2] = x[2] - y[2];
}//-

// z = x - y
void 
Math::vsub(const real *x, const real y, real *z) {
  z[0] = x[0] - y;
  z[1] = x[1] - y;
  z[2] = x[2] - y;
}//-

// z = x - y
 void 
Math::vsub(const real x, const real* y, real *z) {
  z[0] = x - y[1];
  z[1] = x - y[2];
  z[2] = x - y[3];
}

// v = u1/u2 if u2 != 0;    0 othwise 
void 
Math::vdiv (real* u1, real* u2, real* v) {
  v[0] = (u2[0]!=0)? u1[0] / u2[0] : 0;
  v[1] = (u2[1]!=0)? u1[1] / u2[1] : 0;
  v[2] = (u2[2]!=0)? u1[2] / u2[2] : 0;
}//-

// v = u1/u2 if u2 != 0;    0 othwise 
void 
Math::vdiv (real* u1, real u2, real* v) {
  v[0] = (u2!=0)? u1[0] / u2 : 0;
  v[1] = (u2!=0)? u1[1] / u2 : 0;
  v[2] = (u2!=0)? u1[2] / u2 : 0;
}//-

// Compute the dot product of two vectors
real 
Math::vdot(real *x, real *y) {
  return (x[0]*y[0] + x[1]*y[1] + x[2]*y[2]);
}//-

// Compute the dot product of two vectors
real 
Math::vdot2(const vec3& x, const vec3& y) {
  return (x[0]*y[0] + x[1]*y[1] + x[2]*y[2]);
}//-

// Compute the cross-product of two vectors
void 
Math::vcross(real *a, real *b, real *n) {
  n[0] = a[1] * b[2] - a[2] * b[1];
  n[1] = a[2] * b[0] - a[0] * b[2];
  n[2] = a[0] * b[1] - a[1] * b[0];
}//-

// Compute the cross-product of two vectors
void 
Math::vcross2(const vec3& a, const vec3 & b, vec3& n) {
  n[0] = a[1] * b[2] - a[2] * b[1];
  n[1] = a[2] * b[0] - a[0] * b[2];
  n[2] = a[0] * b[1] - a[1] * b[0];
}//-

// Compute the middle point
void
Math::middle_point (const point& a, const point& b, point& c) {
  c[0] = (a[0] + b[0]) / 2;
  c[1] = (a[1] + b[1]) / 2;
  c[2] = (a[2] + b[2]) / 2;
}


// Given a box representatino by two points, dilatate
// its bounds by a distance d
void 
Math::dilatate (point& box_l, point& box_u, real d) {
  for (uint i=0; i<3; i++) {
    box_l[i] -= d;
    box_u[i] += d;
  }
}//-

void 
Math::dilatate (std::pair<point, point>& box, real d) {
  for (uint i=0; i<3; i++) {
    box.first[i]  -= d;
    box.second[i] += d;
  }
}//-

// check if boxA \cup boxB != \emptyset
bool
Math::does_intersect (const std::pair<point, point>& boxA, 
		      const std::pair<point, point>& boxB) { 
  if ((boxA.first[0] <= boxB.second[0] && boxA.second[0] >= boxB.first[0]) ||
      (boxA.first[1] <= boxB.second[1] && boxA.second[1] >= boxB.first[1]) ||
      (boxA.first[2] <= boxB.second[2] && boxA.second[2] >= boxB.first[2]) )
    return true;
  return false;
}//-


bool
Math::in_range (real p, real min, real max) {
  if (p >= min && p <= max)
    return true;
  return false;
}

bool
Math::in_range (const point p, const point min, const point max, real eps) {
  return (in_range(p[0], min[0]-eps, max[0]+eps) &&
	  in_range(p[1], min[1]-eps, max[1]+eps) &&
	  in_range(p[2], min[2]-eps, max[2]+eps));
}//-

bool
Math::in_range (const point p, const std::pair<point, point> bound, real eps) {
  return (in_range(p[0], bound.first[0]-eps, bound.second[0]+eps) &&
	  in_range(p[1], bound.first[1]-eps, bound.second[1]+eps) &&
	  in_range(p[2], bound.first[2]-eps, bound.second[2]+eps));
}//-

bool
Math::in_range (const point p, const point q, real eps) {
  return (in_range(p[0], q[0]-eps, q[0]+eps) &&
	  in_range(p[1], q[1]-eps, q[1]+eps) &&
	  in_range(p[2], q[2]-eps, q[2]+eps));
}//-

void
Math::round (point &p) {
  for (uint i=0; i<3; i++)
    if (p[i] < 0.0001 && p[i] > -0.0001) 
      p[i] = 0.0;
}//-


ReferenceSystem
Math::set_reference_system (point a, point b, point c) {
  ReferenceSystem R;
  vec3 x={1,0,0}, y={0,1,0}, z={0,0,1}, v;
  // Build the Orthogonal Base
  Math::vsub (b, a, v);
  Math::vsub (c, a, z); // z-axes  
  Math::vcross (z, v, y); 	// y orthogonal to z, v         (y-axis)
  Math::vcross (y, z, x);	// x orthogonal to z and y      (x-axis)
  // Normalize: Obtain the Orthonormal Base
  Math::vnorm (x);
  Math::vnorm (y);
  Math::vnorm (z);
  // Build the Rotation matrix (orthonormal)
  for (int i = 0; i < 3; i++) {
    R.orthonormal_base[i][0] = x[i] < 0.00001 ? 0 : x[i];
    R.orthonormal_base[i][1] = y[i] < 0.00001 ? 0 : y[i];
    R.orthonormal_base[i][2] = z[i] < 0.00001 ? 0 : z[i];
  }
  // Get the normal to the planes xy, xz, yz
  Math::vcross (x, y, R.normal_xy);  // z
  Math::vcross (x, z, R.normal_xz);  // y
  Math::vcross (y, z, R.normal_yz);  // x
  Math::round (R.normal_xy);
  Math::round (R.normal_xz);
  Math::round (R.normal_yz);
  
  return R;
}//-
