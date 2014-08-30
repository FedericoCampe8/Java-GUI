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
  int CurrentCol;
  int CurrentPos;
  real dx,dy,dz;
  int N;
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
  real mindis = distance[0];
  int ii;
  for (ii = 1 ; ii < N ;){
    (mindis > distance[ii]) ? (mindis = distance[ii] , ii++) : (ii++);
  }
  return mindis;
}//-

// the maximum value of distance
real 
Math::amax(const int N, const real *distance) {
  real maxdis = distance[0];
  int ii;
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
  real x = v1[0] - v2[0];
  real y = v1[1] - v2[1];
  real z = v1[2] - v2[2];
  return sqrt (x*x + y*y + z*z);
}//-

// Squared eucleudian distance 
real 
Math::eucl_dist2 (point v1, point v2) {  
  real x = v1[0] - v2[0];
  real y = v1[1] - v2[1];
  real z = v1[2] - v2[2];
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
  real scale = sqrt(x[0]*x[0] + x[1]*x[1] + x[2]*x[2]);
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


real
Math::bond_angle ( real* a, real* b, real* c ) {
  real a_vec[3];
  real b_vec[3];
  vsub( a, c, a_vec );
  vsub( b, c, b_vec );
  
  real div = vdot ( a_vec, b_vec ) / ( eucl_dist(a, c) * eucl_dist(b, c) );
  if ( div < -1 ) return 180.0;
  if ( div > 1 )  return 0;
  return acos ( div ) * 180.0 / _pi;
}//bond_angle

real
Math::torsion_angle ( real* a, real* b, real* c, real* d ) {
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
  else angle = ( acos ( angle ) ) * 180.0 / _pi;
  
  /// --- Here it is possible to return angle without sign ---
  
  // Calc the sign
  real vec_prod[3];
  vcross ( abc, bcd, vec_prod );
  real val = vdot( cb, vec_prod );
  //if ( fabs( val ) < CLOSE_TO_ZERO_VAL  ) val = 0.0;
  if ( val < close_to_zero_val ) val = 0.0;
  if ( val < 0.0 ) angle *= -1;
  
  return angle;
}//torsion_angle

void
Math::calculate_h_atom (const real *c,
                        const real *n,
                        const real *ca,
                        real *h) {
  vec3 nc, nca, added, added2;
  vsub(c, ca, nc);
  vsub(ca, n, nca);
  vcross (nc, nca, added);
  vcross (added, nc, added2);
  vnorm (added2);
  vsub (n, added2, h);
}//calculate_h_atom

void
Math::calculate_cg_atom ( aminoacid a,
                              real* ca1, real* ca2, real* ca3,
                              real* cg, int* radius ) {
  /// Placement of the centroid using dist, chi2, e tors
  /// v1 is the normalized vector w.r.t. ca1, ca2
  real v1[3];
  vsub ( ca2, ca1, v1 );
  vnorm ( v1 );
  
  /// v2 is the normalized vector w.r.t. ca2, ca3
  real v2[3];
  vsub ( ca3, ca2, v2 );
  vnorm ( v2 );
  
  /// Compute v1 (subtracting the component along v2)
  /// in order to obtain v1 and v2 orthogonal each other
  real x = vdot ( v1, v2 );
  v1[ 0 ] = v1[ 0 ] - x * v2[ 0 ];
  v1[ 1 ] = v1[ 1 ] - x * v2[ 1 ];
  v1[ 2 ] = v1[ 2 ] - x * v2[ 2 ];
  vnorm ( v1 );
  
  /// Compute v3 orthogonal to v1 and v2
  real v3[3];
  vcross ( v1, v2, v3 );
  
  /// Using Cramer method
  real factor;
  real b[3];
  real R[3][3];
  real D, Dx, Dy, Dz;
  real tors   = centroid_torsional_angle ( a ) * _pi/180;
  b[0] = cos( (centroid_chi2 ( a )) * _pi/180 );
  factor = sqrt( 1 - ( b[0] * b[0] ) );
  b[1] = sin( tors ) * factor ;
  b[2] = cos( tors ) * factor ;
  
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
  
  real v[3];
  v[ 0 ] = Dx/D;
  v[ 1 ] = Dy/D;
  v[ 2 ] = Dz/D;
  
  /// Now compute centroids coordinates
  v[ 0 ] = centroid_distance( a ) * v[ 0 ];
  v[ 1 ] = centroid_distance( a ) * v[ 1 ];
  v[ 2 ] = centroid_distance( a ) * v[ 2 ];
  
  // Update the output
  vadd ( v, ca2, cg );
  *radius = centroid_radius( a );
}//calculate_cg_atom

real
Math::centroid_torsional_angle ( aminoacid a ) {
  if (a==ala) return -138.45;
  if (a==arg) return -155.07;
  if (a==asn) return -144.56;
  if (a==asp) return -142.28;
  if (a==cys) return -142.28;
  if (a==gln) return -149.99;
  if (a==glu) return -147.56;
  if (a==gly) return -0;
  if (a==his) return -144.08;
  if (a==ile) return -151.72;
  if (a==leu) return -153.24;
  if (a==lys) return -153.03;
  if (a==met) return -159.50;
  if (a==phe) return -146.92;
  if (a==pro) return -105.62;
  if (a==ser) return -139.94;
  if (a==thr) return -142.28;
  if (a==trp) return -155.35;
  if (a==tyr) return -149.29;
  if (a==val) return -150.47;
  return 0;
}//centroid_torsional_angle

real
Math::centroid_chi2 ( aminoacid a ) {
  if (a==ala) return 110.53;
  if (a==arg) return 113.59;
  if (a==asn) return 117.73;
  if (a==asp) return 116.03;
  if (a==cys) return 115.36;
  if (a==gln) return 115.96;
  if (a==glu) return 115.98;
  if (a==gly) return 0;
  if (a==his) return 115.38;
  if (a==ile) return 118.17;
  if (a==leu) return 119.90;
  if (a==lys) return 115.73;
  if (a==met) return 115.79;
  if (a==phe) return 114.40;
  if (a==pro) return 123.58;
  if (a==ser) return 110.33;
  if (a==thr) return 111.67;
  if (a==trp) return 109.27;
  if (a==tyr) return 113.14;
  if (a==val) return 114.46;
  return 0;
}//centroid_chi2

real
Math::centroid_distance ( aminoacid a ) {
  if (a==ala) return 1.53;
  if (a==arg) return 3.78;
  if (a==asn) return 2.27;
  if (a==asp) return 2.24;
  if (a==cys) return 2.03;
  if (a==gln) return 2.85;
  if (a==glu) return 2.83;
  if (a==gly) return 0;
  if (a==his) return 3.01;
  if (a==ile) return 2.34;
  if (a==leu) return 2.62;
  if (a==lys) return 3.29;
  if (a==met) return 2.95;
  if (a==phe) return 3.41;
  if (a==pro) return 1.88;
  if (a==ser) return 1.71;
  if (a==thr) return 1.94;
  if (a==trp) return 3.87;
  if (a==tyr) return 3.56;
  if (a==val) return 1.97;
  return 0;
}//centroid_distance

int
Math::centroid_radius ( aminoacid a ) {
  if (a==ala) return 190;
  if (a==arg) return 280;
  if (a==asn) return 222;
  if (a==asp) return 219;
  if (a==cys) return 213;
  if (a==gln) return 241;
  if (a==glu) return 238;
  if (a==gly) return 120;
  if (a==his) return 249;
  if (a==ile) return 249;
  if (a==leu) return 249;
  if (a==lys) return 265;
  if (a==met) return 255;
  if (a==phe) return 273;
  if (a==pro) return 228;
  if (a==ser) return 192;
  if (a==thr) return 216;
  if (a==trp) return 299;
  if (a==tyr) return 276;
  if (a==val) return 228;
  return 100; // default
}//centroid_radius
