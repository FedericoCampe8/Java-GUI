#include "typedefs.h"
#include "mathematics.h"
#include "string.h"
#include <cmath>

real 
Math::min (real a, real b) {
    return ( a < b ) ? a : b;
}//min

real 
Math::max (real a, real b) {
    return ( a > b ) ? a : b;
}//max

/* 
 *Compute the distance from the current point to previous point (left hand size points)
 */
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
}//distance

/* Find the minimum distance from a point to other points */
real 
Math::amin(const int N, const real *distance){
    register real mindis = distance[0];
    register int ii;
    for (ii = 1 ; ii < N ;){
        (mindis > distance[ii]) ? (mindis = distance[ii] , ii++) : (ii++);
    }
    return mindis;
}//amin

/* Find the maximum value of distance from a point to other points */
real 
Math::amax(const int N, const real *distance) {
    register real maxdis = distance[0];
    register int ii;
    for (ii = 1 ; ii < N ;){
        (maxdis < distance[ii]) ? ( maxdis = distance[ii] , ii++) : (ii++);
    }
    return maxdis;
}//amax

/* Abs for real */
real 
Math::abs (const real x) {
  return (x >= 0)? x : -x;
}//abs

/* Compute the exponential moving average */
real 
Math::exponential_moving_avg (real alpha,real current_val, real prev_avg){
  return (alpha*current_val) + ((1 - alpha)*prev_avg);
}//exponential_moving_avg


real
Math::truncate_number ( real n, int n_of_decimals ) {
  int ten_to = (int) pow ( 10.0, n_of_decimals );
  double intpart, fractpart;
  fractpart =  modf ( n , &intpart);
  fractpart *= ten_to;
  fractpart =  floor ( fractpart );
  fractpart /= ten_to;
  
  return intpart + fractpart;
}//truncate_number



/***************************************************************************/
/* Matrix/Vectors Operation */
/***************************************************************************/

void 
Math::set_identity(R_MAT &m){
    m[0][0] = m[1][1] = m[2][2] = 1;
    m[0][1] = m[0][2] = m[1][0] = m[1][2] = m[2][0] = m[2][1] = 0;
}//set_identity

void 
Math::set_identity(vec3 &v){
    v[0] = v[1] = v[2] = 0;
}//set_identity

void
Math::translate(point& p, const vec3& v) {
  p[0]+=v[0]; p[1]+=v[1]; p[2]+=v[2];
}//translate

void
Math::rotate(point& p, const R_MAT& rot_m) {
    point _p;
    memcpy(&_p, &p, sizeof(point));
    p[0] = rot_m[0][0] * _p[0] + rot_m[0][1] * _p[1] + rot_m[0][2] * _p[2];
    p[1] = rot_m[1][0] * _p[0] + rot_m[1][1] * _p[1] + rot_m[1][2] * _p[2];   
    p[2] = rot_m[2][0] * _p[0] + rot_m[2][1] * _p[1] + rot_m[2][2] * _p[2]; 
}//rotate

void
Math::rotate_inverse(point& p, const R_MAT& rot_m) {
    point _p;
    memcpy(&_p, &p, sizeof(point));
    p[0] = rot_m[0][0] * _p[0] + rot_m[1][0] * _p[1] + rot_m[2][0] * _p[2];
    p[1] = rot_m[0][1] * _p[0] + rot_m[1][1] * _p[1] + rot_m[2][1] * _p[2];   
    p[2] = rot_m[0][2] * _p[0] + rot_m[1][2] * _p[1] + rot_m[2][2] * _p[2]; 
}//rotate_inverse

/* Calculate the eucleudian distance between two vectors */
real 
Math::eucl_dist (point v1, point v2) {
    register real x = v1[0] - v2[0];
    register real y = v1[1] - v2[1];
    register real z = v1[2] - v2[2];
    return sqrt (x*x + y*y + z*z);
}//eucl_dist

/* Squared eucleudian distance */
real 
Math::eucl_dist2 (point v1, point v2) {  
    register real x = v1[0] - v2[0];
    register real y = v1[1] - v2[1];
    register real z = v1[2] - v2[2];
    return (x*x + y*y + z*z);
}//eucl_dist2

/* Calculate the nrm2 of a given vector -- magnitude of a vector */
real 
Math::vnorm2(real *x) {
    return sqrt(x[0]*x[0] + x[1]*x[1] + x[2]*x[2]);
}//vnorm2

/* Normalize a given vector by its Frobenius norm */
int 
Math::vnorm(real *x) {
    register real scale = sqrt(x[0]*x[0] + x[1]*x[1] + x[2]*x[2]);
    if (scale > 0){
        x[0] /= scale;
        x[1] /= scale;
        x[2] /= scale;
    } 
    else {
        x[0] = x[1] = x[2] = 0.0;
        return 0;
    }
    return 1;
}//vnorm

/* |v| */
void
Math::vabs (real* v) {
    v[0] = fabs(v[0]);
    v[1] = fabs(v[1]);
    v[2] = fabs(v[2]);
}//vabs

/* z = x + y */
void 
Math::vadd(const real *x, const real *y, real *z) {
    z[0] = x[0] + y[0];
    z[1] = x[1] + y[1];
    z[2] = x[2] + y[2];
}//vadd

/* z = x + y */
void 
Math::vadd(const real *x, const real y, real *z) {
    z[0] = x[0] + y;
    z[1] = x[1] + y;
    z[2] = x[2] + y;
}//vadd

/* z = x - y */
void 
Math::vsub(const real *x, const real *y, real *z) {
  z[0] = x[0] - y[0];
  z[1] = x[1] - y[1];
  z[2] = x[2] - y[2];
}//vsub

/* z = x - y */
void 
Math::vsub(const real *x, const real y, real *z) {
    z[0] = x[0] - y;
    z[1] = x[1] - y;
    z[2] = x[2] - y;
}//vsub

/* z = x - y */
void 
Math::vsub(const real x, const real* y, real *z) {
    z[0] = x - y[1];
    z[1] = x - y[2];
    z[2] = x - y[3];
}//vsub

/* v = u1/u2 if u2 != 0; */
void vdiv (real* u1, real* u2, real* v) {
    v[0] = (u2[0]!=0)? u1[0] / u2[0] : 0;
    v[1] = (u2[1]!=0)? u1[1] / u2[1] : 0;
    v[2] = (u2[2]!=0)? u1[2] / u2[2] : 0;
}//vdiv

/* v = u1/u2 if u2 != 0; */ 
void 
Math::vdiv (real* u1, real u2, real* v) {
    v[0] = (u2!=0)? u1[0] / u2 : 0;
    v[1] = (u2!=0)? u1[1] / u2 : 0;
    v[2] = (u2!=0)? u1[2] / u2 : 0;
}//vdiv

/* Compute the dot product of two vectors */
real 
Math::vdot(real *x, real *y) {
    return (x[0]*y[0] + x[1]*y[1] + x[2]*y[2]);
}//vdot

/* 
 * Compute the cross-product of two vectors
 * @note: 
 * a × b = [a2b3 − a3b2, a3b1 − a1b3, a1b2 − a2b1]
 */
void 
Math::vcross(real *a, real *b, real *n) {
  n[0] = a[1] * b[2] - a[2] * b[1];
  n[1] = a[2] * b[0] - a[0] * b[2];
  n[2] = a[0] * b[1] - a[1] * b[0];
}//vcross


real
Math::bond_angle ( real* a, real* b, real* c ) {
  real a_vec[3];
  real b_vec[3];
  vsub( a, c, a_vec );
  vsub( b, c, b_vec );
  
  real div = vdot ( a_vec, b_vec ) / ( eucl_dist(a, c) * eucl_dist(b, c) );
  if ( div < -1 ) return 180.0;
  if ( div > 1 )  return 0;
  return acos ( div ) * 180.0 / PI_VAL;
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
  else angle = ( acos ( angle ) ) * 180.0 / PI_VAL;
  
  /// --- Here it is possible to return angle without sign ---
  
  // Calc the sign
  real vec_prod[3];
  vcross ( abc, bcd, vec_prod );
  real val = vdot( cb, vec_prod );
  if ( val < CLOSE_TO_ZERO_VAL ) val = 0.0;
  if ( val < 0.0 ) angle *= -1;
  return angle;
}//torsion_angle





