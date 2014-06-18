#include "atom.h"
#include "typedefs.h"
#include "globals.h"
#include "mathematics.h"
#include "utilities.h"
#include "protein.h"

#include <cmath>
#include <string.h>

using namespace Math;
using namespace Utilities;
using namespace std;


Atom::Atom(real x, real y, real z, atom_type t, int idx){
  position[0] = x;
  position[1] = y;
  position[2] = z;
  ref_aa = idx;
  type = t;
  set_radius(t);
}//-

Atom::Atom(real x, real y, real z, atom_type t){
  position[0] = x;
  position[1] = y;
  position[2] = z;
  ref_aa = -1;
  type = t;
  set_radius(t);
}//-

Atom::Atom(const point& p, atom_type t){
  position[0] = p[0];
  position[1] = p[1];
  position[2] = p[2];
  ref_aa = -1;
  type = t;
  set_radius(t);
}//-

Atom::Atom(const Atom& other) {
  memcpy(position, other.position, sizeof(point));
  type = other.type;
  radius = other.radius;
  ref_aa = other.ref_aa;
}//-

Atom&
Atom::operator= (const Atom& other) {
  if (this != &other) {
    memcpy(position, other.position, sizeof(point));
    type = other.type;
    radius = other.radius;
    ref_aa = other.ref_aa;
  }
  return *this;
}//-

real
Atom::operator[] (const int index) const{
  if (index >= 0 && index < 3) 
    return position[index];
  return -1;
}//[]

void 
Atom::set_position (point p) {
  position[0] = p[0];
  position[1] = p[1];
  position[2] = p[2];
}//set_position

void 
Atom::set_position (real x, real y, real z) {
  position[0] = x;
  position[1] = y;
  position[2] = z;
}//set_position

void
Atom::set_type(atom_type t) {
  type = t;
  set_radius(t);
}//set_type

void 
Atom::set_radius (atom_type t) {
  switch (t) {
  case H: radius = rH;
    break;
  case CA: radius = rC;
    break;
  case CB: radius = rC;
    break;
  case O: radius = rO; 
    break;
  case N: radius = rN;
    break;
  case S: radius = rS;
    break;
  case CG:
    radius = centroid_radius (g_target.sequence[ref_aa-g_target.sequence.length()]);
    break;
  case X: radius = rC;
    break;
  }
}//set_radius

real 
Atom::centroid_distance (char a){
  if (a=='a' || a=='A') return 1.55;
  if (a=='r' || a=='R') return 4.16;
  if (a=='n' || a=='N') return 2.55;
  if (a=='d' || a=='D') return 2.55;
  if (a=='c' || a=='C') return 2.38;
  if (a=='q' || a=='Q') return 3.13;
  if (a=='e' || a=='E') return 3.18;
  if (a=='g' || a=='G') return 0;
  if (a=='h' || a=='H') return 3.19;
  if (a=='i' || a=='I') return 2.31;
  if (a=='l' || a=='L') return 2.63;
  if (a=='k' || a=='K') return 3.56;
  if (a=='m' || a=='M') return 3.27;
  if (a=='f' || a=='F') return 3.41;
  if (a=='p' || a=='P') return 1.85;
  if (a=='s' || a=='S') return 1.95;
  if (a=='t' || a=='T') return 1.95;
  if (a=='w' || a=='t') return 3.98;
  if (a=='y' || a=='Y') return 3.88;
  if (a=='v' || a=='V') return 1.95;
  return 0;	
}//centroid_distance

real 
Atom::centroid_chi2 (char a) {
  if (a=='a' || a=='A') return 110.53;
  if (a=='r' || a=='R') return 113.59;
  if (a=='n' || a=='N') return 117.73;
  if (a=='d' || a=='D') return 116.03;
  if (a=='c' || a=='C') return 115.36;
  if (a=='q' || a=='Q') return 115.96;
  if (a=='e' || a=='E') return 115.98;
  if (a=='g' || a=='G') return 0;
  if (a=='h' || a=='H') return 115.38;
  if (a=='i' || a=='I') return 118.17;
  if (a=='l' || a=='L') return 119.90;
  if (a=='k' || a=='K') return 115.73;
  if (a=='m' || a=='M') return 115.79;
  if (a=='f' || a=='F') return 114.40;
  if (a=='p' || a=='P') return 123.58;
  if (a=='s' || a=='S') return 110.33;
  if (a=='t' || a=='T') return 111.67;
  if (a=='w' || a=='W') return 109.27;
  if (a=='y' || a=='Y') return 113.14;
  if (a=='v' || a=='V') return 114.46;
  return 0;	
}//chi2

real 
Atom::centroid_torsional_angle (char a){
  if (a=='a' || a=='A') return -138.45;
  if (a=='r' || a=='R') return -155.07;
  if (a=='n' || a=='N') return -144.56;
  if (a=='d' || a=='D') return -142.28;
  if (a=='c' || a=='C') return -142.28;
  if (a=='q' || a=='Q') return -149.99;
  if (a=='e' || a=='E') return -147.56;
  if (a=='g' || a=='G') return -0;
  if (a=='h' || a=='H') return -144.08;
  if (a=='i' || a=='I') return -151.72;
  if (a=='l' || a=='L') return -153.24;
  if (a=='k' || a=='K') return -153.03;
  if (a=='m' || a=='M') return -159.50;
  if (a=='f' || a=='F') return -146.92;
  if (a=='p' || a=='P') return -105.62;
  if (a=='s' || a=='S') return -139.94;
  if (a=='t' || a=='T') return -142.28;
  if (a=='w' || a=='W') return -155.35;
  if (a=='y' || a=='Y') return -149.29;
  if (a=='v' || a=='V') return -150.47;
  return 0;	
}//centroid_torsional_angle

atom_radii 
Atom::centroid_radius (char a){
  if (a=='a' || a=='A') return r_a;//2.40;
  if (a=='r' || a=='R') return r_r;//3.23;
  if (a=='n' || a=='N') return r_n;//3.40;
  if (a=='d' || a=='D') return r_d;//4.03;
  if (a=='c' || a=='C') return r_c;//4.26;
  if (a=='q' || a=='Q') return r_q;//2.40;
  if (a=='e' || a=='E') return r_e;//4.04;
  if (a=='g' || a=='G') return r_g;//3.16;
  if (a=='h' || a=='H') return r_h;//4.41;
  if (a=='i' || a=='I') return r_i;//3.48;
  if (a=='l' || a=='L') return r_l;//4.12;
  if (a=='k' || a=='K') return r_k;//3.40;
  if (a=='m' || a=='M') return r_m;//2.70;
  if (a=='f' || a=='F') return r_f;//3.98;
  if (a=='p' || a=='P') return r_p;//2.80;
  if (a=='s' || a=='S') return r_s;//2.80;
  if (a=='t' || a=='T') return r_t;//5.01;
  if (a=='w' || a=='W') return r_w;//2.80;
  if (a=='y' || a=='Y') return r_y;//4.83;
  if (a=='v' || a=='V') return r_v;//4.73;
  return rCG; // default
}//centroid_radius

// Compute the centroid coordinates and its radius.
void 
Atom::compute_cg(const char a, const point &ca1, const point &ca2,
		 const point &ca3) {
  vec3 v1,v2,v3,v,b; 
  R_MAT R;
  real chi2, tors, dist, x;
  int i;
  register real D, Dx, Dy, Dz;
  // Placement of the centroid using dist, chi2, e tors
  chi2 = centroid_chi2 (a);
  tors = centroid_torsional_angle (a);
  dist = centroid_distance(a);

  // v1 is the normalized vector w.r.t. ca1, ca2
  Math::vsub (ca2, ca1, v1);
  Math::vnorm (v1);
  // v2 is the normalized vector w.r.t. ca2, ca3
  Math::vsub (ca3, ca2, v2); 
  Math::vnorm (v2); 	

  // Compute v1 (subtracting the component along v2) so that to 
  // obtain v1 and v2 orthogonal each other
  x = Math::vdot (v1, v2);
  for (i = 0; i < 3; i++)
    v1[i] = v1[i] - x*v2[i];
  Math::vnorm (v1);

  // compute v3 orthogonal to v1 and v2
  Math::vcross (v1, v2, v3);
  
  // Using Cramer method:
  // Let v = {x,y,z} be the 1-norm vector w.r.t. ca2 and cg 
  // Let the system:
  //  v2*v = cos(chi2)
  //  -v1*v = cos(tors) * sqrt(1 - cos^2(chi2))
  //  v3*v = sin(tors) * sqrt(1 - cos^2(chi2))
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
  for(i = 0; i < 3; i++)
    v[i] = dist * v[i];
  // Update the output
  Math::vadd (v, ca2, position);
  //radius = centroid_radius (a);
}//-

void
Atom::dump() {
  cout << "Atom ";
  switch (type) {
  case H: 
    cout << "H  ";
    break;
  case CA: 
    cout << "Ca ";
    break;
  case CB: 
    cout << "C' ";
    break;
  case O: 
    cout << "O  ";
    break;
  case N: 
    cout << "N  ";
    break;
  case S: 
    cout << "S  ";
    break;
  case CG: 
    cout << "CG ";
    break;
  default:
    cout << "X  ";
  }
  cout << "<" << position[0] << ", " << position[1] << ", " << position[2] << ">\t";
  cout << "r=" << (real)radius/100 << endl;
}//-
