#include "atom.h"
#include "math.h"

using namespace std;

Atom::Atom ( real x, real y, real z, atom_type t, int idx ) :
ref_aa ( idx ),
type   ( t ) {
  position[0] = x;
  position[1] = y;
  position[2] = z;
  set_radius ( t );
}//-

Atom::Atom ( real* p, atom_type t ) :
ref_aa ( -1 ),
type   ( t ) {
  position[0] = p[0];
  position[1] = p[1];
  position[2] = p[2];
  set_radius ( t );
}//-

Atom::Atom( const Atom& other ) {
  ref_aa      = other.ref_aa;
  type        = other.type;
  radius      = other.radius;
  position[0] = other.position[0];
  position[1] = other.position[1];
  position[2] = other.position[2];
}//_

Atom&
Atom::operator= ( const Atom& other ){
  ref_aa      = other.ref_aa;
  type        = other.type;
  radius      = other.radius;
  position[0] = other.position[0];
  position[1] = other.position[1];
  position[2] = other.position[2];
  return *this;
}//=

real
Atom::operator[] ( const int index ) const {
  if ( index >= 0 && index < 3 )
    return position[ index ];
  return -1;
}//[]

void
Atom::set_type ( atom_type t ) {
  type = t;
  set_radius ( t );
}//set_type

void 
Atom::set_radius ( atom_type t ) {
  switch (t) {
    case H:
      radius = rH;
      break;
    case CA:
      radius = rC;
      break;
    case CB:
      radius = rC;
      break;
    case O:
      radius = rO;
      break;
    case N:
      radius = rN;
      break;
    case S:
      radius = rS;
      break;
    case CG:
      radius = rCG;
      break;
    case X:
      radius = rC;
      break;
  }//switch
}//set_radius

void 
Atom::set_position ( real* p ) {
  ( fabs((double)p[0] ) < CLOSE_TO_ZERO_VAL ) ?
  position[0] = 0.0 : position[0] = p[0];
  
  ( fabs((double)p[1] ) < CLOSE_TO_ZERO_VAL ) ?
  position[1] = 0.0 : position[1] = p[1];
  
  ( fabs((double)p[2] ) < CLOSE_TO_ZERO_VAL ) ?
  position[2] = 0.0 : position[2] = p[2];
}//set_position

void 
Atom::set_position ( real x, real y, real z ) {
  real p[3];
  p[0] = x; p[1] = y; p[2] = z;
  set_position ( p );
}//set_position

bool
Atom::is_type ( atom_type t ) const {
  return ( type == t ) ? true : false;
}//is_type

void
Atom::dump() {
  cout << "Atom ";
  switch ( type ) {
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
  cout << "<" <<
  position[0] << ", " <<
  position[1] << ", " <<
  position[2] << ">\t";
  cout << "r=" << (real) radius/100 << endl;
}//dump
