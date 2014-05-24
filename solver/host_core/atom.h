/********************************************************************* 
 * Atom
 *********************************************************************/
#ifndef FIASCO_ATOM__
#define FIASCO_ATOM__

#include "typedefs.h"

#include <iostream>

/* Atom - all members are public */
class Atom {
 public:
  point position; // x,y,z coordinates
  atom_type type;         // atom type
  atom_radii radius;
  int ref_aa;  // Used to skip tests for atoms belonging to the same
               // aa in grid constr (ca and cg may be very close!)

  Atom(){};
  Atom(real x, real y, real z, atom_type t);
  Atom(real x, real y, real z, atom_type t, int idx);
  Atom(const point& p, atom_type t); 
  ~Atom() {};
  Atom (const Atom& other);
  Atom& operator= (const Atom& other);
  real operator[]  (const int index) const;
  bool is_type (atom_type t) const {return (type == t) ? true : false;}

  void set_type(atom_type t);
  void set_radius(atom_type t);
  void set_position(point p);
  void set_position(real x, real y, real z);

  void compute_cg(const char a, const point &ca1, const point &ca2,
		  const point &ca3);
  
  atom_radii centroid_radius (char a);
  real centroid_torsional_angle (char a);
  real centroid_chi2 (char a);
  real centroid_distance (char a);
  void dump ();
};

#endif
