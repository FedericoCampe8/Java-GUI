/*********************************************************************
 * object: Ellipsoid Constraint
 * Ellipsoid constraint is a spatial constraint over an atom A, which 
 * is satisfied iff A falls inside the specified ellipsoid. 
 *
 * Input Syntax
 * --ellipsoid aa_1 .. aa_n : f1= X Y Z f2= X Y Z sum-radii= K
 * where:
 *   - aa_1 .. aa_n, is a list of amino acids (starting from 0) 
 *     for which CAs will be involved in the constraint
 *   - f1 and f2 X Y Z, are the coordinates for the two focus
 *   - K, is the radius sum

 * TODO:
 * Include parser for syntax like:
 * --ellipsoid-constraint(markers=(, , , ), focal1=(x,y,z), 
 *   focal2=(x,y,z) radsum=K)*
 * where:
 *  - markers is a list of atoms involed in the constarint,
 *  - focal1 and focal2 are the two focals of the ellipsoid
 *  - sumrad is the sum of the two radii.
 * 
 * TODO:
 * Include parser
 * Change constructor parameters
 * override cardinality constraint 
 *********************************************************************/
#ifndef FIASCO_ELLIPSOID_CONSTRAINT_
#define FIASCO_ELLIPSOID_CONSTRAINT_

#include "typedefs.h"
#include "constraint.h"

class OptionsParser;

class EllipsoidConstraint : public Constraint {
 private:
  real a, b, c;
  real a2, b2, c2; // squared 
  point center;

 public:
  EllipsoidConstraint (const OptionsParser &opts, int weight = 5); // TODO
  EllipsoidConstraint (int argc, char* argv[], int& parse_pos, 
		       const R_MAT& rot_ellipsoid, const vec3& sh_ellispoid, 
		       int weight = 5); // TEMP PARSER SOLUTION
  EllipsoidConstraint (VariablePoint *p_ptr, point _f1, point _f2, real _sum_radii, int weight = 5);
  ~EllipsoidConstraint () {};
  
  bool propagate(int trailtop);
  bool consistency();
  bool check_cardinality (size_t& backjump) {return true; }//TODO
  bool synergic_consistency 
    (const point& p, atom_type t = CA, int aa_idx = 0) {return true; }
  void reset_synergy() {};
  void dump(bool all=true);  

};//-

  
#endif 

