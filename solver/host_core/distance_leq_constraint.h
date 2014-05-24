/*********************************************************************
 * DistanceLEQ Constraint
 * Binary constraint over two variables V_1 V_2 which is satisfied iff
 * || V_1 - V_2 || <= d
 * where ||.|| is the norm 2
 *        d \in Real is the distance measure
 *
 * Input Syntax
 * --distance-geq aa_1 [ATOM_TYPE] aa_2 [ATOM_TYPE] d
 * where:
 *   - aa_1, aa_2 are the amino acid involved in the constraint 
 *     (index starts from 0)
 *   - ATOM_TYPE denotes the atoms over which the constraint is applied
 *     (default is CA). ATOM_TYPE = {CA, N, O, CB}
 *   - d is the distance.
 *********************************************************************/

#ifndef FIASCO_DISTANCE_LEQ_CONSTRAINT_
#define FIASCO_DISTANCE_LEQ_CONSTRAINT_

#include "constraint.h"

class DistanceLEQConstraint : public Constraint {
 private:
  real squared_distance;

 public:
  DistanceLEQConstraint(int argc, char* argv[], int& parse_pos, int weight = 3);
  ~DistanceLEQConstraint();

  bool propagate(int trailtop);
  bool consistency();
  bool check_cardinality (size_t& backjump); 
  bool synergic_consistency (const point& p, atom_type t = CA, int aa_idx = 0);
  void reset_synergy() {};
  void dump(bool all=true);

};//-

#endif
