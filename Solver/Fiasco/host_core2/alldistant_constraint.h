/*********************************************************************
 * Alldistant Constraint
 * Is a global constraint used to ensure that no staric clashes would 
 * be made over the protein backbone. 
 * This constraint is active by DEFAULT 
 *********************************************************************/

#ifndef FIASCO_ALLDISTANT_CONSTRAINT_
#define FIASCO_ALLDISTANT_CONSTRAINT_

#include "constraint.h"

class AlldistantConstraint : public Constraint {
 public:
  AlldistantConstraint();

  bool propagate(int trailtop);
  bool consistency();
  bool check_cardinality (size_t& backjump) {return true; } 
  bool synergic_consistency (const point& p, atom_type t = CA, int aa_idx = 0);
  void reset_synergy() {};
  void dump(bool all=true);

};//-

#endif
