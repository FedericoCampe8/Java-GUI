/*********************************************************************
 * Bundle Constraint
 * Constraint multiple fragments relative positions and/or orientations.
 *********************************************************************/
#ifndef FIASCO_BUNDLE_CONSTRAINT_
#define FIASCO_BUNDLE_CONSTRAINT_

#include "constraint.h"
#include <iostream>

class VariableFragment;


class BundleConstraint : public Constraint {
 private:
  bool use_bidirectional;

 public:
  // <label F1, label F2>
  // std::pair<int, int> bundle_info; // HACK! sobstuite this with a vect!
  
  BundleConstraint (std::vector<VariableFragment*> vf_ptr); 
  bool propagate (int trailtop);
  bool consistency ();
  bool check_cardinality (size_t& backjump) {return true; }
  bool synergic_consistency 
    (const point& p, atom_type t = CA, int aa_idx = 0) {return true; }
  void reset_synergy() {};
  void dump (bool all=true);
};//-


#endif
