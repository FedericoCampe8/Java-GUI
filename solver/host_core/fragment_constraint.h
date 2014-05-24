/*********************************************************************
 * Fragment Constraint
 * Constrain the positions of a sequence of atoms (k amino-acids, 
 * k \geq 1). Use fragment Assembly technique.
 *
 * references: 
 * A. Dal Palù, A. Dovier, F. Fogolari, and E. Pontelli CLP-based 
 * protein fragment assembly. Theory and Practice of Logic Programming, 
 * special issue dedicated to ICLP 2010. 10(4-6): pp 709-724. 
  *********************************************************************/
#ifndef FIASCO_FRAGMENT_CONSTRAINT_
#define FIASCO_FRAGMENT_CONSTRAINT_

#include "constraint.h"
#include "fragment.h" // for AssemblyDirection

class VariableFragment;

class ConstraintFragment : public Constraint {
 private:
    AssemblyDirection overlap_plane;

 public:
 ConstraintFragment (VariableFragment *f_ptr, 
		     AssemblyDirection dir = LEFT_TO_RIGHT);
  
  bool propagate(int trailtop);
  bool consistency();
  bool check_cardinality (size_t& backjump) {return true; }
  bool synergic_consistency 
    (const point& p, atom_type t = CA, int aa_idx = 0) {return true; }
  void reset_synergy() {};
  void dump(bool all=true);
};//-


#endif
