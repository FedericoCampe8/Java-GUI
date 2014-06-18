/*********************************************************************
 * object: End Anchors Different Constraint
 * 
 * Input Syntax
 * --end-anchor-diff aa voxel-side= K 
 * where:
 *   - aa, is the amino acid (starting from 0) which anchor will be 
 *     involved in the constraint
 *   - K, is the side of a voxel in Angstroms.
 *
 * NOTE:
 * The lattice side is now fixed to 200 Angstroms. If you experience
 * any problem with the lattice size, please enlarge the lattice side.
 *
 *********************************************************************/
#ifndef FIASCO_END_ANCHOR_DIFF_CONSTRAINT_
#define FIASCO_END_ANCHOR_DIFF_CONSTRAINT_

#include "typedefs.h"
#include "constraint.h"

class CubicLattice;

class EndAnchorDifferentConstraint : public Constraint {
 private:
  CubicLattice* grid;
   
 public:
  EndAnchorDifferentConstraint  (int argc, char* argv[], int& parse_pos, int weight = 4);
  ~EndAnchorDifferentConstraint (); 
  
  bool propagate (int trailtop); 
  bool consistency ();
  bool check_cardinality (size_t& backjump);  
  bool synergic_consistency (const point& p, atom_type t = CA, int aa_idx = 0);
  void reset_synergy();
  void dump (bool all = true);
};
  
#endif 
