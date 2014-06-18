/*********************************************************************
 * Centroid Constraint
 * This contraint it is used to model the simplified representation of 
 * the amino acid side chain. A centroid 'CG_k' is constrained to a 
 * specific position, when the three consecutive amino acid CA 
 * positions: CA_{k-1}, CA_{k} CA_{k+1} are known.
 *
 * references: 
 * A. Dal Palù, A. Dovier, F. Fogolari, and E. Pontelli CLP-based 
 * protein fragment assembly. Theory and Practice of Logic Programming, 
 * special issue dedicated to ICLP 2010. 10(4-6): pp 709-724. 
  *********************************************************************/
#ifndef FIASCO_CENTROID_CONSTRAINT_
#define FIASCO_CENTROID_CONSTRAINT_

#include "constraint.h"

class VariablePoint; 

class CentroidConstraint : public Constraint {
 public:
  CentroidConstraint (VariablePoint *p1_ptr, 
		      VariablePoint *p2_ptr,
		      VariablePoint *p3_ptr);

  bool propagate(int trailtop);
  bool consistency();
  bool check_cardinality (size_t& backjump) {return true; }
  bool synergic_consistency 
    (const point& p, atom_type t = CA, int aa_idx = 0) {return true; }
  void reset_synergy() {};
  void dump(bool all=true);
};//- 

#endif
