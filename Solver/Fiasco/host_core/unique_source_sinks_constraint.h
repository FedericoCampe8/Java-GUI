/*********************************************************************
 * object: Unique Source (Mulitple) Sinks constraint
 * 
 * The unique-source-sinks constraints between the varaibles V_i 
 * and V_j is a cardinaliry constraints which given an assignement for 
 * V_i, ensures that in the solutions pool there is at most one 
 * assignemtn for V_j, for each grid voxel of G_j.
 * This constraint associates a grid to the variable V_j.
 *
 * Semantics:
 * Fixed V_i, \not\exists V_j^m, V_j^n s.t.
 *               find_voxel(V_j^m) = find_voxel(V_j^n)
 * where:
 *   - V_j^{k} represents the assignment for variable V_j in the 
 *     k-th solution generated.
 *   - find_voxel : V -> \mathbb{N} is a function which associates
 *     each variable assignemnt with its grid voxel index (if any).
 * 
 * Input Syntax:
 * --unique-source-sinks a_i '->' a_j : voxel-side= K 
 * where:
 *   - a_i, is the source atom (the CA_i, with i >= 0)  
 *   - a_j, is the sink atom (the CA_j, with i >= 0)
 *   - K, is the side of a voxel in Angstroms.
 *
 * NOTE:
 * The lattice side is now fixed to 200 Angstroms. If you experience
 * any problem with the lattice size, please enlarge the lattice side.
 *
 *********************************************************************/
#ifndef FIASCO_UNIQUE_SOUCRE_SINKS_CONSTRAINT_
#define FIASCO_UNIQUE_SOUCRE_SINKS_CONSTRAINT_

#include "typedefs.h"
#include "constraint.h"

class CubicLattice;

class UniqueSourceSinksConstraint : public Constraint {
 private:
  CubicLattice* grid;
  int source, sink;

 public:
  UniqueSourceSinksConstraint  (int argc, char* argv[], int& parse_pos, int weight = 3);
  ~UniqueSourceSinksConstraint (); 
  
  bool propagate (int trailtop); 
  bool consistency ();
  bool check_cardinality (size_t& backjump);  
  bool synergic_consistency (const point& p, atom_type t = CA, int aa_idx = 0);
  void reset_synergy();
  void dump (bool all = true);
};
  
#endif 
