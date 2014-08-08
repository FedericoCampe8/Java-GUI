/*********************************************************************
 * object: Unique Sequence Constraint
 * The UNIQUE constraint is a spatial constraint which guarantees that
 * a given discretization for a chain of adjacent points is visited by 
 * at most one solution.

 * Consider a 3D lattice approximating the 3D space. We say that 
 * \gamma(aa_i) gives the lattice position for the amino acid aa_i. 
 * An assignment (x_1, ..., x_n) of variables P_1, .., P_n satisfy the 
 * UNIFORM constraint over a sequence of amino acids aa_1, .., aa_n, 
 * if there is no other solution SOL_k s.t. (\gamma_k(aa_1) = x_1 \and
 * ... \and \gamma_k(aa_n) = x_n.
 * 
 * Input Syntax
 * --unique-seq aa_1 .. aa_n : voxel-side= K (INCR?)
 * where:
 *   - aa_1 .. aa_n, is a list of amino acids (starting from 0) 
 *     for which CAs will be involved in the uniform constraint
 *     (every a_i will be placed using one grid)
 *   - K, is the side of a voxel in Angstroms.
 *
 * NOTE:
 * The lattice side is now fixed to 200 Angstroms. If you experience
 * any problem with the lattice size, please enlarge the lattice side.
 *
 *********************************************************************/
#ifndef FIASCO_UNIQUESEQ_CONSTRAINT_
#define FIASCO_UNIQUESEQ_CONSTRAINT_

#include "typedefs.h"
#include "constraint.h"
#include <map>

class CubicLattice;
class OptionsParser;

class UniqueSeqConstraint : public Constraint {
 private:
  std::vector<CubicLattice*> grids;
  std::map <int, int> grid_indexes;
   
 public:
  UniqueSeqConstraint  (int argc, char* argv[], int& parse_pos, int weight = 4);
  ~UniqueSeqConstraint (); 
  
  bool propagate (int trailtop); 
  bool consistency ();
  bool check_cardinality (size_t& backjump);  
  bool synergic_consistency (const point& p, atom_type t = CA, int aa_idx = 0);
  void reset_synergy() {};
  void dump(bool all=true);  
};
  
#endif 
