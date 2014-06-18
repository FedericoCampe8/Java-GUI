/*********************************************************************
 * object: UniformConstraint
 * The Uniform constraint is a spatial constraint which guarantees that
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
 * --uniform aa_1 .. aa_n : voxel-side= K [center= X Y Z ]
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
 *
 * TODO:
 * Include parser for syntax like:
 * --uniform-constraint(markers=(, , , ), grid_side=XX, voxels=XX [, center=(x,y,z)])
 * where:
 *  - markers. is a list of atoms involved in the constraint,
 *  - grid_side, is the side of the grid in Angstroms
 *  - voxels, is the total number of voxels in the lattice
 *  - center, is the center of mass of the lattice
 *********************************************************************/
#ifndef FIASCO_UNIFORM_CONSTRAINT_
#define FIASCO_UNIFORM_CONSTRAINT_

#include "typedefs.h"
#include "constraint.h"

class CubicLattice;
class OptionsParser;

class UniformConstraint : public Constraint {
 private:
  CubicLattice* grid;
  //std::vector<int> markers;
  
 public:
  UniformConstraint (const OptionsParser &opts, int weight = 4); // TODO
  UniformConstraint  (int argc, char* argv[], int& parse_pos, 
		      const R_MAT& rot, const vec3& sh, int weight = 4); // TEMP SOLUTION
  ~UniformConstraint (); 
  
  bool propagate (int trailtop); 
  bool consistency ();
  bool check_cardinality (size_t& backjump);  
  bool synergic_consistency (const point& p, atom_type t = CA, int aa_idx = 0);
  void reset_synergy() {};
  void dump(bool all=true);  
};
  
#endif 
