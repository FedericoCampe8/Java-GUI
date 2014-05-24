/*********************************************************************
 * Trail Stack implementation
 *********************************************************************/
#ifndef FIASCO_TRAILSTACK__
#define FIASCO_TRAILSTACK__

#include "typedefs.h"
#include "variable_fragment.h"

#include <vector>
#include <stack>

class VariablePoint;
class VariableFragment;
class Constraint;
class CubicLattice;

struct domain_frag_info;

enum trail_type {_trail_pt, 
		 _trail_fr, 
		 _trail_cg,
		 _trail_en, 
		 _trail_en_cca, 
		 _trail_en_ccg, 
		 _trail_en_ori, 
		 _trail_en_tor, 
		 _trail_en_cor,
		 _trail_constr,
		 _trail_constraint_propagation,
		 _trail_constraint_consistency,
		 _trail_synergic_constraint,
		 _trail_constr_post_backtrack_porcess,
		 _trail_grid_pt,
		 _trail_cubic_lattice_voxel,
		 _reset_vfrag};

class TrailElem {
 public:
  trail_type type;
  point lb, ub;   // previous stored point values
  int vlist_idx;  // position of the element in the variable list 
                  // {applies to point, frag, pair and centroids}
  std::vector <bool> domain_explored; // previous domain explored ///// DEPR
  std::vector < domain_frag_info > domain_info;

  real en_cca, en_ccg, en_ori, en_tor, en_cor;
  
  // used to manage the Uniform backtrack constraint
  CubicLattice* cubic_lattice;
  size_t cubic_lattice_voxel_idx;

  VariablePoint* point_var_ptr;
  VariableFragment* fragment_var_ptr;
  Constraint* constr;
    
  int fr_id; //id fragment for variable point 
  int FDV_label; // meaningful label to store
  size_t previous_trail;
 
  // Constraint Trails
  bool propagation_flag;
  bool consistency_flag;
  //-

  TrailElem(){};
  // constructor for points
  TrailElem(VariablePoint* v, point l, point u, int fid);
  // constructor for fragments
  TrailElem(VariableFragment* v, std::vector<domain_frag_info> domain); 
  TrailElem(VariableFragment* v); 
  // constructor for centroids
  TrailElem(point p, int vidx);
  // constructor for energy
  TrailElem(real ori, real cca, real ccg, real tors, real corr);
  // constructor for grid point
  TrailElem(size_t grid_idx);
  // constructor for cubic lattice grid
  TrailElem (CubicLattice* _cubic_lattice, size_t voxel_idx = 0);
  // constructor for constraint
  TrailElem (Constraint *c);

  // Rules of 3
  ~TrailElem() {};
  TrailElem(const TrailElem& other);
  TrailElem& operator= (const TrailElem& other);
};


class TrailStack {
 private:
  std::stack<TrailElem> trail_list;
  size_t continuation; 

 public:
  TrailStack(){};
  ~TrailStack(){}
  TrailStack(const TrailStack& other);
  TrailStack& operator=(const TrailStack& other);

  bool is_empty();
  size_t size() {return trail_list.size();}  
  void backtrack(size_t top);
  size_t get_continuation(){return continuation;}
  void set_continuation(){continuation = trail_list.size();}
  void reset_continuation(){continuation = 0;}
  void reset ();

  void trail_constraint (Constraint *c);
  void trail_constraint_propagated (Constraint *c, bool val=false);
  void trail_constraint_consistent (Constraint *c, bool val=false);
  void trail_synergic_constraint (Constraint *c);
  void trail_post_backtrack_porcess (Constraint* c);
  
  void trail_variable_point(VariablePoint* vp, point l, point u, 
			    size_t trailtop);
  void trail_variable_fragment(VariableFragment* var, 
			       const std::vector < domain_frag_info >& domain,
			       size_t trailtop);
  void reset_at_backtracking (VariableFragment* var, size_t trailtop);

  void trail_centroid (point c, int list_idx);
  void trail_energy(real ori, real cca, real ccg, real tors, real corr);
  void trail_en_cca (real cca);
  void trail_en_ccg (real ccg);
  void trail_en_ori (real ori);
  void trail_en_tor (real tor);
  void trail_en_cor (real cor);
  void trail_gridpoint (size_t id);
  void trail_unique_seq_grid (CubicLattice* _cubic_lattice, size_t voxel_idx = 0);
};

#endif
