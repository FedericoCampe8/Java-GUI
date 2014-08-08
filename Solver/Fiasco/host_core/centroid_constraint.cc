#include "centroid_constraint.h"
#include "globals.h"
#include "atom.h"
#include "atom_grid.h"
#include "constraint_store.h"
#include "logic_variables.h"
#include "protein.h"
#include "variable_point.h"
#include "trailstack.h"
#include "statistics.h"

#include <string>

using namespace std;

CentroidConstraint::CentroidConstraint 
  (VariablePoint *p1_ptr, 
   VariablePoint *p2_ptr, 
   VariablePoint *p3_ptr) {
  weight = 4;
  vpt.push_back(p1_ptr);
  vpt.push_back(p2_ptr);
  vpt.push_back(p3_ptr);

  this->add_dependencies();
  g_constraints.push_back(this);
#ifdef VERBOSE
     std::cout << "CENTROID constraint (c_" << get_id() 
	       << ") created : ";
     for (int i = 0; i < vpt.size(); i++) {
       std::cout << "Var-Pt_" << vpt[i]->idx() << ", ";  
     }
     std::cout << std::endl;
#endif
}//-


// Effects of propagating the centroid constraint 
//  1. compute the value of the centorid 
//  2. add the centroid to the atom_grid 
bool 
CentroidConstraint::propagate (int trailtop) {
  string dbg = "CentroidConstraint::propagate() - ";

  VariablePoint *p1 = vpt.at(0);  //ca1
  VariablePoint *p2 = vpt.at(1);  //ca2
  VariablePoint *p3 = vpt.at(2);  //ca3
    
  if (!p1->is_ground() || !p2->is_ground() || !p3->is_ground()) {
    return true; // the constraint has not failed but cannot propagate
  }
  int cg_idx = Utilities::get_aaidx_from_bbidx(p2->idx(), CA);
  int cg_len_idx = g_target.get_bblen() + cg_idx;
  g_logicvars->var_cg_list.at(cg_idx-1).compute_cg (g_target.sequence[cg_idx],
						   p1->lower_bound,
						   p2->lower_bound,
						   p3->lower_bound);
  // Test Centroid value consistency on the grid
  if (!g_grid.query(g_logicvars->var_cg_list.at(cg_idx-1))) {
    // invalidate centroid computed
    memset (g_logicvars->var_cg_list.at(cg_idx-1).position, 0, sizeof(point));
    g_statistics->incr_propagation_failures(__c_centroid);
    return false;
  }

  g_grid.add (g_logicvars->var_cg_list.at(cg_idx-1).position, 
	      CG, cg_len_idx);
  g_statistics->incr_propagation_successes(__c_centroid);
  // @todo Compute contact CG component and Orientation compoments
  return true;
}//-

bool 
CentroidConstraint::consistency() {
  return true;
}//-

void 
CentroidConstraint::dump(bool all) {
  std::cout << "Centroid constraint (w_" << get_weight()  << ")  ";
  if (all) {
    for (uint i=0; i<vpt.size(); i++)
      cout << " vPt_" << vpt.at(i)->idx();
  }
  cout << endl;
}//-
