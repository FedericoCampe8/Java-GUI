#include "alldistant_constraint.h"
#include "globals.h"
#include "utilities.h"

#include "atom_grid.h"
#include "constraint_store.h"
#include "logic_variables.h"
#include "trailstack.h"
#include "statistics.h"
#include "variable_point.h"

#include <cmath>
#include <vector>

using namespace std;
//#define ALLDIST_DBG

AlldistantConstraint::AlldistantConstraint() {
  weight = 3;
  synergic = true;

  vpt.reserve(g_logicvars->var_point_list.size());  
  for(uint i=0; i < g_logicvars->var_point_list.size(); i++)
    vpt.push_back(&g_logicvars->var_point_list.at(i));

  // Add dependencies
  this->add_dependencies();
  g_constraints.push_back(this);

#ifdef VERBOSE
     std::cout << "ALLDISTANT constraint (c_" << get_id() 
	       << ") created\n";
#endif
}//-

bool
AlldistantConstraint::propagate(int trailtop) {
  return true;
}//-


bool
AlldistantConstraint::consistency () {
  string dbg = "ConstraintAlldist::consistency() - ";

  // Test atom in the grid for consistency check
  for (uint i=0; i < ncaused_by_vpt; i++) {
    VariablePoint* vp = caused_by_vpt.at(i);
    if (vp->is_ground()) // && (atype == CA || atype == CG))
      if (!g_grid.query (vp)) { 
          consistent = 0;
#ifdef STATISTICS
	  g_statistics->incr_propagation_failures(__c_alldist);
#endif

#ifdef ALLDIST_DBG
	  cout << dbg << "atom in grid not conistent - FALSE\n";
	  vp->dump();
#endif
          return false;
      }
  }
  consistent = 1;

  g_trailstack.trail_constraint_consistent (this);

  // Add atoms to the grid
  for (uint i=0; i < ncaused_by_vpt; i++) {
    VariablePoint* vp = caused_by_vpt.at(i);
    atom_type atype =
      Utilities::get_atom_type (vp->idx());
    int aa_idx = 
      Utilities::get_aaidx_from_bbidx(vp->idx(), atype);
      if (vp->is_ground()){// && (atype == CA || atype == CG)) {
	g_grid.add (vp->lower_bound, atype, aa_idx);
      }
  }
#ifdef STATISTICS
  g_statistics->incr_propagation_successes(__c_alldist);
#endif

  return true;
}//-


bool 
AlldistantConstraint::synergic_consistency 
  (const point& p, atom_type t, int aa_idx) {
  if (!g_grid.query (p, t, aa_idx)) { 
    string dbg = "AlldistantConstraint::synergic_consistency() - ";
    g_statistics->incr_propagation_failures(__c_alldist); 
#ifdef ALLDIST_DBG
    cout << dbg << "not synergy consistent - FALSE\n";
#endif
    return false;
  }
  return true;  
}//-


void 
AlldistantConstraint::dump(bool all) {
  cout << "AllDistant constraint (w_ " << get_weight() <<")  ";
  if (all) {
    if (!synergic) cout << "not ";
    cout << "Synergic" << endl;
  }
}//-

