#include "distance_leq_constraint.h"
#include "globals.h"
#include "logic_variables.h"
#include "variable_point.h"
#include "utilities.h"

#include <iostream>
#include <cstring>
#include <cstdio>
#include <cmath>
#include <stdlib.h>

using namespace std;
//#define VERBOSE_LEQ

DistanceLEQConstraint::DistanceLEQConstraint
  (int argc, char* argv[], int& parse_pos, int _weight) {
  weight = _weight;

  for (; parse_pos < argc; parse_pos++) {
    //cout << "qui1 " << parse_pos << endl;
    if (!strcmp ("--distance-leq", argv[parse_pos])) {
	//cout << "qui2\n";
      std::pair<int, atom_type> atom1 (-1, CA);
      std::pair<int, atom_type> atom2 (-1, CA);
      real dist = 0.0;

      atom1.first = atoi(argv[++parse_pos]);
      if (Utilities::get_atom_type (argv[parse_pos+1]) != X)
	atom1.second = Utilities::get_atom_type (argv[++parse_pos]);
      
      atom2.first = atoi(argv[++parse_pos]);
      if (Utilities::get_atom_type (argv[parse_pos+1]) != X)
	atom2.second = Utilities::get_atom_type (argv[++parse_pos]);
      
      dist = atof(argv[++parse_pos]);
      //cout << "dist " << dist << endl;
      if (dist == 0.0) return; // not valid constraint
      
      int vpt1_idx = 
	Utilities::get_bbidx_from_aaidx (atom1.first, atom1.second); 
      vpt.push_back (&g_logicvars->var_point_list[vpt1_idx]);
      int vpt2_idx = 
	Utilities::get_bbidx_from_aaidx (atom2.first, atom2.second); 
      vpt.push_back (&g_logicvars->var_point_list[vpt2_idx]);

      // squared_dist
      squared_distance = dist*dist;

      // Add dependencies
      this->add_dependencies();
      g_constraints.push_back (this);
    }
    
#ifdef VERBOSE_LEQ
      std::cout << "DISTANCE_LEQ constraint (c_" << get_id() 
		<< ") created : ";
      
      //std::cout << "CA_" << vpt[i]->idx() << ", ";  
      }
      std::cout << std::endl;
#endif
      //break; 
  }
  //cout << "qui------ " << parse_pos << " and " << argc << endl ;
 //	getchar();
  // invalidate parser position for next constraint handling
  if (parse_pos == argc) 
    parse_pos = -1;
}//-


DistanceLEQConstraint::~DistanceLEQConstraint () {
}//-


bool 
DistanceLEQConstraint::propagate(int trailtop) {
  return true;
}//-


bool
DistanceLEQConstraint::consistency() {
  if (vpt.front()->is_ground() && vpt.back()->is_ground()) {
    real dist_up   = Math::eucl_dist2 (vpt.front()->upper_bound , vpt.back()->lower_bound);
    real dist_down = Math::eucl_dist2 (vpt.front()->lower_bound , vpt.back()->upper_bound);
  
    if (std::min(dist_up, dist_down) > squared_distance) 
      return false;
  }
  return true;
}//-


bool 
DistanceLEQConstraint::check_cardinality (size_t& backjump) {
  return true;
}//-


bool 
DistanceLEQConstraint::synergic_consistency 
(const point& p, atom_type t, int aa_idx) {
  if (vpt.front()->idx() == aa_idx && vpt.back()->is_ground()) {
    point pp; memcpy (pp, p, sizeof(point));
    real dist = Math::eucl_dist2 (pp, vpt.front()->lower_bound);
    if (dist > squared_distance)
      return false;
  }
  else if (vpt.back()->idx () == aa_idx) {
    point pp; memcpy (pp, p, sizeof(point));
    real dist = Math::eucl_dist2 (pp, vpt.back()->lower_bound);
    if (dist > squared_distance)
      return false;
  }
  return true;
}//-


void 
DistanceLEQConstraint::dump(bool all) {
  std::cout << "Distance LEQ constraint (c_" << get_id()  << ")  ";
  if (all) {
    std::cout << "Var d(PT_" << vpt.front()->idx()
	      << " PT_" << vpt.back()->idx()
	      << ")  <= " << std::sqrt(squared_distance) << std::endl;
    std::cout << std::endl;
  }
}//-
