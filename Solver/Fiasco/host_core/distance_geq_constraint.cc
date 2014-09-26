#include "distance_geq_constraint.h"
#include "globals.h"
#include "logic_variables.h"
#include "variable_point.h"
#include "utilities.h"

#include <iostream>
#include <cstring>
#include <cstdio>
#include <cmath>
#include <stdlib.h>


//#define VERBOSE_GEQ

DistanceGEQConstraint::DistanceGEQConstraint
  (int argc, char* argv[], int& parse_pos, int _weight) {
  weight = _weight;

  for (; parse_pos < argc; parse_pos++) {
    if (!strcmp ("--distance-geq", argv[parse_pos])) {
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
      
      if (dist == 0.0) return; // not valid constraint

      int vpt1_idx = 
	Utilities::get_bbidx_from_aaidx (atom1.first, atom1.second); 
      vpt.push_back (&g_logicvars->var_point_list[vpt1_idx]);
      int vpt2_idx = 
	Utilities::get_bbidx_from_aaidx (atom2.first, atom2.second); 
      vpt.push_back (&g_logicvars->var_point_list[vpt2_idx]);

      squared_distance = dist*dist;

      // Add dependencies
      this->add_dependencies();
      g_constraints.push_back (this);


#ifdef VERBOSE_GEQ
      std::cout << "DISTANCE_GEQ constraint (c_" << get_id() 
		<< ") created : ";
      
      std::cout << "CA_" << vpt1_idx << ", " << vpt2_idx 
		<< " dist: " << squared_distance << std::endl;

#endif
    }
  }
  
  // invalidate parser position for next constraint handling
  if (parse_pos == argc) 
    parse_pos = -1;
}//-


DistanceGEQConstraint::~DistanceGEQConstraint () {
}//-


bool 
DistanceGEQConstraint::propagate(int trailtop) {
  return true;
}//-


bool
DistanceGEQConstraint::consistency() {
  
  real dist_up   = Math::eucl_dist (vpt.front()->upper_bound , vpt.back()->lower_bound);
  real dist_down = Math::eucl_dist (vpt.front()->lower_bound , vpt.back()->upper_bound);
  
  if (std::max(dist_up, dist_down) < squared_distance) 
    return false;
  
  return true;
}//-


bool 
DistanceGEQConstraint::check_cardinality (size_t& backjump) {
  return true;
}//-


bool 
DistanceGEQConstraint::synergic_consistency 
(const point& p, atom_type t, int aa_idx) {
  return true;
}//-


void 
DistanceGEQConstraint::dump(bool all) {
  std::cout << "Distance GEQ constraint (c_" << get_id()  << ")  ";
  if (all) {
    std::cout << "Var d(PT_" << vpt.front()->idx()
	      << " PT_" << vpt.back()->idx()
	      << ")  >= " << std::sqrt(squared_distance) << std::endl;
    std::cout << std::endl;
  }
}//-
