#include "uniform_constraint.h"
#include "globals.h"
#include "cubic_lattice.h"
#include "trailstack.h"
#include "logic_variables.h"
#include "variable_point.h"

#include <iostream>
#include <vector>
#include <cstring>
#include <cstdio>
#include <cmath>
#include <limits>
#include <stdlib.h>

using namespace std;
using namespace Utilities;

UniformConstraint::UniformConstraint 
  (int argc, char* argv[], int& parse_pos, 
   const R_MAT& rot, const vec3& sh, int _weight) {
  weight = _weight;
  synergic = true;

  for (; parse_pos < argc; parse_pos++) {

    if (!strcmp ("--uniform", argv[parse_pos])) {
      point center_of_mass = {0, 0, 0};
    
      while (parse_pos < (argc-1) && 
	     strcmp(":", argv[++parse_pos])) {
	int vpt_idx = 
	  Utilities::get_bbidx_from_aaidx (atoi(argv[parse_pos]), CA); 
	vpt.push_back (&g_logicvars->var_point_list[vpt_idx]);
      }

      real voxel_side=0;
      if (parse_pos < (argc-1) && 
	  !strcmp ("voxel-side=", argv[++parse_pos])) {
	voxel_side = atof(argv[++parse_pos]);
      }
      real lattice_side = 128;
      size_t numof_voxels = floor(pow ((lattice_side / voxel_side), 3));
      
      // lattice_side = atof (argv[++parse_pos]);
      // numof_voxels = atoi (argv[++parse_pos]);

      if (parse_pos < (argc-1) && 
	  !strcmp ("center=", argv[++parse_pos])) {
	center_of_mass[0] = atof (argv[++parse_pos]);
	center_of_mass[1] = atof (argv[++parse_pos]);
	center_of_mass[2] = atof (argv[++parse_pos]);
	if (rot != NULL && sh != NULL) {
	  Math::translate (center_of_mass, sh);
	  Math::rotate_inverse (center_of_mass, rot);
	}
      }
      
      // Cardinality constraint 1{C}1
      Constraint::set_cardinality (1, 1);
      grid = new CubicLattice (lattice_side, numof_voxels, center_of_mass);
      
      // Add dependencies
      this->add_dependencies();
      g_constraints.push_back (this);

#ifdef VERBOSE
      dump();
#endif
      break; 
    }
  }
  
  // invalidate parser position for next constraint handling
  if (parse_pos == argc) 
    parse_pos = -1;
}//-


// UniformConstraint::UniformConstraint 
//   (const OptionsParser &opts, int _weight) {
//   weight = _weight;
//   real x = opts.get<real> ("center-x");
//   real y = opts.get<real> ("center-y");
//   real z = opts.get<real> ("center-z");
//   point center_of_mass = {x, y, z};
//   grid = new CubicLattice (opts.get<size_t> ("grid_side"),
// 			   opts.get<size_t> ("voxels"),
// 			   center_of_mass); // can be NULL?
//   std::vector<int> markers = opts.get_list<int> ("markers");
//   weight = 4;
//   for (int i=0; i<markers.size(); i++)
//     vpt.push_back (&g_logicvars.var_point_list[markers[i]]);
// }//-


UniformConstraint::~UniformConstraint () {
  delete grid;
}//-


/*
 * The uniform constraint consistency is verified for every variable point
 * involved in the constraint.
 * Any time the constraint is woken up a variable assignment, checking 
 * if any of the point is consistent with the constraint, ensure that no
 * two solutions will share the same points in the grid.
 */
bool
UniformConstraint::consistency() {
  string dbg = "UniformConstraint::debug - ";
  
  for (int var_pt = 0; var_pt < vpt.size(); var_pt++) {
    point marker_coordinates;
    memcpy (marker_coordinates, vpt[var_pt]->lower_bound, sizeof (point));
    
    // Check if current choice was already set in corresponding voxel
    if (grid->test (marker_coordinates)) {
      return false;
    }
  }
  return true;
  
}//-


bool 
UniformConstraint::check_cardinality (size_t& backjump) {
  string dbg = "UniformConstraint::is_solution_consistent() - ";
  
  // For this constraint we check cardinality in an implict manner:
  // analysing the content of the grid
  Constraint::reset_cardinality ();
  size_t min_backjump = std::numeric_limits<size_t>::max();
  
  for (int var_pt = 0; var_pt < vpt.size(); var_pt++) {
    point marker_coordinates;
    memcpy (marker_coordinates, vpt[var_pt]->lower_bound, sizeof (point));
 
    if (grid->test (marker_coordinates)) {
      size_t curr_backjump = vpt[var_pt]->get_last_trailed();
      min_backjump = std::min (min_backjump, curr_backjump);
    }
    else {
      size_t voxel_idx = grid->set (marker_coordinates); 
      if (voxel_idx == std::numeric_limits<size_t>::max()) {
	std::cout << "Point variable " << vpt[var_pt]->idx() 
		  << "was out of the grid specified\n";
	//sleep(2);
      }
    }
  }
  
  backjump = min_backjump;
  
  if (min_backjump == std::numeric_limits<size_t>::max()) {
    Constraint::incr_cardinality();
    if (Constraint::check_max_cardinality ())
      return true;
    else 
      ; // need to handle this case!
  }

  return false;  
}//-


bool 
UniformConstraint::propagate(int trailtop) {
  return true;
}//-


bool 
UniformConstraint::synergic_consistency 
(const point& p, atom_type t, int aa_idx) {
  if (grid->test (p)) 
    return false;
  return true;
}//-


void 
UniformConstraint::dump(bool all) {
  std::cout << "Constraint UNIFORM  (w_" << get_weight() << ")\n";
}//-
