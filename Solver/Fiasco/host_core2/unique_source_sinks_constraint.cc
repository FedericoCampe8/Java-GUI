#include "unique_source_sinks_constraint.h"
#include "globals.h"
#include "cubic_lattice.h"
#include "trailstack.h"
#include "logic_variables.h"
#include "variable_point.h"
#include "utilities.h"

#include <iostream>
#include <cstring>
#include <cstdio>
#include <cmath>
#include <limits>
#include <stdlib.h>

//#define USS_DBG
using namespace std;

UniqueSourceSinksConstraint::UniqueSourceSinksConstraint 
  (int argc, char* argv[], int& parse_pos, int _weight) {
  weight = 5;
  synergic = true;
  int aa_s = 0, aa_e = 0;
  for (; parse_pos < argc; parse_pos++) {
    
    if (!strcmp ("--unique-source-sinks", argv[parse_pos])) {
      point center_of_mass = {0, 0, 0};
      while (parse_pos < (argc-1) && strcmp(":", argv[parse_pos+1])) {
	aa_s = aa_e = atoi(argv[++parse_pos]);
	if (!strcmp("->", argv[++parse_pos]))
	  aa_e = atoi(argv[++parse_pos]);
	
	source = Utilities::get_bbidx_from_aaidx (aa_s, CA); 
	vpt.push_back (&g_logicvars->var_point_list[source]);	
	sink = Utilities::get_bbidx_from_aaidx (aa_e, CA); 
	vpt.push_back (&g_logicvars->var_point_list[sink]);
      }
      parse_pos++;

      real voxel_side=0;
      if (parse_pos < (argc-1) && 
	  !strcmp ("voxel-side=", argv[++parse_pos])) {
	voxel_side = atof(argv[++parse_pos]);
      }
      if (voxel_side == 0.0) 
	return;
      real lattice_side = 128;
      size_t numof_voxels = floor (pow ((lattice_side / voxel_side), 3));
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


UniqueSourceSinksConstraint::~UniqueSourceSinksConstraint () {
  delete grid;
}//-


void
UniqueSourceSinksConstraint::reset_synergy() {
  string dbg = "UniqueSourceSinksConstraint::reset_sinergy() - ";
  //cout << dbg << "Reset grid: " << source << " -> " << sink << endl;
  grid->reset();
}//-


// HACK - reset the grid at previous level (source - 1) to avoid 
// imposing the constraint starting by AA_i-1.
// Need to retrieve previous (and min among all previous) trailtop size 
bool 
UniqueSourceSinksConstraint::propagate(int trailtop) {
  string dbg = "UniqueSourceSinksConstraint::propagate - ";
  VariablePoint* cause = Constraint::caused_by_vpt[0];
  if (!cause->is_ground())
    return true; // cannot propagate
  
  if (source == cause->idx()) {
#ifdef USS_DBG
    cout << dbg << "Reset grid: " << source << " -> " << sink << endl;
#endif
    grid->reset();
  }

  if (sink == cause->idx()) {
    ; // nothing ... 
  }
#ifdef USS_DBG
  cout << dbg << "nothing\n";
#endif
  return true;
}//-


bool
UniqueSourceSinksConstraint::consistency() {
  string dbg = "UniqueSourceSinksConstraint::consistency - ";
  VariablePoint* cause = Constraint::caused_by_vpt[0];

  if (sink == cause->idx()) {
    point marker_coordinates;
    memcpy (marker_coordinates, cause->lower_bound, sizeof (point));
#ifdef USS_DBG
    cout << dbg << "sink chcking marker grid: ";
#endif
    if (!grid->test(marker_coordinates)) { // grud not set
#ifdef USS_DBG
      cout << "absent\n";
#endif
      Constraint::reset_cardinality ();
      Constraint::set_consistent (true);
      g_trailstack.trail_constraint_consistent (this, true);
      // Set Grid value (This must not be trailed!)
      grid->set (marker_coordinates);
      return true;
    }
    else {
#ifdef USS_DBG
      cout << "marker already singed!\n";
#endif
      Constraint::set_consistent (false);
      g_trailstack.trail_constraint_consistent (this, false);
      fails++;
      return false;
    }
  }
#ifdef USS_DBG
  cout << dbg << "nothing\n";
#endif
  return true;
}//-


bool 
UniqueSourceSinksConstraint::check_cardinality (size_t& backjump) {
  string dbg = "UniqueSourceSinkConstraint::check_cardinality() - ";
  
  size_t min_backjump = std::numeric_limits<size_t>::max();
  size_t curr_backjump = vpt.back()->get_last_trailed();
  
  // Enforce backjump (ONLY FOR LEFTMOST ORDERING)
  if (!Constraint::is_consistent()) {
#ifdef USS_DBG
    cout << dbg << "not consistent\n";
#endif
    min_backjump = std::min (min_backjump, curr_backjump);
  }
  else {
#ifdef USS_DBG
    cout << dbg << "consistent, incr cardinality\n";
#endif
    Constraint::incr_cardinality();
  }

  backjump = min_backjump;
#ifdef USS_DBG
  cout << dbg << "min bkjump:" << min_backjump << endl;
#endif
  if (Constraint::check_max_cardinality ()) {
#ifdef USS_DBG
    cout << dbg << "max cardinality test SUCCEEDED\n";
#endif    
    return true;
  }
  else {
#ifdef USS_DBG
    cout << dbg << "max cardinality test FAILED\n";
#endif
    backjump = std::min (min_backjump, curr_backjump);
    ; // need to handle this case!
  }
  return false;  
}//-



/*
 * NOTE:
 * The Grid got marked here even though this is not an actual propagation,
 * i.e., the variable involved in this constraint is not labelled, but 
 * this need to be taken into account for the sinergy with JM
 * note: atom type: N=0, CA=1, CB=2, O=3
 */
bool 
UniqueSourceSinksConstraint::synergic_consistency 
(const point& p, atom_type atom, int aa_idx) {
  string dbg = "UniqueSourceSinksConstraint::synergic_consistency() - ";
  if (sink == Utilities::get_bbidx_from_aaidx (aa_idx, CA)) {
    if (grid->test(p)){
#ifdef USS_DBG
      cout << dbg << "point fail \n";
#endif
      return false;
    }

    grid->set(p);
#ifdef USS_DBG
    cout << dbg << "point set \n";
#endif
  }
  return true;
}//-

void
UniqueSourceSinksConstraint::dump (bool all) {
  cout << "UniqueSourceSinks constraint id= "
       << get_id() <<"(w_" << get_weight()  << ")  ";
  if (all) { 
    cout << "CA_" << Utilities::get_aaidx_from_bbidx(vpt[0]->idx(), CA)
	 << " -> " 
	 << "CA_" << Utilities::get_aaidx_from_bbidx(vpt[1]->idx(), CA);
    if (!synergic) cout << "  NOT";
    cout << "  Synergic";
  }
  cout << endl;
}//-
