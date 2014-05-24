#include "uniqueseq_constraint.h"
#include "globals.h"
#include "cubic_lattice.h"
#include "trailstack.h"
#include "logic_variables.h"
#include "variable_point.h"

#include <iostream>
#include <cstring>
#include <cstdio>
#include <cmath>
#include <limits>
#include <stdlib.h>

using namespace std;
using namespace Utilities;

//#define UNIQUESEQ_DBG

UniqueSeqConstraint::UniqueSeqConstraint 
  (int argc, char* argv[], int& parse_pos, int _weight) {
  weight = _weight;

  int aa_s = 0, aa_e = 0;
  for (; parse_pos < argc; parse_pos++) {

    if (!strcmp ("--unique-seq", argv[parse_pos])) {
      point center_of_mass = {0, 0, 0};
    
      while (parse_pos < (argc-1) && 
	     strcmp(":", argv[parse_pos+1])) {

	aa_s = aa_e = atoi(argv[++parse_pos]);

	if (!strcmp("->", argv[++parse_pos]))
	  aa_e = atoi(argv[++parse_pos]);

	for (int ica = aa_s; ica <= aa_e; ica++) { 
	  int vpt_idx = Utilities::get_bbidx_from_aaidx (ica, CA); 
	  vpt.push_back (&g_logicvars->var_point_list[vpt_idx]);
	}
      }
      
      real voxel_side=0;
      if (++parse_pos < (argc-1) && 
	  !strcmp ("voxel-side=", argv[++parse_pos])) {
	voxel_side = atof(argv[++parse_pos]);
      }
      if (voxel_side == 0.0) 
	return;
      real lattice_side = 200;
      size_t numof_voxels = floor(pow ((lattice_side / voxel_side), 3));
      
      // Cardinality constraint 1{C}1
      Constraint::set_cardinality (1, 1);

      // Create a grid for each amino acid
      grids.resize(vpt.size());
      for (int i = 0; i < vpt.size(); i++) {
	grids[i] = new CubicLattice (lattice_side, numof_voxels, center_of_mass);
	// map grid with variables
	grid_indexes[vpt[i]->idx()] = i;
      }

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


UniqueSeqConstraint::~UniqueSeqConstraint () {
  for (int i = 0; i < grids.size(); i++) 
    delete grids[i];
}//-


/*
 * Propagation of the UniqueSeq constraint works as follows:
 * The grid corresponding to the variable waking up this constraint
 * is checked to verify its consistency (We need at least one 
 * marker not to fall in the grid, hence if we found one, the 
 * constraint will be set as consistent).
 * The grid is hence marked as visited, the next grid is trailed 
 * so as to restore it once current variable is selected again.
 */
bool 
UniqueSeqConstraint::propagate(int trailtop) {
  string dbg = "UniqueSeqConstraint::propagate - ";

  VariablePoint* cause = Constraint::caused_by_vpt[0];
  if (!cause->is_ground())
    return true; // cannot propagate
  int g_idx = grid_indexes[cause->idx()];

  // Check constraint consistency.
  point marker_coordinates;
#ifdef UNIQUESEQ_DBG
  std::cout << "Testing grid_" << g_idx << std::endl;
#endif
  memcpy (marker_coordinates, cause->lower_bound, sizeof (point));
  if (!grids[g_idx]->test (marker_coordinates)) { // grid not set
#ifdef UNIQUESEQ_DBG
    std::cout << dbg << " marker (" 
    	      << marker_coordinates[0] << ","
    	      << marker_coordinates[1] << ","
    	      << marker_coordinates[2] << ") V_"
    	      << cause->idx() << " not visited yet\n";
#endif
    Constraint::reset_cardinality ();
    Constraint::set_consistent (true);
    g_trailstack.trail_constraint_consistent (this, true);
  }
  else {
#ifdef UNIQUESEQ_DBG
     std::cout << dbg << " marker (" 
    	      << marker_coordinates[0] << ","
    	      << marker_coordinates[1] << ","
    	      << marker_coordinates[2] << ") V_"
    	      << cause->idx() << " already visited\n";
#endif
   Constraint::set_consistent (false);
    g_trailstack.trail_constraint_consistent (this, false);
    return false;
  }

  // Set Grid value
  // This must not be trailed!
  size_t voxel_idx = grids[g_idx]->set (marker_coordinates); 
  if (voxel_idx == std::numeric_limits<size_t>::max()) {
    std::cout << "Point variable " << cause->idx() 
	      << "was out of the grid specified\n";
    //sleep(2);
  }

  // Save pointer of grid associated to next var point, so it 
  // can be cleared while backtracking.
  if (g_idx + 1 < grids.size()){
#ifdef UNIQUESEQ_DBG
    std::cout << dbg << " Trailing: G_" << g_idx+1;
#endif
    g_trailstack.trail_unique_seq_grid (grids[g_idx + 1]);
  }
#ifdef VERBOSE_UNIQUESEQ
  getchar();
#endif
  return true;
}//-


bool
UniqueSeqConstraint::consistency() {
  return true;
}//-


// Note that the variables here are guaranteed to be ground, as this
// procedure is enforced at the leaf levels.
// NOTE: 
//  This check holds only if the variable selection is LEFTMOST within
// the variables involved in this constraint 
bool 
UniqueSeqConstraint::check_cardinality (size_t& backjump) {
  string dbg = "UniqueSeqConstraint::check_cardinality() - ";
  
  size_t min_backjump = std::numeric_limits<size_t>::max();
  size_t curr_backjump = vpt.back()->get_last_trailed();
  
  // Enforce backjump (ONLY FOR LEFTMOST ORDERING)
  if (!Constraint::is_consistent()) {
#ifdef UNIQUESEQ_DBG
    std::cout << dbg << "Constraint read NOT to be consistent\n";
#endif
    min_backjump = std::min (min_backjump, curr_backjump);
  }
  else {
#ifdef UNIQUESEQ_DBG
    std::cout << dbg << "Constraint read consistent\n"; 
#endif
    Constraint::incr_cardinality();
  }

  backjump = min_backjump;
  
  if (Constraint::check_max_cardinality ()) {
#ifdef UNIQUESEQ_DBG
    std::cout << dbg << "max cardinality check succeeded\n";
    getchar();
#endif
    return true;
  }
  else { 
    backjump = std::min (min_backjump, curr_backjump);
#ifdef UNIQUESEQ_DBG
    std::cout << dbg << "max cardinality check failed\n";
    getchar();
 #endif
    ; // need to handle this case!
  }
  return false;  
}//-


// TODO!!
bool 
UniqueSeqConstraint::synergic_consistency 
(const point& p, atom_type t, int aa_idx) {
  if (Constraint::caused_by_vpt.empty())
    return true;
  
  int cause = Constraint::caused_by_vpt[0]->idx();
  int g_idx = grid_indexes[cause];
  
  if (grids[g_idx]->test (p)) 
    return false;
  return true;
}//-


void 
UniqueSeqConstraint::dump(bool all) {
  std::cout << "UNIQUE SEQ constraint (w_" << get_weight()  << ")  ";
  if (all) {
    for (int i = 0; i < vpt.size(); i++) {
      std::cout << "CA_" << vpt[i]->idx() << ", ";  
    }
  }
  std::cout << std::endl;
}//-
