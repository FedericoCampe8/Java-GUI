#include "end_anchor_different_constraint.h"
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

#define VERBOSE_EADIFF
//#define EADIFF_DBG
using namespace std;

EndAnchorDifferentConstraint::EndAnchorDifferentConstraint 
  (int argc, char* argv[], int& parse_pos, int _weight) {
  weight = _weight;
  synergic = true;
  int aa = 0;
  for (; parse_pos < argc; parse_pos++) {
    
    if (!strcmp ("--end-anchor-diff", argv[parse_pos])) {
      point center_of_mass = {0, 0, 0};
      aa = atoi(argv[++parse_pos]);
      
      int vpt_idx = Utilities::get_bbidx_from_aaidx (aa, N); 
      vpt.push_back (&g_logicvars->var_point_list[vpt_idx]);   // N
      vpt.push_back (&g_logicvars->var_point_list[vpt_idx+1]); // CA
      vpt.push_back (&g_logicvars->var_point_list[vpt_idx+2]); // CB
      vpt.push_back (&g_logicvars->var_point_list[vpt_idx+3]); // O
      
      real voxel_side=0;
      if (parse_pos < (argc-1) && 
	  !strcmp ("voxel-side=", argv[++parse_pos])) {
	voxel_side = atof(argv[++parse_pos]);
      }
      if (voxel_side == 0.0) 
	return;

      real lattice_side = 200;
      size_t numof_voxels = floor (pow ((lattice_side / voxel_side), 3));
      
      // Cardinality constraint 1{C}1
      // Constraint::set_cardinality (1, 1);

      grid = new CubicLattice (lattice_side, numof_voxels, center_of_mass);

      // Add dependencies
      this->add_dependencies();
      g_constraints.push_back (this);
      
#ifdef VERBOSE_EADIFF
      dump();
#endif
      break; 
    }
  }
    
  // invalidate parser position for next constraint handling
  if (parse_pos == argc) 
    parse_pos = -1;
}//-


EndAnchorDifferentConstraint::~EndAnchorDifferentConstraint () {
  delete grid;
}//-


void
EndAnchorDifferentConstraint::reset_synergy() {
  string dbg = "EndAnchorDifferentConstraint::reset_sinergy() - ";
  grid->reset();
}//-

bool 
EndAnchorDifferentConstraint::propagate(int trailtop) {
  return true;
  // should i do a trick here to reset the grid once backtracking?
  
}//-


bool
EndAnchorDifferentConstraint::consistency() {
  return true;
}//-


bool 
EndAnchorDifferentConstraint::check_cardinality (size_t& backjump) {
  return true;  
}//-



/*
 * NOTE:
 * The Grid got marked here even though this is not an actual propagation,
 * i.e., the variable involved in this constraint is not labelled, but 
 * this need to be taken into account for the sinergy with JM
 * note: atom type: N=0, CA=1, CB=2, O=3
 */
bool 
EndAnchorDifferentConstraint::synergic_consistency 
(const point& p, atom_type atom, int aa_idx) {
  string dbg = "EndAnchorDifferentConstraint::synergic_consistency() - ";

  // consistency check only relevant for the aa of this constraint
  if (Utilities::get_bbidx_from_aaidx(aa_idx, atom) != vpt[atom]->idx())
    return true; 

  if (grid->test (p)) {
#ifdef EADIFF_DBG
    Utilities::dump (p, dbg + "test on grid FAILED on point: ");
#endif
    return false;
  }
  else 
    grid->set (p);
#ifdef EADIFF_DBG
  Utilities::dump (p, dbg + "test on grid SUCCEEDED: ");
#endif
  return true;
}//-


void 
EndAnchorDifferentConstraint::dump (bool all) {
  std::cout << "EndAnchorDiffrent constraint (w_" << get_weight()  << ")  ";
  if (all) { 
    std::cout << "CA_" << Utilities::get_aaidx_from_bbidx(vpt[1]->idx(), CA);  
    if (!synergic) std::cout << "  NOT";
    std::cout << "  Synergic";
  }
  std::cout << std::endl;
}//-
