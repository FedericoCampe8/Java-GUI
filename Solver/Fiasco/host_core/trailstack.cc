#include "trailstack.h"
#include "logic_variables.h"
#include "globals.h"
#include "atom_grid.h"
#include "constraint.h"
#include "cubic_lattice.h"
#include "constraint_store.h"
#include "variable_fragment.h"

#include <iostream>
#include <string>

using namespace std;

//#define TRAILSTACK_DBG

// Trail Constraint constructor
TrailElem::TrailElem(Constraint *c) {
  previous_trail = 0;
  constr = c;
}//-


// Trail Point constructor
TrailElem::TrailElem(VariablePoint* vp, point l, point u, int fid) {
  type = _trail_pt; // Point type
  memcpy(lb, l, sizeof(point));
  memcpy(ub, u, sizeof(point));
  //previous_trail = vp->get_last_trailed();
  point_var_ptr = vp;
  fr_id  = fid;
}//-


// Trail Fragment constructor
TrailElem::TrailElem(VariableFragment* vf, vector<domain_frag_info> domain) {
  type = _trail_fr;
  domain_info.resize(domain.size());
  for (uint fidx=0; fidx < domain.size(); fidx++) 
    domain_info[fidx] = domain[fidx];
  //previous_trail = vf->get_last_trailed();
  fragment_var_ptr = vf;
}//-


// Trail Fragment constructor
TrailElem::TrailElem(VariableFragment* vf) {
  fragment_var_ptr = vf;
}//-

// Trail Centroid constructor
TrailElem::TrailElem(point p, int vidx) {
  type = _trail_cg;
  vlist_idx = vidx;
  memcpy (lb, p, sizeof(point));
  previous_trail = 0;
}//-


// Trail Energy
TrailElem::TrailElem(real ori, real cca, real ccg, real tors, real corr):
  en_cca (cca), 
  en_ccg (ccg),
  en_ori (ori),
  en_tor (tors),
  en_cor (corr) {
  type = _trail_en;
  previous_trail = 0;
}//-


// Trail GridPoint
TrailElem::TrailElem(size_t grid_idx) {
  type = _trail_grid_pt;
  vlist_idx = grid_idx;
  previous_trail = 0;
}//-


TrailElem::TrailElem(CubicLattice* _cubic_lattice, size_t voxel_idx)
  : type (_trail_cubic_lattice_voxel), 
    cubic_lattice (_cubic_lattice),
    cubic_lattice_voxel_idx (voxel_idx), 
    previous_trail (0) {
}//-


TrailElem::TrailElem(const TrailElem& other) {
  //previous_trail = other.previous_trail;
  type = other.type;
  memcpy (lb, other.lb, sizeof(point));
  memcpy (ub, other.ub, sizeof(point));
  vlist_idx = other.vlist_idx;
  domain_explored = other.domain_explored;
  domain_info = other.domain_info;
  en_cca = other.en_cca;
  en_ccg = other.en_ccg;
  en_ori = other.en_ori;
  en_tor = other.en_tor;
  en_cor = other.en_cor;
  cubic_lattice = other.cubic_lattice;
  cubic_lattice_voxel_idx = other.cubic_lattice_voxel_idx;
  point_var_ptr = other.point_var_ptr;
  fragment_var_ptr = other.fragment_var_ptr;
  constr = other.constr;
  propagation_flag = other.propagation_flag;
  consistency_flag = other.consistency_flag;
  fr_id = other.fr_id;
  FDV_label = other.FDV_label;
}//-


TrailElem& 
TrailElem::operator= (const TrailElem& other) {
  if (this != &other) {
    //previous_trail = other.previous_trail;
    type = other.type;
    memcpy (lb, other.lb, sizeof(point));
    memcpy (ub, other.ub, sizeof(point));
    vlist_idx = other.vlist_idx;
    domain_explored = other.domain_explored;
    domain_info = other.domain_info;
    en_cca = other.en_cca;
    en_ccg = other.en_ccg;
    en_ori = other.en_ori;
    en_tor = other.en_tor;
    en_cor = other.en_cor;
    cubic_lattice = other.cubic_lattice;
    cubic_lattice_voxel_idx = other.cubic_lattice_voxel_idx;
    point_var_ptr = other.point_var_ptr;
    fragment_var_ptr = other.fragment_var_ptr;
    constr = other.constr;
    propagation_flag = other.propagation_flag;
    consistency_flag = other.consistency_flag;
    fr_id = other.fr_id;
    FDV_label = other.FDV_label;
  }
  return *this;
}//-
//---------------------------------------------------------------------//

TrailStack::TrailStack (const TrailStack& other) {
  trail_list = other.trail_list;
  continuation = other.continuation;
}//-

TrailStack& 
TrailStack::operator= (const TrailStack& other) {
  if (this != & other) {
    trail_list = other.trail_list;
    continuation = other.continuation;
  }
  return *this;
}//-

void
TrailStack::reset () {
  while (!trail_list.empty())
    trail_list.pop();

  continuation = 0;
}

void
TrailStack::backtrack (size_t trailtop) {
  string dbg = "TrailStack::backtrack() - ";
  while (trail_list.size() > trailtop) {
    TrailElem te = trail_list.top();
    switch (te.type) {
    case _trail_pt:{
      //te.point_var_ptr->set_last_trailed(te.previous_trail);
      memcpy (te.point_var_ptr->lower_bound, te.lb, sizeof(point));
      memcpy (te.point_var_ptr->upper_bound, te.ub, sizeof(point));    
      te.point_var_ptr->test_ground();
      te.fr_id = -1;  
      trail_list.pop();
      break;
    }
    case _trail_fr:{
      if (te.fragment_var_ptr->get_label() !=
	  te.fragment_var_ptr->get_next_label()) {
	// This hack is made to re-iterate over the same labeling choice 
	// for the variable being trailed, in order to read correctly rows
	// in the TABLE constraint.
	te.fragment_var_ptr->set_domain(te.domain_info);
      }
      trail_list.pop();
      break;
    }
    case _trail_cg:{
      int idx = te.vlist_idx;
      g_logicvars->var_cg_list.at(idx).set_position(te.lb);
      trail_list.pop();
      break;
    }
    case _trail_grid_pt:{
      g_grid.remove(te.vlist_idx);
      trail_list.pop();
      break;
    }
    case _trail_cubic_lattice_voxel:{
      if (te.cubic_lattice_voxel_idx)
	te.cubic_lattice->Bitset::set (te.cubic_lattice_voxel_idx, false);
      else	
	te.cubic_lattice->Bitset::reset ();     
      trail_list.pop();
      break;
    }
    case _trail_en:{
      g_logicvars->en_cca = te.en_cca;
      g_logicvars->en_ccg = te.en_ccg;
      g_logicvars->en_ori = te.en_ori;
      g_logicvars->en_cor = te.en_cor;
      g_logicvars->en_tor = te.en_tor;
      trail_list.pop();
      break;
    }
    case _trail_en_cca:{
      g_logicvars->en_cca = te.en_cca;
      trail_list.pop();
      break;
    }
    case _trail_en_ccg:{
      g_logicvars->en_ccg = te.en_ccg;
      trail_list.pop();
      break;
    }
    case _trail_en_ori:{
      g_logicvars->en_ori = te.en_ori;
      trail_list.pop();
      break;
    }
    case _trail_en_cor:{
      g_logicvars->en_cor = te.en_cor;
      trail_list.pop();
      break;
    }
    case _trail_en_tor:{
      g_logicvars->en_tor = te.en_tor;
      trail_list.pop();
      break;
    }
    case _trail_constr:{
      break;
      // for constraints,go to constraint store and remove the constraint
      g_constraintstore.remove(te.constr);
      trail_list.pop();
      break;
    }
    case _trail_constraint_consistency:{
      te.constr->set_consistent (te.consistency_flag); // reset consistency of constraint trailed
      trail_list.pop();
      break;
      }
    case _trail_constraint_propagation: {
      te.constr-> set_propagated (te.propagation_flag);
      trail_list.pop();
      break;
      }
    case _trail_synergic_constraint: {
      te.constr->reset_synergy();
      trail_list.pop();
      break; 
      }
    case _trail_constr_post_backtrack_porcess: {
      te.constr->post_backtrack_process();
      trail_list.pop();
      break; 
      }
    case _reset_vfrag: {
#ifdef TRAILSTACK_DBG
	cout << dbg <<"Setting LABEL of V_" << te.fragment_var_ptr->get_idx()
	     << " to: " << te.FDV_label << endl;
#endif
        // this is because when make_singleton_domain is forced
	// all the labeling choices are set to false!
	te.fragment_var_ptr->domain_info[te.FDV_label].explored = false;
	te.fragment_var_ptr->domain_info[te.FDV_label].frag_mate_idx = 0;
	te.fragment_var_ptr->set_labeled(te.FDV_label);
      trail_list.pop();
      break;
    }
    }
  }
}//-


void
TrailStack::trail_constraint(Constraint *c) {
  TrailElem te(c);  
  te.type = _trail_constr;
  trail_list.push(te);
}//-

void
TrailStack::trail_constraint_consistent (Constraint *c, bool val) {
  TrailElem te(c);
  te.consistency_flag = val;
  te.type = _trail_constraint_consistency;
  trail_list.push(te);
}//-

void
TrailStack::trail_constraint_propagated (Constraint *c, bool val) {
  TrailElem te(c);
  te.propagation_flag = val;
  te.type = _trail_constraint_propagation; 
  trail_list.push(te);
}//-

void
TrailStack::trail_synergic_constraint (Constraint *c) {
  TrailElem te(c);
  te.type = _trail_synergic_constraint; 
  trail_list.push(te);
}//-

void 
TrailStack::trail_post_backtrack_porcess (Constraint* c) {
  TrailElem te(c);
  te.type = _trail_constr_post_backtrack_porcess;
  trail_list.push (te);
}//-

void
TrailStack::trail_variable_point(VariablePoint* vp, point l, point u, 
				 size_t trailtop) {
  // Check if we need to trail it; only if there are no
  // other trails for same variable after the last choice point
  // if(vp->get_last_trailed() >= trailtop && trailtop > 0)
  //   return;
    
  TrailElem te(vp, l, u, vp->in_var_fragment());
  //te.point_var_ptr->set_last_trailed(trail_list.size()-1);  
  trail_list.push(te);
}//-

void 
TrailStack::trail_variable_fragment (VariableFragment* var, 
				     const vector<domain_frag_info>& domain, 
				     size_t trailtop) {
  TrailElem te(var, domain);
  trail_list.push (te);
}//-

void
TrailStack::trail_centroid (point c, int list_idx) {
  TrailElem te(c, list_idx);
  trail_list.push(te);  
}//-

void 
TrailStack::trail_energy(real ori, real cca, real ccg, real tors, real corr) {
  TrailElem te(ori, cca, ccg, tors, corr);
  trail_list.push(te);
}//-

void
TrailStack::trail_en_cca (real cca) {
  TrailElem te;
  te.en_cca = cca;
  te.type = _trail_en_cca;
  //te.previous_trail = 0;
  trail_list.push(te);
}//-

void
TrailStack::trail_en_ccg (real ccg){
  TrailElem te;
  te.en_ccg = ccg;
  te.type = _trail_en_ccg;
  //te.previous_trail = 0;
  trail_list.push(te);
}//-

void 
TrailStack::trail_en_ori (real ori){
  TrailElem te;
  te.en_ori = ori;
  te.type = _trail_en_ori;
  //te.previous_trail = 0;
  trail_list.push(te);
}//- 

void 
TrailStack::trail_en_tor (real tors){
  TrailElem te;
  te.en_tor = tors;
  te.type = _trail_en_tor;
  //te.previous_trail = 0;
  trail_list.push(te);
}//-

void 
TrailStack::trail_en_cor (real corr){
  TrailElem te;
  te.en_cor = corr;
  te.type = _trail_en_cor;
  //te.previous_trail = 0;
  trail_list.push(te);
}//-

void
TrailStack::trail_gridpoint (size_t idx){
  TrailElem te(idx);
  trail_list.push(te);
}//-

void
TrailStack::trail_unique_seq_grid (CubicLattice* _cubic_lattice, size_t voxel_idx) {
  TrailElem te(_cubic_lattice, voxel_idx);
  trail_list.push (te);
}//-

void 
TrailStack::reset_at_backtracking 
  (VariableFragment* var, size_t trailtop) {
  TrailElem te(var);
  te.type = _reset_vfrag;
  te.FDV_label = var->get_label();
  trail_list.push (te);
}//-
