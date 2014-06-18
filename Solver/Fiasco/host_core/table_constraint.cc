#include "table_constraint.h"
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
#include <limits>
#include <algorithm>

#include <stdio.h>
#include <stdlib.h>

using namespace std;
//#define TABLE_DBG

TableConstraint::TableConstraint 
(uint v_start, uint v_end, uint nrow) {
  vector<uint> variables;
  for (uint i = v_start; i <= v_end; i++) 
    variables.push_back(i);
  
  weight = 2;
  synergic = false;
  _nrows = nrow;
  _ncols = variables.size();

  vpt.reserve(variables.size());
  for (uint i=0; i < variables.size(); i++) {
    vfrag.push_back (&g_logicvars->var_fragment_list[variables[i]]);
    int v_idx = g_logicvars->var_fragment_list[variables[i]].get_idx();
    variables_indexes[v_idx] = i;
  }
  
  table.resize (_nrows);
  for (uint i=0; i < _nrows; i++)
    table[i].resize (_ncols, NOT_INITIALIZED); 
  current_row = 0;

  // Add dependencies
  this->add_dependencies();
  g_constraints.push_back(this);

#ifdef VERBOSE
  dump(true);
#endif
}//-

void
TableConstraint::reset() {
  current_row = 0;
}//-

void
TableConstraint::reset_row(uint row) {
#ifndef JM_PREPROCESS
  for (int c=0; c<table[row].size(); c++)
    table[row][c] = NOT_INITIALIZED;
#endif
}

void
TableConstraint::set (uint row, uint col, int value) {
  table[row][col] = value;
}//-

void 
TableConstraint::post_backtrack_process () {
  TableConstraint::reset_row (current_row++); 
}

// The idea here is to restrict the domain of the target variable 
// to a singleton, hence propagate the specific constraint over it 
// if the labeling choice made was the right guess (can make 
// a mistake at most once) 
bool
TableConstraint::propagate (int trailtop) {
  string dbg = "TableConstraint::propagate - ";
  VariableFragment* cause = Constraint::caused_by_vfrag[0];
  //int current_col = variables_indexes[cause->get_idx()];
  size_t head = Constraint::vfrag[0]->get_idx();
  //size_t tail = Constraint::vfrag[vfrag.size()-1]->get_idx();

#ifdef TABLE_DBG
  cout << dbg << "VarFrag_" << cause->get_idx()
       << "("<<cause->get_label()<<")\tTable Row: " 
       << current_row << "/" << _nrows <<" col: " << current_col 
       << "/" <<_ncols << endl; 
#endif

  // Base Case -- exploration finished
  if (current_row == _nrows || 
      table[current_row][0] == NOT_INITIALIZED) {
#ifdef TABLE_DBG
    cout << dbg << "Base -> ret FALSE" << endl; 
#endif
    return false;
  }
  // Variable in the body of the table  
  if (cause->get_idx() != head) {
    return true;
  }
  // Head variable with wrong labeling
  if (cause->get_label() != table[current_row][0]) {
#ifdef TABLE_DBG
    cout << "Head with wrong label\n";
#endif
    table_head++;
    return false;
  }
  g_trailstack.reset_at_backtracking (cause, trailtop);
  g_trailstack.trail_post_backtrack_porcess (this);
  body_head++;
  // HEAD 
#ifdef TABLE_DBG
  dump(true);
  cout << dbg << "setting all singletons: "  << " -[" << current_row << "]: ";
#endif
  for (int vf=0; vf < Constraint::vfrag.size(); vf++) {
#ifdef TABLE_DBG
    cout << table[current_row][vf] << ", ";
#endif
    if (Constraint::vfrag[vf]->get_idx() != head)
      Constraint::vfrag[vf]->set_domain_singleton (table[current_row][vf]);
    Constraint::vfrag[vf]->set_labeled (table[current_row][vf]);
  }
  
#ifdef TABLE_DBG
    getchar();
#endif
    return true;
}//-


bool
TableConstraint::consistency () {
  return true;
}//-


bool 
TableConstraint::synergic_consistency 
  (const point& p, atom_type t, int aa_idx) {
  return true;  
}//-


void 
TableConstraint::dump (bool all) {
  cout << "TABLE constraint (w_ " << get_weight() <<") ";
  if (all) {
    for (int i = 0; i < vfrag.size(); i++) {
      cout << "VarF_" << vfrag[i]->get_idx() << ", ";  
    }
    cout << "\t";
    if (!synergic) cout << "not ";
    cout << "Synergic" << endl;
    
#ifdef TABLE_DBG
    for (int r=0; r<std::min(table.size(), (size_t)20); r++) {
      for (int c=0; c<table[r].size(); c++) { 
    	cout << table[r][c] << " ";
      }
      if (current_row == r)
    	cout << "  <-- ";
      cout << endl;
    }
#endif
  }
  cout << endl;
}//-


void 
TableConstraint::radix_sort() {
  for (int curr_col=_ncols-1; curr_col>=0; curr_col--) 
  {
    int top_r = 0;
    while (top_r < _nrows) {
      // find minimum 
      int min_c = std::numeric_limits<int>::max();
      int min_r = -1;
      for (int r = top_r; r < _nrows; r++) {
	if (table[r][curr_col] < min_c) {
	  min_c = table[r][curr_col];
	  min_r = r;
	}
      }
      if (min_c == NOT_INITIALIZED) break;
      // Swap with top row
      for (int c = 0; c < _ncols; c++) {
	std::swap (table[top_r][c], table[min_r][c]); 
      }
      top_r++;
    }
  }
}//-
