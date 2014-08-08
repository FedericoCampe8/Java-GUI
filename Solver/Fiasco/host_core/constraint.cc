#include "globals.h"
#include "constraint.h"
#include "variable_fragment.h"
#include "variable_point.h"

Constraint::Constraint() {
  id = g_constraints.size();
  consistent = false;
  propagated = false;
  synergic   = false;
  ncaused_by_vpt = ncaused_by_vfrag = 0;
  cardinality = std::make_pair(-1,-1);  
  current_cardinality = 0;
}//-

Constraint::~Constraint() {
  vpt.clear();
  vfrag.clear();
}//-

void 
Constraint::reset() {
  consistent = false;
  propagated = false;
  clear_causes();
  current_cardinality = 0;
}//-

void 
Constraint::add_dependencies() {
  // add reference to each variable involved
  for (uint i = 0; i < vpt.size(); i++) {
    vpt.at(i)->constr_dep.push_back(this);
  }
  for (uint i = 0; i < vfrag.size(); i++) {
    vfrag.at(i)->constr_dep.push_back(this);
  }
}//-

void 
Constraint::reset_causes() {
  ncaused_by_vpt = ncaused_by_vfrag = 0;
}//-

void 
Constraint::clear_causes() {
  caused_by_vpt.clear();
  caused_by_vfrag.clear();
  reset_causes();
}//-

void 
Constraint::save_cause(VariablePoint *v) {
  // check if cause is already present
  for (int i=0; i<caused_by_vpt.size(); i++) {
    if (caused_by_vpt[i]->idx() == v->idx())
      return;
  }
  // check if enough space is allocated
  if (ncaused_by_vpt >= caused_by_vpt.size())
    caused_by_vpt.push_back(NULL);
  caused_by_vpt[ncaused_by_vpt++] = v;
}//-

void 
Constraint::save_cause(VariableFragment *v) {
  // check if cause is already present
  for (int i=0; i<caused_by_vfrag.size(); i++) {
    if (caused_by_vfrag[i]->get_idx() == v->get_idx())
      return;
  }
  // check if enough space is allocated
  if (ncaused_by_vfrag >= caused_by_vfrag.size())
    caused_by_vfrag.push_back(NULL);
  caused_by_vfrag[ncaused_by_vfrag++] = v;
}//-

void 
Constraint::set_cardinality (uint min, uint max) {
  cardinality.first = min;
  cardinality.second = max;
}//-

bool 
Constraint::is_cardinality_constraint() {
  if (cardinality.first == -1 && cardinality.second == -1)
    return false;
  return true;
}//-
