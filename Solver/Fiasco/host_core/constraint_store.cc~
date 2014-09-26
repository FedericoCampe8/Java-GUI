#include "constraint_store.h"
#include "globals.h"
#include "constraint.h"
#include "logic_variables.h"
#include "trailstack.h"

#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <cassert>
#include <limits>

#include <stdio.h> // used by getchar (DEBUG)

using namespace std;
//#define CSTORE_DBG

/*
 * Syntactic Sugar: Comparison Function
 * sort in an Incresing order
 */
bool constraint_cmp (const Constraint* ci, const Constraint* cj) {
  return ( ci->get_weight() > cj->get_weight() || 
	   ci->get_id() > cj->get_id()); 
}//-

bool
ConstraintStore::propagate (size_t trailtop) {
  return arc_consistency3(trailtop);
}//-

Constraint* 
ConstraintStore::fetch(){
  Constraint *ret = NULL;
  if (store.size()>0) {
    ret = store.back();
    store.pop_back();
  }
  return ret;
}//-

void 
ConstraintStore::add(Constraint *c) {
  store.push_back(c);
  c->clear_causes();
  c->set_id(store.size()-1);  // Set the constraint id;
  
  // Add reference to each variable involved in a constraint
  for (uint i = 0; i < c->vpt.size(); i++)
    c->vpt.at(i)->constr_dep.push_back(c);
  for (uint i = 0; i < c->vfrag.size(); i++)
    c->vfrag.at(i)->constr_dep.push_back(c);
}//-


/*
 * @note: dynamically removes a constraint c from each variable 
 * in which it is involved in
 */
void 
ConstraintStore::remove (Constraint* c){
  vector<Constraint*>::iterator it; 
  
  // Remove the constraint c from the store
  it = find (store.begin(), store.end(), c);
  assert(it != store.end());
  store.erase (it);

  // Remove reference from Point variables
  for (uint i = 0; i < c->vpt.size(); i++) {
    it = find (c->vpt.at(i)->constr_dep.begin(),
               c->vpt.at(i)->constr_dep.end(), 
               c);
    if (it != c->vpt.at(i)->constr_dep.end())
      c->vpt.at(i)->constr_dep.erase(it);
  }

  // Remove reference from Fragment variables
  for (uint i = 0; i < c->vfrag.size(); i++) {
    it = find (c->vfrag.at(i)->constr_dep.begin(),
               c->vfrag.at(i)->constr_dep.end(), 
               c);
    if (it != c->vfrag.at(i)->constr_dep.end())
      c->vfrag.at(i)->constr_dep.erase(it);
  }
  
  // Finally remove the constraint
  delete c;
}//-

void 
ConstraintStore::reset () {
  store.clear();
  constr_dep_point.clear();
  constr_dep_fragment.clear();
  changed_point_vars.clear();
  changed_fragment_vars.clear();
  nchanged_point = 0;
  nchanged_frag  = 0;
}//-


/*
 * @note: Arc Consistency 3, only for non-labelled variables.
 * Arc reduction only active for those variables that HAVE BEEN CHANGED
 */
bool
ConstraintStore::arc_consistency3(size_t trailtop) {
  string dbg = "ConstraintStore::AC3() - ";
  /*
   * Initialization:
   * add constraint dependencies for Var changed to the constraint list
   */
  add_constr_dep_var_point_changed();
  add_constr_dep_var_fragment_changed();
  sort (store.begin(), store.end(), constraint_cmp);
#ifdef CSTORE_DBG
  cout << dbg << "constraint store sotred\n";
#endif
  /*
   * Before the ac loop all changed flags are = 0  and corresponding 
   * lists are emptied -> transfered all into constraint_store
   */
  nchanged_point = nchanged_frag = 0;

  // AC Loop
  bool failing = false;
  while (store.size()>0 && !failing) {
    Constraint* c = fetch();
#ifdef CSTORE_DBG
    cout << dbg << "constraint fetched:";
    c->dump(false);
#endif
    
    if(!c->propagate (trailtop)){
#ifdef CSTORE_DBG
      cout << dbg << "propagation failed\n";
#endif
      failing = true;
    }
    else if(!c->consistency()) { 
#ifdef CSTORE_DBG
      cout << dbg << "consistency failed\n";
#endif
      failing = true;
    }

    // Constraint resets the trigger causes
    //c->reset_causes();
    //c->clear_clauses();
    c->clear_causes();
    if (!failing) {
#ifdef CSTORE_DBG
      cout << dbg << "ok\n";
#endif
      add_constr_dep_var_point_changed();
      add_constr_dep_var_fragment_changed();
      sort(store.begin(), store.end(), constraint_cmp);
      nchanged_frag = nchanged_point = 0;
    }
  }//---ac3 loop

  /*
   * In case of failure -> 
   * reset changed flag for every logic variable 
   */
  if(failing) {
    g_logicvars->reset_allvars_changed();

    /* 
     * Flush everything from store (could not be emptied by completing
     * the loop).
     * @note: the backtrack from the trailstack will be done in the search 
     * method, after returning from AC3 
     */
    for(uint i=0; i < store.size(); i++)
      store[i]->reset_causes();
    store.clear();
    return false;
  }
  return true;
}//arc_consistency3


/*
 * Given a constraint C we allow expressions of the form: l{C}K.
 * Such expression is inteded as: at least l, but not more then k solutions
 * satisfing constraint C are model for the set of solutions returned.
 * This function return a the farther backjump index if some cardinality
 * constraint is not satisfied or max::size_t otherwise.
 */
bool 
ConstraintStore::check_cardinality_constraints(size_t& backjump) {
  string dbg = "ConstraintStore::check_cardinality_constraints() - "; 
  size_t min_bk_jump = std::numeric_limits<size_t>::max(),
    bk_jump = std::numeric_limits<size_t>::max();
  
  for (int i = g_constraints.size()-1; i >= 0 ; i--) {
    if (g_constraints[i]->is_cardinality_constraint ()) {
      g_constraints[i]->check_cardinality (bk_jump);
      min_bk_jump = std::min (min_bk_jump, bk_jump);
#ifdef CSTORE_DBG
      std::cout << dbg << "checking constraint: " << i << endl;
      g_constraints[i]->dump(false);
      std::cout << " bkjump: " << backjump << std::endl;
#endif
      break; //// NOTE THIS IS TO LET THE SOURCE_SINK_CONSTRAINT WORK AS 
      /// ALE NEW SEMANTICS -- which does not fit for the general case! 
    }
  }
  backjump = min_bk_jump;
#ifdef CSTORE_DBG
  std::cout << "bkjump returned is: " << backjump << std::endl;
#endif
  return (backjump == std::numeric_limits<size_t>::max()) ? true
	  : false;
}//-


/* @note: for all the Point variables that have CHANGED, add the constraints 
 * to the costraint list, that  involve this variable.
 * These constraints will be checked for propagation in the AC loop.
 */
void
ConstraintStore::add_constr_dep_var_point_changed() {
  // @todo find from the old dimension to the new one!
  vector<Constraint*>::iterator it;
#ifdef CSTORE_DBG
  cout << "PT nchanged: " << nchanged_frag << endl;
#endif
  for(uint i=0; i < nchanged_point; i++) {
    VariablePoint* vp_pt = changed_point_vars.at(i);
#ifdef CSTORE_DBG
    cout << "PT changed: VP_" << vp_pt->idx() << endl;
#endif        
    // Iterate on all the constr in which the changed var is involved
    for (uint j=0; j < vp_pt->constr_dep.size(); j++) {
      vp_pt->set_changed(false); //reset status
      Constraint* constr_dep = vp_pt->constr_dep.at(j);
#ifdef CSTORE_DBG
      cout << "\tconstr dept C_: " << vp_pt->constr_dep[j]->get_id() << endl;
      constr_dep->dump(false);
      cout << endl;
#endif


      /*
       * If the constraint is shared among different variables 
       * changed add it only once
       * @hack: do not need to search in the whole constraint store!!
       */
      it = find (store.begin(), store.end(), constr_dep);
      constr_dep->save_cause(vp_pt);

      // Add the variable as reason to trigger propagation.
      if (it == store.end()) {
#ifdef CSTORE_DBG
      	cout << "\tadding var\n";
#endif
	store.push_back(constr_dep);
      }
#ifdef CSTORE_DBG
      else
	cout << "\talready present\n";
#endif
    }
  }
}//-

void
ConstraintStore::add_constr_dep_var_fragment_changed() {
  vector<Constraint*>::iterator it = store.begin();
#ifdef CSTORE_DBG
  cout << "nchanged: " << nchanged_frag << endl;
#endif
  for (uint i=0; i < nchanged_frag; i++) {
    VariableFragment* vf_ptr = changed_fragment_vars.at(i);
#ifdef CSTORE_DBG
   cout << "frag changed: VF_" << vf_ptr->get_idx() << endl;
#endif    
    // Iterate on all the constr in which the changed var is involved
    for (uint j=0; j < vf_ptr->constr_dep.size(); j++) {
      vf_ptr->set_changed(false); // reset status
      Constraint* constr_dep = vf_ptr->constr_dep.at(j);
#ifdef CSTORE_DBG
      cout << "\tconstr dept C_: " << vf_ptr->constr_dep[j]->get_id() << endl;
      constr_dep->dump(false);
      cout << endl;
#endif
      /*
       * If the constraint is shared among different variables 
       * changed add it only once
       * @hack: do not need to search in the whole constraint store!!
       */
      //      it = find (store.begin(), store.end(), constr_dep);
      for (it = store.begin(); it != store.end(); ++it) {
	if ((*it)->get_id() == constr_dep->get_id()) 
	  break;
      }
      constr_dep->save_cause(vf_ptr);

      // Add the variable as reason to trigger propagation.
      if (it == store.end()) {
#ifdef CSTORE_DBG
	cout << "\tadding var\n";
#endif
	store.push_back(constr_dep);
      }
#ifdef CSTORE_DBG
      else	
	cout << "\talready present\n";
#endif
    }
  }
}//-

// @note: check variables not pushed more then once
void
ConstraintStore::upd_changed(VariablePoint *v) {
  if (v->is_changed()){
    if (nchanged_point >= changed_point_vars.size())
      changed_point_vars.push_back(0);
    changed_point_vars[nchanged_point++] = v;
  }
}//-

// @note: check variables not pushed more then once
void
ConstraintStore::upd_changed(VariableFragment *v) {
  if (v->is_changed()) {
    if (nchanged_frag >= changed_fragment_vars.size())
      changed_fragment_vars.push_back(0);
    changed_fragment_vars[nchanged_frag++] = v;
  }
}//-

void
ConstraintStore::dump() {
  cout << "DBG: constraint-store content:" << endl;
  for (uint i=0; i<store.size(); i++) {
    store.at(i)->dump();
  }
  cout <<"     --------------------------" << endl;
}//-
