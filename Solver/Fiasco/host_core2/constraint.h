/*********************************************************************
 * Constraint
 * Constraint base class implementation.
 *********************************************************************/
#ifndef FIASCO_CONSTRAINT_
#define FIASCO_CONSTRAINT_

#include "typedefs.h"

#include <vector>
#include <iostream>

class VariablePoint;
class VariableFragment;


class Constraint {
 protected:
  uint id;
  uint ncaused_by_vpt;
  uint ncaused_by_vfrag;
  bool consistent;  //  0 is consistency not yet determined,
                    //  1 if all vars assigned and consistent
                    //  (no more consistency attemps are done)
  bool propagated;
  int weight;       // constraints with lower weights are woken
                    // up before constraints with higer weights

  // Cardinality constraint
  std::pair<int, int> cardinality; 
  uint current_cardinality;
  bool synergic;

 public:
  // Variables involved in a constraint
  std::vector<VariablePoint*>    vpt;
  std::vector<VariableFragment*> vfrag;
  // When a constraint propagation is invoked, here is the list
  // of variables that are changed and cause further propagation
  std::vector<VariablePoint*>    caused_by_vpt;
  std::vector<VariableFragment*> caused_by_vfrag;
  

  Constraint();
  virtual ~Constraint();
  
  bool operator() (const Constraint& ci, const Constraint& cj) {
    return ( ci.weight > cj.weight); 
  }//-

  int get_weight() const {return weight;}
  void set_id(uint cid) {id = cid;}
  uint get_id() const {return id;}
  uint get_ncaused_by_vpt()  const  {return ncaused_by_vpt;}
  uint get_ncaused_by_vfrag() const {return ncaused_by_vfrag;}
  void save_cause(VariablePoint *v);
  void save_cause(VariableFragment *v);
  void reset_causes();
  void clear_causes();
  bool is_consistent() { return consistent; }
  void set_consistent(bool b=true) { consistent = b; }
  bool is_propagated() { return propagated; }
  void set_propagated(bool b=true) { propagated = b; }  
  void add_dependencies();
  void set_synergic (bool b=true) {synergic = b; }
  bool is_synergic() {return synergic; }

  // cardinality constraint methods
  void set_cardinality (uint min=1, uint max=1);
  bool check_min_cardinality () {return (current_cardinality >= cardinality.first); }
  bool check_max_cardinality () {return (current_cardinality <= cardinality.second); }
  void incr_cardinality (uint n=1) {current_cardinality += n; }
  uint get_cardinality () {return current_cardinality; }
  void reset_cardinality () {current_cardinality = 0; }
  bool is_cardinality_constraint ();

  void reset();

  virtual bool propagate (int trailtop) = 0;
  virtual bool consistency () = 0;
  virtual void pre_backtrack_process () {};
  virtual void post_backtrack_process () {};
  virtual bool check_cardinality (size_t& backjump) = 0;
  virtual bool synergic_consistency (const point& p, atom_type t = CA, int aa_idx = 0) = 0;
  virtual void reset_synergy () = 0;
  virtual void dump (bool all=true) = 0;
};//-

#endif
