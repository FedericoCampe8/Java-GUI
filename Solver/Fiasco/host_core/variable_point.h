/*********************************************************************
 * CP-Variable POINT definition
 * 
 * A point variable describes an atom position. 
 * It's domain is a the possible points within the interval
 * [lower_bound, upper_bound]
 *
 * References:
 * (wcb-11)
 *********************************************************************/
#ifndef FIASCO_VARIABLE_POINT__
#define FIASCO_VARIABLE_POINT__

#include "typedefs.h"
#include "atom.h"
//#include "constraint.h"

#include <vector>
class Constraint;

class VariablePoint {
 private:
  int labeled;      //-1    if not labeled;
                    // >=0  choice value after labeling
  bool ground;
  bool failed;
  bool changed;
  size_t last_trailed;

  int _idx;        // index in the Point var array (@todo improve?)
  int _in_var_fragment;
  
 public:
  point lower_bound;
  point upper_bound;
  // List of constraints to be checked after variable is changed
  std::vector<Constraint*> constr_dep;

  VariablePoint () {};
  VariablePoint (int vidx);
  ~VariablePoint (){};
  VariablePoint (const VariablePoint& other);
  VariablePoint& operator= (const VariablePoint& other);
  real operator[]  (int index) const;

  int idx() const {return _idx;}
  int in_var_fragment () const {return _in_var_fragment;}  
  void in_var_fragment (int n) {_in_var_fragment = n;}    
  bool is_ground() {return ground ? true : false;}
  bool is_failed() {return failed ? true : false;}
  bool is_changed() {return changed ? true : false;}
  void set_changed(bool bv){ changed = bv; }
  size_t get_last_trailed() {return last_trailed;}
  void set_last_trailed(size_t lt) {last_trailed=lt;}
  int  label() {return labeled;}
  void labeling(); // set labeling

  bool set_ground (const Atom& a);
  bool set_ground (const point& p);
  bool intersect_box (const point& lb, const point& ub);
  bool intersect_ellipsoid (const real a, const real b, const real c,
			    const point& center);
  bool test_intersect_ellipsoid (const real a, const real b, const real c, 
				 const point& center) const;
  void test_ground();

  void test_failed();
  void fail(bool b);
  void reset ();

  void dump();

};

#endif
