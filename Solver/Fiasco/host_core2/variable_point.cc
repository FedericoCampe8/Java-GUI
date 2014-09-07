#include "variable_point.h"
#include "utilities.h"
#include "mathematics.h"

#include <iostream>
#include <string.h>

using namespace std;
using namespace Utilities;

VariablePoint::VariablePoint(int vidx) {
  for (int i = 0; i < 3; i++) {
    lower_bound[i] = -256;
    upper_bound[i] = +256;
  }
  labeled      = -1;
  ground       = false;
  failed       = false;
  changed      = false;
  last_trailed = 0;
  _in_var_fragment = -1;
  _idx = vidx;
}//-

void
VariablePoint::reset () {
  for (int i = 0; i < 3; i++) {
    lower_bound[i] = -256;
    upper_bound[i] = +256;
  }
  labeled      = -1;
  ground       = false;
  failed       = false;
  changed      = false;
  last_trailed = 0;
  _in_var_fragment = -1;
  // Constraint dependencies should not be removed. 
  // Otherwise, the constraints will loose their
  // pointer to the current variable
  // constr_dep.clear();
}//-

VariablePoint::VariablePoint (const VariablePoint& other) {
  labeled = other.labeled;
  ground = other.ground;
  failed = other.failed;
  changed = other.changed;
  last_trailed = other.last_trailed;
  _idx = other._idx;
  _in_var_fragment = other._in_var_fragment;
  memcpy (lower_bound, other.lower_bound, sizeof(point));
  memcpy (upper_bound, other.upper_bound, sizeof(point));
  constr_dep = other.constr_dep;
}//-

VariablePoint& 
VariablePoint::operator= (const VariablePoint& other) {
  if (this != &other) {
    labeled = other.labeled;
    ground = other.ground;
    failed = other.failed;
    changed = other.changed;
    last_trailed = other.last_trailed;
    _idx = other._idx;
    _in_var_fragment = other._in_var_fragment;
    memcpy (lower_bound, other.lower_bound, sizeof(point));
    memcpy (upper_bound, other.upper_bound, sizeof(point));
    constr_dep = other.constr_dep;
  }
  return *this;
}//-


real
VariablePoint::operator[] (int index) const {
  if (index >= 0 && index < 3)
    return lower_bound[index];
  return -1;
} //-

// Given an atom, set the variable point value to ground.
bool
VariablePoint::set_ground (const Atom& a) {
  return intersect_box (a.position, a.position);
}//-

bool
VariablePoint::set_ground (const point& p) {
  return intersect_box (p, p);
}//-

//Intersect a domain variable with 2 new bounds.
bool
VariablePoint::intersect_box (const point& lb, const point& ub) {
  for (int i = 0; i < 3; i++) {
    if (lower_bound[i] < lb[i]) {
      lower_bound[i] = lb[i];
      changed = true;
    }
    if (upper_bound[i] > ub[i]) {
      upper_bound[i] = ub[i];
      changed = true;
    }
  }
  //if(changed)
  //update_changed_status();

  test_ground ();
  test_failed ();

  if(changed)
    return true;
  else
    return false;
}//-

bool
VariablePoint::test_intersect_ellipsoid 
(const real a2, const  real b2, const real c2, const point& center) const { 
  point l, u, cg, centroid;
  Math::vsub(upper_bound, lower_bound, centroid);
  Math::vsub(lower_bound, center, l);  
  Math::vsub(upper_bound, center, u);
  Math::vsub(centroid, center, cg);
  
  for (int i=0; i<3; i++) {
    cg[i] = cg[i]*cg[i];
    l[i]  = l[i]*l[i];
    u[i]  = u[i]*u[i];
  }

  // check if box is inside the ellipsoid area
  if ((cg[0]/a2 + cg[1]/b2 + cg[2]/c2) <= 1 &&
      (l[0]/a2 + l[1]/b2 + l[2]/c2) <= 1 &&
      (u[0]/a2 + u[1]/b2 + u[2]/c2) <= 1 ) {
    return true;
  }
  return false;
}//-


// Intersect current domain variable with an ellispoid 
// of parameters a, b, c.
// note: the intersection is approximated to a box
// note: for efficiency reasons a,b,c are squared
bool
VariablePoint::intersect_ellipsoid 
(const real a2, const real b2, const real c2, const point& center) {
  point l, u, cg, centroid;
  Math::vsub(upper_bound, lower_bound, centroid);
  Math::vsub(lower_bound, center, l);  
  Math::vsub(upper_bound, center, u);
  Math::vsub(centroid, center, cg);
  
  for (int i=0; i<3; i++) {
    cg[i] = cg[i]*cg[i];
    l[i]  = l[i]*l[i];
    u[i]  = u[i]*u[i];
  }
  
  // check if box is outside the ellipsoid area
  if ((cg[0]/a2 + cg[1]/b2 + cg[2]/c2) > 1 &&
      (l[0]/a2 + l[1]/b2 + l[2]/c2) > 1 &&
      (u[0]/a2 + u[1]/b2 + u[2]/c2) > 1 ) {
    return false;
  }
  if ((l[0]/a2 + l[1]/b2 + l[2]/c2) > 1) {
    lower_bound[0] = l[0]/a2;
    lower_bound[1] = l[1]/b2;
    lower_bound[2] = l[2]/c2;
    changed = true;
  }
  if ((u[0]/a2 + u[1]/b2 + u[2]/c2) > 1) {
    upper_bound[0] = u[0]/a2;
    upper_bound[1] = u[1]/b2;
    upper_bound[2] = u[2]/c2;
    changed = true;
  }
  
  test_ground ();
  test_failed ();
  return true;

}//-


// test if current domain is ground
void
VariablePoint::test_ground() {
  ground = true;
  for (int i = 0; i < 3; i++)
    if (lower_bound[i] != upper_bound[i])
      ground = false;
}//-

// test if current variable is failed
void
VariablePoint::test_failed() {
  failed = false;
  for (int i = 0; i < 3; i++)
      if (lower_bound[i] > upper_bound[i]){
          failed = true;
      }
}//-

void
VariablePoint::fail(bool b) {
  failed = b;
}//-

void
VariablePoint::dump () {
  cout << "VARIABLE POINT: (" << _idx << ")";
  cout << "Label: " << labeled << "\t";
  if (is_ground()) cout << " GROUND ";
  if (is_failed()) cout << " FAILED ";
  if (is_changed()) cout << " CHANGED ";
  cout << "Last Trailed: " << last_trailed << endl;
  Utilities::dump(lower_bound, upper_bound);
  cout << endl;
}//-
