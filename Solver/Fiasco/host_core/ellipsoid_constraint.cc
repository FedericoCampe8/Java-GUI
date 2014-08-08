#include "ellipsoid_constraint.h"
#include "utilities.h"
#include "mathematics.h"
#include "globals.h"
#include "constraint_store.h"
#include "trailstack.h"
#include "logic_variables.h"
#include "variable_point.h"
#include "statistics.h"

#include <cmath>
#include <vector>
#include <stdlib.h> // for atoi

using namespace std;

EllipsoidConstraint::EllipsoidConstraint 
  (int argc, char* argv[], int& parse_pos,
   const R_MAT& rot_ellipsoid, const vec3& sh_ellipsoid, int _weight) {

  weight = _weight;
  
  for (; parse_pos < argc; parse_pos++) {
    
    if (!strcmp ("--ellipsoid", argv[parse_pos])) {
      point focal_1, focal_2;
      real sum_radii=0;

      while (parse_pos < (argc-1) && 
	     strcmp(":", argv[++parse_pos])) {
	int vpt_idx = Utilities::get_bbidx_from_aaidx (atoi(argv[parse_pos]), CA); 
	vpt.push_back (&g_logicvars->var_point_list[vpt_idx]);
      }
      if (parse_pos < (argc-1) &&  
	  !strcmp ("f1=", argv[++parse_pos])) {
	focal_1[0] = atof (argv[++parse_pos]);
	focal_1[1] = atof (argv[++parse_pos]);
	focal_1[2] = atof (argv[++parse_pos]);
      }
      if (parse_pos < (argc-1) && 
	  !strcmp ("f2=", argv[++parse_pos])) {
	focal_2[0] = atof (argv[++parse_pos]);
	focal_2[1] = atof (argv[++parse_pos]);
	focal_2[2] = atof (argv[++parse_pos]);
      }
     if (parse_pos < (argc-1) && 
	 !strcmp ("sum-radii=", argv[++parse_pos])) {
       sum_radii = atof(argv[++parse_pos]);
     }

     // Set Ellipsoid equation
     if (rot_ellipsoid != NULL && sh_ellipsoid != NULL) {
       Math::translate (focal_1, sh_ellipsoid);
       Math::rotate_inverse (focal_1, rot_ellipsoid);
       Math::translate (focal_2, sh_ellipsoid);
       Math::rotate_inverse (focal_2, rot_ellipsoid);
     }

     Math::middle_point (focal_1, focal_2, center);
     real dff = Math::eucl_dist(focal_1, focal_2);
     a = sum_radii/2;
     b = c = sqrt(a*a - dff*dff);          
     a2 = a*a; 
     b2 = b*b;
     c2 = c*c;
     //c = Math::eucl_dist(focal_1, focal_2); c2 = c*c;
     //b = sqrt((c*c)-(a*a));                 b2 = b*b;
     
     // Add dependencies
     this->add_dependencies();
     g_constraints.push_back (this);

#ifdef VERBOSE
     std::cout << "ELLIPSOID constraint (c_" << get_id() 
	       << ") created : ";
     for (int i = 0; i < vpt.size(); i++) {
       std::cout << "CA_" << vpt[i]->idx() << ", ";  
     }
     std::cout << std::endl;
#endif
    }//
  }

  // invalidate parser position for next constraint handling
  if (parse_pos == argc) 
    parse_pos = -1;
  
}//-


bool
EllipsoidConstraint::consistency() {
  // check consistency -- NOT DEFAULT CHECK HERE -- careful (rewrite this
  // function using  bound consistency)
  bool b = vpt[0]->test_intersect_ellipsoid (a2, b2, c2, center);
  if (!b) 
    g_statistics->incr_propagation_failures(__c_ellipsoid);
  else 
    g_statistics->incr_propagation_successes(__c_ellipsoid);
  return b;
}//-


bool 
EllipsoidConstraint::propagate(int trailtop) {
  return true;
  // It is not necessary for the point involved in the constraint to be ground.
  // If is not, we need to trail this operation: the intersection may change
  // the domain of the point.
  // If the point is ground we verify if it lies inside it (consistency).
  for (int i = 0; i < vpt.size(); i++) {
    VariablePoint* p = vpt[i];

    if (!p->is_ground()) {

      if (p->lower_bound[0] == -255) continue; // not yet touched

      real ol[3],oh[3];
      memcpy(ol, p->lower_bound, sizeof(point));
      memcpy(oh, p->upper_bound, sizeof(point));
    
      if (!p->intersect_ellipsoid (a2,b2,c2,center)) {
	g_statistics->incr_propagation_failures(__c_ellipsoid);
	return false;
      }
      if (p->is_failed()) {
	g_statistics->incr_propagation_failures(__c_ellipsoid);
	return false;
      }
      if (p->is_changed()) {
	g_constraintstore.upd_changed (p);
	g_trailstack.trail_variable_point (p, ol, oh, trailtop);
      }
    }
    g_statistics->incr_propagation_successes(__c_ellipsoid);

  }
  
  return true;
}//-


void 
EllipsoidConstraint::dump(bool all) {
  std::cout << "Ellipsoid constraint (c_" << get_id()  << ")  ";
  if (all) {
    cout << "Params: (a=" << a << ", b=" << b << ",c=" << c << ")" << endl;
    for (uint i=0; i<vpt.size(); i++)
      cout << " vPt_" << vpt.at(i)->idx();
    cout << endl;
  }
}//-
