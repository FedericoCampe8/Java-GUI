#include "bundle_constraint.h"
#include "globals.h"
#include "constraint_store.h"
#include "fragment.h"
#include "logic_variables.h"
#include "mathematics.h"
#include "protein.h"
#include "trailstack.h"
#include "utilities.h"
#include "variable_fragment.h"
#include "variable_point.h"
#include "statistics.h"

#include <string>
#include <cassert>

using namespace std;
  
BundleConstraint::BundleConstraint(pair<VariableFragment*, VariableFragment*> vf_ptr, 
		 pair<int, int> l_bundle, bool bi) :
  use_bidirectional (bi) {
  weight = 2;
  bundle_info = l_bundle;
  vfrag.push_back (vf_ptr.first);
  vfrag.push_back (vf_ptr.second);
  
  this->add_dependencies ();
  g_constraints.push_back (this);

#ifdef VERBOSE
  std::cout << "BUNDLE constraint (c_" << get_id() 
	    << ") created : ( ";
  for (int i = 0; i < vfrag.size(); i++)
    std::cout << "Var-Frag_" << vfrag[i]->get_idx() << ", ";  
  std::cout << std::endl;
#endif

}//-


bool 
BundleConstraint::consistency(){
  return true;
}//-


// Assume we have two fragments in the bundle: f1, f2.
// If f1 is ground, we retireve the label of f1, and 
// the link information, from f1, that points to the label
// of the element of the fragment var domain of f2.
// with this information we can propagate,
// assume to work on pairs, and that v[0] is the bundle
bool
BundleConstraint::propagate (int trailtop) {
  string dbg = "ConstraintBundle::propagate() - ";
  // !!!!!!!!! for the moment treat this as a pair!!!!!!!!!
  assert (bundle_info.first >= 0 || bundle_info.second >= 0);
  
  // Check propagtion conditions
  int label_f1 = vfrag[0]->get_label();  // F1
  frag_info f2_info = vfrag[0]->get_domain_elem(label_f1); // F1
  
  if (bundle_info.first   != label_f1         || //
      vfrag[1]->get_idx() != f2_info.first    || //  VF2 idxx
      bundle_info.second  != f2_info.second) {   // label VF2
    return true;
  }
  
  // check propagation conditions
  if (!vfrag[0]->is_ground()) {// || vfrag[1]->is_ground())
    return true;
  }
  // set continuation for backtraking on bundles
  g_trailstack.set_continuation();

  // HACK! mix the order later
  Fragment* f_g  = &vfrag[0]->domain[bundle_info.first];    // f ground
  Fragment f_ng (vfrag[1]->domain[bundle_info.second]);
    
  // propagate second fragment of the pair
  f_ng.transform (f_g->rot_m, f_g->shift_v);
    
  // TRAIL CHANGE for rot and shift vec of frag 2?
  for (uint i=0; i < f_ng.backbone_len(); i++) {
    // skip last N atom of last fragment mapping
    if (i+f_ng.get_bb_s() >= g_target.get_bblen())
      break;
      
    VariablePoint* p = &(g_logicvars->var_point_list[i + f_ng.get_bb_s()]);
      
    // Set the point variable to ground
    if (!p->is_ground()) {
      real ol[3],oh[3];
      memcpy(ol, p->lower_bound, sizeof(point));
      memcpy(oh, p->upper_bound, sizeof(point));

//#ifdef FALSE
      /* Set bounds for the first three points of f2 */
      if (!use_bidirectional && i < 3) {
	point u, l;
	real epsilon;
	switch (i) {
	case 0:
	  epsilon = (cC + cO + cN)*4; // C'
	  break;
	case 1:
	  epsilon = (cO + cN)*3; // O
	  break;
	case 2:
	  epsilon = cN*2; // N
	  break;    
	default:
	  epsilon = (3-i)*0.8;//(3-i)*0.2;
	  break;
	}//switch
        
	epsilon /= 100;
	Math::vadd(f_ng.backbone[i].position, epsilon, u);
	Math::vsub(f_ng.backbone[i].position, epsilon, l);
	p->intersect_box (l, u);
      }
      else {
        
//#endif
        
	p->set_ground(f_ng.backbone[i]);
	p->set_last_trailed (trailtop); // save current-state
        
//#ifdef FALSE
	      }
//#endif
      
      p->in_var_fragment(f_ng.get_id());
      
      if (p->is_changed()) {
	g_constraintstore.upd_changed(p);
	g_trailstack.trail_variable_point (p, ol, oh, trailtop);
      }
      
      if (p->is_failed()) {
	g_statistics->incr_propagation_failures(__c_bundle);
	return false;
      }
    }
  }
    //  vfrag[1]->test_ground(); // Hack!!
  vfrag[1]->set_ground();
  vfrag[1]->set_last_trailed (trailtop); // save current-state
  // std::cout << dbg << "Set VF_" << vfrag[1]->get_idx() 
  // 	    << " last trailed: " << trailtop << std::endl;


  //  g_constraintstore.upd_changed(vfrag[1]);
  g_statistics->incr_propagation_successes(__c_bundle);
  return true;  
}//-


void 
BundleConstraint::dump(bool all) {
  std::cout << "Bundle Constraint (w_" << get_weight()  << ")";
  if (all) {
    for (uint i=0; i<vfrag.size(); i++)
      cout << " vF_" << vfrag.at(i)->get_idx();
    cout << "\n           Vars:";
    for (uint i=0; i<vpt.size(); i++)
      cout << " vPt_" << vpt.at(i)->idx();
  }
  cout << endl;
}//-
