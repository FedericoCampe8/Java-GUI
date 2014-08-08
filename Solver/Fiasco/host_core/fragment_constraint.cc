#include "fragment_constraint.h"
#include "globals.h"
#include "anchor.h"
#include "atom.h"
#include "constraint_store.h"
#include "mathematics.h"
#include "logic_variables.h"
#include "trailstack.h"
#include "protein.h"
#include "utilities.h"
#include "variable_fragment.h"
#include "variable_point.h"
#include "statistics.h"

#include <string>
#include <stdio.h>

using namespace std;
//#define FRAG_DBG

ConstraintFragment::ConstraintFragment (VariableFragment *f_ptr, 
					AssemblyDirection dir) :
  overlap_plane(dir) {
  weight = 1;
  vfrag.push_back (f_ptr);
  
  this->add_dependencies();
  g_constraints.push_back(this);
#ifdef VERBOSE
     std::cout << "FRAGMENT constraint (c_" << get_id() 
	       << ") created : Var-Frag_" << f_ptr->get_idx()
	       << std::endl;
#endif

}//-


bool
ConstraintFragment::propagate (int trailtop) {
  string dbg = "ConstraintFragment::propagate() - ";
  VariableFragment* variable = vfrag.at(0);
  int label = variable->get_label();
#ifdef FRAG_DBG
  cout << dbg << "V_" << variable->get_idx() << "(" << label << ")\n";
#endif
  
  if (label < 0) { // Check labeling status
#ifdef FRAG_DBG
    cout << dbg << "Labling not set - ret FALSE\n";
#endif
    return false;
  } 
  Anchor *hook = NULL;  
      
  Fragment* f = &(variable->domain.at(label));
  Fragment f_geo(variable->domain.at(label));
  int vpt_s = f->get_bb_s(); // C'
  int vpt_e = f->get_bb_e(); // N
    
  // Check proagation conditions
  bool propagate_conditions = false;  
  if (variable->assembly_direction() == LEFT_TO_RIGHT) {
    propagate_conditions = 
      vpt_s == 0 ||
      (g_logicvars->var_point_list[vpt_s+0].is_ground() &&
       g_logicvars->var_point_list[vpt_s+1].is_ground() &&
       g_logicvars->var_point_list[vpt_s+2].is_ground());
    if (propagate_conditions)
      hook = new Anchor (g_logicvars->var_point_list[vpt_s+0].lower_bound,
			 g_logicvars->var_point_list[vpt_s+1].lower_bound,
			 g_logicvars->var_point_list[vpt_s+2].lower_bound);
  }
  else {
    propagate_conditions = 
      (g_logicvars->var_point_list[vpt_e-0].is_ground() &&
       g_logicvars->var_point_list[vpt_e-1].is_ground() &&
       g_logicvars->var_point_list[vpt_e-2].is_ground());
    if (propagate_conditions)
      hook = new Anchor (g_logicvars->var_point_list[vpt_e-2].lower_bound,
			 g_logicvars->var_point_list[vpt_e-1].lower_bound,
			 g_logicvars->var_point_list[vpt_e-0].lower_bound);
  }
  //-

  if (propagate_conditions) {
    // For the first fragment we need a special treatment of the 
    // first two atoms C' O not used -- 
    // to simplify the operation we do not translate points on 0,0,0    
    if (vpt_s > 0) {
      f_geo.overlap (*hook, variable->assembly_direction());
    }      
      f->copy_rot_mat(f_geo);
      f->copy_sh_vec(f_geo);
  }

    // Access to the Pt Variables starting from atom ptvar_start
    uint skip = 0;
    if (vpt_s == 0)
      while (!f->backbone[skip].is_type(N)) 
	skip++;
    for (uint i=0; i+skip < f->backbone_len(); i++) {
      // skip last N atom of last fragment mapping
      if (i+skip+vpt_s >= g_target.get_bblen()){
	break;
      }
      
      VariablePoint* p = &(g_logicvars->var_point_list[i + vpt_s]);
      VariablePoint p_aux = g_logicvars->var_point_list[i + vpt_s];
      
      // Set the point variable to ground      
      if (!p->is_ground()) {
	real ol[3],oh[3];
	memcpy(ol, p->lower_bound, sizeof(point));
	memcpy(oh, p->upper_bound, sizeof(point));
	p_aux.set_ground(f_geo.backbone[i+skip]);
	  
	if(p_aux.is_failed()) {
	  g_statistics->incr_propagation_failures(__c_fragment);
	  if (hook) delete hook;
	  return false;
	}
	if (p_aux.is_changed()) {
	  p->set_ground (f_geo.backbone[i+skip]);
	  p->in_var_fragment (f_geo.get_id());
          p->set_last_trailed (trailtop); // save current-state
	  g_constraintstore.upd_changed(p);
	  g_trailstack.trail_variable_point (p, ol, oh, trailtop);
        }
      }
    }

    // set var fragment to ground
    vfrag[0]->test_ground();
    vfrag[0]->set_last_trailed (trailtop); // save current-state
    g_constraintstore.upd_changed(vfrag[0]);
    g_statistics->incr_propagation_successes(__c_fragment);
    //- 
    if (hook) delete hook;
    return true;
}//-


bool
ConstraintFragment::consistency() {
  return true;
}
//-
 

void 
ConstraintFragment::dump(bool all) {
  std::cout << "Fragment constraint (w_" << get_weight()  << ")  ";
  if (all) {
    for (uint i=0; i<vfrag.size(); i++)
      cout << " vF_" << vfrag.at(i)->get_idx();
    cout << "\n           Vars:";
    for (uint i=0; i<vpt.size(); i++)
      cout << " vPt_" << vpt.at(i)->idx();
  }
  cout << endl;
}//-
