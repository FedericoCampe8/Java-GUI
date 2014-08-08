#include "jm_constraint.h"
#include "typedefs.h"
#include "globals.h"
#include "constraint.h"
#include "constraint_store.h"
#include "table_constraint.h"
#include "utilities.h"
#include "trailstack.h"
#include "atom.h"
#include "atom_grid.h"
#include "mathematics.h"
#include "logic_variables.h"
#include "variable_fragment.h"
#include "variable_point.h"
#include "statistics.h"
#include "anchor.h"
#include "bitset.h"
#include "b_cluster.h"
#include "k_medoids.h"

#include <iostream>
#include <string>
#include <cmath>
#include <vector>
#include <set>
#include <algorithm>
#include <iterator>

#include <stdio.h>
#include <stdlib.h>

using namespace std;
using namespace Utilities;

//#define JM_DBG

JMConstraint::JMConstraint
(int argc, char* argv[], int& parse_pos, int& scanner) {
  string dbg = "Jmconstraint::Constructor - ";

  weight = 4;
  
  int aa_s = 0, aa_e = 0;
  int jm_index = -1;
  for (; parse_pos < argc; parse_pos++) {   
    
    if (!strcmp ("--jm", argv[parse_pos])) {
      jm_index = parse_pos;
      _front_anchor_is_fixed = false;
      _end_anchor_is_fixed = false;
      head_of_jm_multibody = (scanner == 0) ? true
	: false;
      
      while (parse_pos < argc && 
	     strcmp(":", argv[parse_pos])) {

	aa_s = atoi(argv[++parse_pos]) + scanner;
	if (!strcmp("->", argv[++parse_pos])){
	  _front_anchor_is_fixed = true;
	}
	else if (!strcmp("<-", argv[++parse_pos])) {
	  _end_anchor_is_fixed   = true;
	}
	else if (!strcmp("<->", argv[++parse_pos])) {
	  _front_anchor_is_fixed = true;
	  _end_anchor_is_fixed   = true;	 
	}
	aa_e = atoi(argv[++parse_pos]);
	
       	_front_anchor.set_associated_variable (aa_s);
	_end_anchor.set_associated_variable (aa_e);
	// N CA C' O 
	if (_front_anchor_is_fixed) {
	  int idxC = get_bbidx_from_aaidx (aa_s-1, CB);
	  int idxO = get_bbidx_from_aaidx (aa_s-1, O);
	  int idxN = get_bbidx_from_aaidx (aa_s, N);
	  vpt.push_back (&g_logicvars->var_point_list[idxC]); 
	  vpt.push_back (&g_logicvars->var_point_list[idxO]);
 	  vpt.push_back (&g_logicvars->var_point_list[idxN]);
	}
	if (_end_anchor_is_fixed) {
	  int idxC = get_bbidx_from_aaidx (aa_e, CB);
	  int idxO = get_bbidx_from_aaidx (aa_e, O);
	  int idxN = get_bbidx_from_aaidx (aa_e+1,   N);
	  vpt.push_back (&g_logicvars->var_point_list[idxC]); 
	  vpt.push_back (&g_logicvars->var_point_list[idxO]); 
	  vpt.push_back (&g_logicvars->var_point_list[idxN]);
	}
	parse_pos++;
      }
      
      // Default Clustering parameters
      int kmin = 60, kmax = 100;
      real cluster_max_distance    = 0.5;
      real cluster_max_orientation = Math::deg_to_rad(15);
      // Default anchor THS (intersections) JM parameters
      ths_anchor_distance = 0, ths_anchor_orientation = 0;

      if (parse_pos < (argc-1) && 
	  !strcmp ("numof-clusters=", argv[++parse_pos])) {
	kmin = atoi(argv[++parse_pos]);
	kmax = atoi(argv[++parse_pos]);
      }
      if (parse_pos < (argc-1) && 
	  !strcmp ("sim-params=", argv[++parse_pos])) {
	cluster_max_distance    = atof(argv[++parse_pos]);
	cluster_max_orientation = Math::deg_to_rad(atof(argv[++parse_pos]));
      }
      if (parse_pos < (argc-1) && 
	  !strcmp ("tolerances=", argv[++parse_pos])) {
	ths_anchor_distance    = atof(argv[++parse_pos]);
	ths_anchor_orientation = Math::deg_to_rad(atof(argv[++parse_pos]));
      }

      // Add synergies%   $\domQ = $ 

      int idxCa = get_bbidx_from_aaidx (aa_e, CA);
      for (int i = 0; i < g_constraints.size(); i++) {
	Constraint* c = g_constraints[i];
	if (c->is_synergic()) {
	  for (int vp = 0; vp < c->vpt.size(); vp++) {
	    if (c->vpt[vp]->idx() == idxCa) { 
	      this->add_synergy_constraint (c);
#ifdef JM_DBG
	      std::cout << "JM Synergy: ";
	      c->dump(false);
#endif
	      break;
	    }
	  }
	}
      }
      //-

      // Allocate Memory
      Kmedoids_alg = new K_Medoids (kmin, kmax, cluster_max_distance, cluster_max_orientation);
      clusters_left_to_right.resize(aa_e-aa_s+1);
      clusters_left_to_right_size.resize(aa_e-aa_s+1);
      domains_filtered.resize(aa_e-aa_s+1);
      for (int i=0; i<clusters_left_to_right.size(); i++)
	clusters_left_to_right[i].resize(kmax);
      //-


      // Add dependencies
      this->add_dependencies();
      g_constraints.push_back (this);

      // Report:
#ifdef VERBOSE
      dump();
#endif

      TableConstraint* table_constraint = 
       	new TableConstraint (aa_s, aa_e, kmax); 
      // table has just been added in g_constraints.
      table = (TableConstraint*)(g_constraints[g_constraints.size()-1]);
#ifdef JM_PREPROCESS
      table_done = false;
#endif
      scanner++;
      //-

      // break condition for new synergy stuff
      aa_s = aa_e-1;

      // done reading this jm constraint (all of it)
      if (aa_s == aa_e-1) {
	scanner = 0;
 	return;
      }
      else {
	// done reading a partial jm
	parse_pos = jm_index;
	return;
      }
    }
  }
  scanner = -1;
  parse_pos = -1;

}//-


bool
JMConstraint::consistency() {
  return true;
}
//-

bool
JMConstraint::propagate(int trailtop) {
  string dbg = "JMF::propagate() - ";
#ifdef JM_PREPROCESS
  if (table_done) {table->reset(); return true;} 
#endif
  
  // Check JM propagation conditions
  // Bidirectional requires both the anchors to be fixed, while monodirectional
  // only requires one.
  if (!front_anchor_is_fixed() && !end_anchor_is_fixed())
    return true;  // cannot propagate
  
  uint fC=0,fO=1,fN=2,eC=3,eO=4,eN=5;

  for (int i=0; i<vpt.size(); i++)
  if (front_anchor_is_fixed()) { // Chain growing in a leftmost order
    _front_anchor.set(vpt[fC]->lower_bound, vpt[fO]->lower_bound, vpt[fN]->lower_bound);
  }
  else { 
    _front_anchor.set(vpt[fC]->lower_bound, vpt[fC]->upper_bound, CB);
    _front_anchor.set(vpt[fO]->lower_bound, vpt[fO]->upper_bound, O);
    _front_anchor.set(vpt[fN]->lower_bound, vpt[fN]->upper_bound, N);  
  }
    
  if (end_anchor_is_fixed()) {  // Chain grouwing in a rightmost order
    _end_anchor.set(vpt[eC]->lower_bound, vpt[eO]->lower_bound, vpt[eN]->lower_bound);

    // useless
    int varfrag_idx  = _end_anchor.associated_variable_idx();
    int frag_labeled = g_logicvars->var_fragment_list[varfrag_idx].get_label();
    _end_anchor.set_associated_fragment (varfrag_idx, frag_labeled);
  }
  else {
    _end_anchor.set(vpt[fC]->lower_bound, vpt[fC]->upper_bound, CB);
    _end_anchor.set(vpt[fO]->lower_bound, vpt[fO]->upper_bound, O);
    _end_anchor.set(vpt[fN]->lower_bound, vpt[fN]->upper_bound, N);
  } 

  int chain_front_aa  = _front_anchor.associated_variable_idx();  
  int chain_end_aa    = _end_anchor.associated_variable_idx();
  //int chain_middle_aa = (chain_front_aa + chain_end_aa) /2;
  bool clusters_creation_flag, clusters_consistency_flag;

  g_statistics->set_timer (t_jm); 
    // BIDIRECTIONAL JM PROPAGATION
    if (front_anchor_is_fixed() && end_anchor_is_fixed()) {
      ;
      // cluster_rigid_bodies (
      //   clusters_left_to_right, clusters_left_to_right_size,
      // 	chain_front_aa, chain_middle_aa, LEFT_TO_RIGHT);

      // cluster_rigid_bodies (
      //   clusters_right_to_left, clusters_right_to_left_size, 
      // 	chain_middle_aa+1, chain_end_aa, RIGHT_TO_LEFT);

      // get_JMconsistent_rigid_bodies (
      // 	 clusters_left_to_right, clusters_right_to_left);
    }
    // SIMPLE JM PROPAGATION
    else if (front_anchor_is_fixed()) {
#ifdef JM_DBG
      cout << dbg << "Clustering L[" << _front_anchor.associated_variable_idx() 
	   << "]->[" << _end_anchor.associated_variable_idx() << "]R" << endl;
#endif
      clusters_creation_flag = 
	cluster_rigid_bodies (clusters_left_to_right, 
			      clusters_left_to_right_size, 
			      chain_front_aa, chain_end_aa, 
			      LEFT_TO_RIGHT);
#ifdef JM_DBG
      cout << dbg << "getting consistent Rgidi bodies" << endl;
#endif
      if (!clusters_creation_flag) {
#ifdef JM_DBG
	cout << dbg << "Failed Cluster creation\n";
#endif
	g_statistics->stopwatch (t_jm);
	return false;
      }
      clusters_consistency_flag = 
	get_JMconsistent_rigid_bodies (clusters_left_to_right, 
				       clusters_left_to_right_size);
      if (!clusters_consistency_flag) {
#ifdef JM_DBG
	cout << dbg << "Failed Cluster consistency\n";
#endif
	g_statistics->stopwatch (t_jm);
	return false;
      }
    }
    else if (end_anchor_is_fixed()) {
      cluster_rigid_bodies (
	 clusters_right_to_left, clusters_right_to_left_size,
	 chain_end_aa, chain_front_aa, RIGHT_TO_LEFT);

      get_JMconsistent_rigid_bodies (
         clusters_right_to_left, clusters_right_to_left_size);
    }
  g_statistics->stopwatch (t_jm);
        
  if (g_statistics->timeout())
    return false;


  // Set the new domains to eplore for each variable of the flexible chain
  int vf_idx  = _front_anchor.associated_variable_idx(); 
  int dom = 0;
  for ( ; vf_idx <= end_anchor()->associated_variable_idx(); vf_idx++, dom++ ) {
    VariableFragment *VF = &g_logicvars->var_fragment_list[vf_idx];
    g_trailstack.trail_variable_fragment (VF, VF->domain_info, trailtop);
    VF->set_domain_explored (domains_filtered[dom]); // explored := filtered 
  }

  if (domains_filtered[0].count(false) == 0) { // all filtered
    g_statistics->incr_propagation_failures(__c_look_ahead);
#ifdef JM_DBG
    std::cout << dbg << "No intersection found\n";
#endif
    return false;
  }
  g_statistics->incr_propagation_successes(__c_look_ahead);

  // Trail the synergic constraints only if this is the JM woken up by 
  // the leftmost variable
  if (head_of_jm_multibody) {
    for (int k = 0; k < synergy_constraints.size(); k++) {
#ifdef JM_DBG
      std::cout << dbg << "Trailing Synergy constraint: ";
      synergy_constraints[k]->dump(false);
      std::cout << std::endl;
#endif
      g_trailstack.trail_synergic_constraint ( synergy_constraints[k] ); 
    }
  }
  
  return true;
}
//-


/* 
 * This is a simple clustering process aimed at reaching the biggest
 * space coverage possible from one point (chain_front_aa) to another
 * point (chain_end_aa).
 * 
 * chain_front_aa and chain_end_aa are the start
 * and end amino acids (included) associated to the flexible chain 
*/
bool 
JMConstraint::cluster_rigid_bodies 
  (vector<vector<Linked_Anchor> >& linked_anchors,
   vector<uint>& linked_anchors_size,
   uint chain_front_aa, uint chain_end_aa, 
   AssemblyDirection direction) 
{  
  string dbg  = "JMconstraint::cluster_rigid_bodies() - ";

  if (direction == RIGHT_TO_LEFT)
    std::swap (chain_front_aa, chain_end_aa);
  
  /* 
   * This is the first level of the JMf. This phase does not rely on
   * previous clusters (output_clusters). We do not apply a clustering
   * step here, but every rigid body is forced into a separate cluster.
   */  
  set_of_anchors_in_cluster anchors_to_cluster; 
  const VariableFragment* var_frag = 
    &g_logicvars->var_fragment_list[chain_front_aa];  
  

  for (uint f_idx = 0; f_idx < var_frag->domain_size(); f_idx++) {
    // This should not be considered when TABLE CONSTRAINT is used
    // if (var_frag->is_domain_explored (f_idx)) {
    //   continue;
    // }
    Fragment fragment_aux = var_frag->at(f_idx);
    Anchor* end_anchor_to_cluster = NULL;

    if (direction == LEFT_TO_RIGHT) {
      fragment_aux.overlap (_front_anchor, LEFT_TO_RIGHT); 
      end_anchor_to_cluster = fragment_aux.end_anchor();
    }
    else {
      fragment_aux.overlap (_end_anchor, RIGHT_TO_LEFT);
      end_anchor_to_cluster = fragment_aux.front_anchor();
    }

    end_anchor_to_cluster->set_associated_fragment (chain_front_aa, f_idx);
    Linked_Anchor 
      current_anchor_in_cluster(*end_anchor_to_cluster);
    anchors_to_cluster.insert (current_anchor_in_cluster);
  }
 
  Kmedoids_alg->get_linked_anchors (
    linked_anchors[0], linked_anchors_size[0], anchors_to_cluster);

  /*
   * The end-anchors generated in every other level of the flexible chain
   * are processed via a kmenas clustering step. This is a recursive step:
   * we employ the anchors - cluster representatives - from a previous 
   * flexible level (in this case amino acid) to generate the placements
   * for the end-anchors associated with the current level.
   */
  uint cluster_idx = 0;
  uint vf_idx = chain_front_aa;
    
  while (vf_idx != chain_end_aa) {
    if (direction == LEFT_TO_RIGHT) vf_idx ++;
    if (direction == RIGHT_TO_LEFT) vf_idx --;
    cluster_idx++;

    anchors_to_cluster.clear(); // try to avoid this

    const VariableFragment* var_frag = 
      &g_logicvars->var_fragment_list[vf_idx];
    
#ifdef JM_DBG
    cout << dbg << "linked anchor["<<cluster_idx-1<<"]size: "
	 << linked_anchors_size[cluster_idx-1] << endl;
#endif
    for (uint c_idx = 0; c_idx < linked_anchors_size[cluster_idx-1]; c_idx++) {
      if (!linked_anchors[cluster_idx-1][c_idx].is_valid){
#ifdef JM_DBG
	cout << dbg << "linked anchor " << cluster_idx << "(" << c_idx 
	     <<") not valid\n"; 
#endif
	continue;
      }
      // Select a representative for the previous cluster
      const Anchor* prev_cluster_representative = 
	&(linked_anchors[cluster_idx-1][c_idx].representative);

      for (uint f_idx = 0; f_idx < var_frag->domain_size(); f_idx++) {
	// This should not be considered when TABLE CONSTRAINT is used
	// if (var_frag->is_domain_explored (f_idx)) {
	//   continue;
	// }
	
	Fragment fragment_aux = var_frag->at(f_idx);
	fragment_aux.overlap (*prev_cluster_representative, direction);
 
	Anchor* end_anchor_to_cluster = 
	  (direction == LEFT_TO_RIGHT) ? fragment_aux.end_anchor() 
	  : fragment_aux.front_anchor();
	end_anchor_to_cluster->set_associated_fragment (vf_idx, f_idx);

	// Synergy check to remove possible anchors 
	bool check_sinergy = true;
	point end_anchor_ca;
	// Retrieve CA associated to this end anchor
	for (uint i = 0; i < fragment_aux.backbone_len(); i++)
	  if (fragment_aux.backbone[i].type == CA) {
	    memcpy (end_anchor_ca, fragment_aux.backbone[i].position, sizeof(point));
	    break;
	  }

	for (int k = 0; k < synergy_constraints.size(); k++) {
	  if (!synergy_constraints[k]->synergic_consistency (end_anchor_ca, CA, vf_idx)) {
	    // (end_anchor_to_cluster->get_N(), N, 
	    //  end_anchor_to_cluster->associated_variable_idx())) {
	    check_sinergy = false;
#ifdef JM_DBG
	    // cout << dbg << "Synergy failed: ";
	    // synergy_constraints[k]->dump(false); cout << endl;
#endif
	    break;
	  }
	}//-

	if (!check_sinergy) continue;

	Linked_Anchor
	  current_anchor_in_cluster (*end_anchor_to_cluster, 
				     &(linked_anchors[cluster_idx-1][c_idx]));
	anchors_to_cluster.insert (current_anchor_in_cluster);

      }// f_idx (flexible chain)
    }// c-idx
 
    // check if the constraint has not failed
    // this is the case in which all synergy constraint cause a failure at this
    // stage of clustering
    if (anchors_to_cluster.size() == 0) {
      return false;
    }

    /* 
     * To enhance the end-hooking-joint accuracy, anchors in the last_level of 
     * the flexible chain do not undergo the clustering process.
     */
#ifdef FALSE
    if (vf_idx == chain_end_aa) {
      Kmedoids_alg->get_linked_anchors (
	linked_anchors[cluster_idx], linked_anchors_size[cluster_idx], anchors_to_cluster);
    }
#endif
    Kmedoids_alg->make_clusters (
      linked_anchors[cluster_idx], linked_anchors_size[cluster_idx], anchors_to_cluster);   
  }// vf_idx
  
  return true;
}//-


bool
JMConstraint::get_JMconsistent_rigid_bodies (
  const vector<vector<Linked_Anchor> >& linked_anchors,
  const vector<uint>& linked_anchors_size) 
{  
  string dbg = "JMConstraint::() - get_JMconsistent_rigid_bodies ";

  for(uint k=0; k<domains_filtered.size(); k++)
    domains_filtered[k].reset();

  int chain_front_aa = _front_anchor.associated_variable_idx();
  int chain_end_aa   = _end_anchor.associated_variable_idx();
  int hook_level = linked_anchors.size() - 1;
  VariableFragment* ref_frag_var = &(g_logicvars->var_fragment_list[chain_end_aa]);
  Bitset reachable_elements (ref_frag_var->domain_size());
  set <Linked_Anchor*> alive_linked_anchors;
  set <const Linked_Anchor*> hooks; // to handle TABLE CONSTRAINT
  set <const Linked_Anchor*>::iterator hooks_it; // to handle TABLE CONSTRAINT

#ifdef JM_DBG
  cout << dbg << "linked anchor["<< hook_level<<"] size: "
       << linked_anchors_size[hook_level] << endl;
#endif

  // Retrieve all the anchors involved in the last clustering level, which 
  // satisfy the JM constraints (position and orientations) at the terminal 
  // flexible sequence hook.
  for (uint c_idx = 0; c_idx < linked_anchors_size[hook_level]; c_idx++) {
    if (!linked_anchors[hook_level][c_idx].is_valid) continue; // for sinergy check

    // TABLE Constraint extension
    hooks.insert (&linked_anchors[hook_level][c_idx]);

    const Anchor* current_end_anchor = 
      &(linked_anchors[hook_level][c_idx].representative);

    // Test if current domain was already set and cluster parents
    // already retrieved
    if (reachable_elements.test (current_end_anchor->associated_fragment_idx())) {
      if (linked_anchors[hook_level][c_idx].parent) {
	alive_linked_anchors.insert (linked_anchors[hook_level][c_idx].parent);
      }
      continue;
    }
    
    // check distances
    if (ths_anchor_distance > 0)
    {
      //if ( ! Math::in_range( current_end_anchor->get_C(), _end_anchor.get_C(), ths_anchor_distance) ) {
	if (! current_end_anchor->is_within_bounds_of (_end_anchor, ths_anchor_distance)) {
	g_statistics->incr_propagation_failures (__c_end_anchor_distance);
	continue;
      }
    }
    g_statistics->incr_propagation_successes (__c_end_anchor_distance);

    // check orientation
    if (ths_anchor_orientation > 0)
      if (!current_end_anchor->is_within_orientation_of (_end_anchor, ths_anchor_orientation)) {
	g_statistics->incr_propagation_failures (__c_end_anchor_orientation);
	continue;
      }
    g_statistics->incr_propagation_successes (__c_end_anchor_orientation);
    
    // set element as reachable
    reachable_elements.set (current_end_anchor->associated_fragment_idx());

    // Retrieve the linked_anchors form the previous level for which there exists
    // a path connecting their anchor representatives with the current anchor
    if (linked_anchors[hook_level][c_idx].parent) {
      alive_linked_anchors.insert (linked_anchors[hook_level][c_idx].parent);
    }
  }
  domains_filtered[hook_level] = reachable_elements.filp();
  //-
#ifdef JM_DBG  
  cout << dbg << "domains filtered of " << hook_level << " updated\n";
  cout << dbg << "HOOKS at last level: " << hooks.size() << endl;
  cout << dbg << "Alive linked_anchors at last level: " << alive_linked_anchors.size() << endl;
#endif
 
  /* 
   * Adaptation for the TABLE CONSTRAINT
   * Retrieve each single path backing from the sink to the source.
   */
#ifdef JM_DBG 
  cout << dbg << "TABLE RESET\n";
#endif
  table->reset();
  hooks_it = hooks.begin();
  for (int row=0; hooks_it != hooks.end(); ++hooks_it, row++) {
    int col = table->get_ncols();
    Anchor* anchor = const_cast<Anchor*> (&(*hooks_it)->representative);
    Linked_Anchor* parent_link = (*hooks_it)->parent;
    //    cout << "row: " << row << ": ";
    while (anchor != NULL && col > 0) {
      table->set (row, --col, anchor->associated_fragment_idx());  
      //      cout << anchor->of_fragment()->get_id() << " -> ";
      anchor = const_cast<Anchor*> (&parent_link->representative);
      if (parent_link) parent_link = parent_link->parent;
    }
    //    cout << endl;
  }
  // Sort Table
  table->radix_sort();
#ifdef JM_PREPROCESS
  table_done = true;
#endif
  //-

#ifdef JM_DBG
  getchar();
#endif

  /*
   * Retrieve the alive linked_anchors at each level, backing from the end of the 
   * flexible chain to the start position.
   */
  hook_level -= 1; 
  for (; hook_level >= 0; hook_level--) {
    ref_frag_var = &(g_logicvars->var_fragment_list[chain_front_aa + hook_level ]); 
    Bitset reachable_elements (ref_frag_var->domain_size());
						      
    set<Linked_Anchor*> awaken_linked_anchors;
    set<Linked_Anchor*>::iterator c_it;
    
    for (c_it = alive_linked_anchors.begin(); c_it != alive_linked_anchors.end(); ++c_it) {
      
      const Anchor* anchor = &(*c_it)->representative;
      reachable_elements.set (anchor->associated_fragment_idx()); 
      if (hook_level > 0 && (*c_it)->parent) {
	awaken_linked_anchors.insert((*c_it)->parent);
      }
    }
    domains_filtered[hook_level] = reachable_elements.filp();
    alive_linked_anchors.swap (awaken_linked_anchors);
  }
  //-
 

#ifdef STATISTICS
  // Compute Filtered search space
  // Hack! Create a separate function that we can call prior aborting the search
  g_statistics->set_timer (t_statistics);  
  long double filtered = 0.0; 

  // Count all the domains filtered up the leaves
  for (uint i = 0; i < domains_filtered.size(); i++) {
    filtered = domains_filtered[i].count() * 
      g_statistics->get_loop_search_space (i + 1);
    g_statistics->incr_filtered_search_space (filtered);
    }
  g_statistics->stopwatch (t_statistics);
#endif

  g_statistics->stopwatch (t_jm);        

  return true;
}
//-


/*
 * Note: Here only the last set of clusters is sufficient for the purpose of the
 * algorithm!!! not need to carry the whole set of clusters
 * Note: This version is affected by the macro CLUSTER_REPRENTATIVE_ONLY declared in 
 * k_medoids.cc to increase the filtering power, possibily affetting the 
 * end-anchor reachibility. We force the selection of the anchor representative
 * only, rather then the whole set of anchor reachable inside this cluster.
 * (only onen element in the following set)
 */
// void
// JMConstraint::get_JMconsistent_rigid_bodies 
//   (vector<Bitset>& domains_filtered,
//    vector<vector<Cluster> >& clusters_L2R, 
//    vector<vector<Cluster> >& clusters_R2L) const {
  
//   string dbg = "JMConstraint::() - get_JMconsistent_rigid_bodies ";
  
//   // chain middle will be by the intersection of the las level of 
//   // clusters_L2R (end anchors) with the first level -- actually the
//   // last in my convention -- of the clusters_R2L (front_anchors)
//   int chain_front_aa = _front_anchor.associated_variable_idx();  
//   int chain_end_aa   = _end_anchor.associated_variable_idx();
//   int chain_middle_aa = (chain_front_aa + chain_end_aa) / 2;
//   uint flexible_chain_len = chain_end_aa - chain_front_aa + 1;
  
//   domains_filtered.resize (flexible_chain_len);
//   int hook_level_L2R = clusters_L2R.size() - 1;
//   int hook_level_R2L = clusters_R2L.size() - 1;
//   set <Cluster*> alive_clusters_L2R;
//   set <Cluster*> alive_clusters_R2L;
  
//   VariableFragment* ref_var_fragL2R = 
//     &(g_logicvars->var_fragment_list[chain_middle_aa]);
//   VariableFragment* ref_var_fragR2L =
//     &(g_logicvars->var_fragment_list[chain_middle_aa + 1]);

//   Bitset reachable_elements_L2R (ref_var_fragL2R->domain_size());
//   Bitset reachable_elements_R2L (ref_var_fragR2L->domain_size());

//   // Sort mid-level anchor vectors to avoid O(n^2) comparisons.
//   // ERR-02: Be careful sort will modify the positions of the 
//   // elements - check pointed structures
//   std::sort (clusters_L2R[hook_level_L2R].begin(), 
// 	     clusters_L2R[hook_level_L2R].end(), 
//   	     Anchor_Nx_compare());
//   std::sort (clusters_R2L[hook_level_R2L].begin(),
//    	     clusters_R2L[hook_level_R2L].end(),
// 	     Anchor_Nx_compare());
 
//   /*
//    * Retrieve all the anchors involved in the middle clustering level, which 
//    * satisfy the JM constraints (position and orientations) at the terminal 
//    * flexible sequence hook.
//    */
//   bool next_start_Aj_set = false;
//   uint next_start_Aj = 0;
//   for (uint Ai = 0; Ai < clusters_L2R[hook_level_L2R].size(); Ai++) {
     
//     const Cluster* cluster_L2R = &(clusters_L2R[hook_level_L2R][Ai]);
//     const Anchor* end_anchor = const_cast<Cluster*>(cluster_L2R)->representative();
    
//     bool hook_was_found = false;	  
//     // Test if current domain was already set and cluster parents already retrieved
//     if (reachable_elements_L2R.test (end_anchor->associated_fragment_idx())) {
//       for (uint p=0; p < cluster_L2R->numof_parents(); p++) {
// 	// map::insert and map::find are both logaritmic in the size of the container,
// 	// therefore we simply force an insertion here. 
// 	alive_clusters_L2R.insert (cluster_L2R->parent(p));
//       }
//       continue;
//     }

//     next_start_Aj_set = false;
//     /*
//      * Test current end-anchor of the middle clustering level done with direction 
//      * Left to Right against any of the front-anchors of the middle clustering level
//      * performed from Right to Left.
//      */
//     for (uint Aj = next_start_Aj; Aj < clusters_R2L[hook_level_R2L].size(); Aj++) {

//       const Cluster* cluster_R2L = &(clusters_R2L[hook_level_R2L][Aj]);
//       const Anchor* front_anchor = const_cast<Cluster*>(cluster_R2L)->representative();
      
//       /* 
//        * Clusters to be checked are the one satisfying the 
//        * end_anchor.x >= front_anchor.x - eps /\ end_anchor.x <= front_anchor.x + eps
//        * Moreover, since the end-anchors and the front-anchors are sorted 
//        * w.r.t. x-coordinate value, if 
//        * end_anchor_i.x < front_anchor_j.x - eps  =>  end_anchor_{i+1}.x < front_anchor_j.x 
//        */
//       real epsilon = kmeans_clustering.get_radius();
//       if (end_anchor->get_N()[0] > front_anchor->get_N()[0] + epsilon) { continue;}
//       if (end_anchor->get_N()[0] < front_anchor->get_N()[0] - epsilon) { break;}
      
//       next_start_Aj = next_start_Aj_set ? next_start_Aj : Aj;
//       next_start_Aj_set = true;
//       //-

//       // Test if current domain was already set and cluster parents already retrieved
//       if (reachable_elements_R2L.test (front_anchor->associated_fragment_idx())) {
// 	for (uint p=0; p < cluster_R2L->numof_parents(); p++) {
// 	  // map::insert and map::find are both logaritmic in the size of the container,
// 	  // therefore we simply force an insertion here. 
// 	  alive_clusters_R2L.insert (cluster_R2L->parent(p));
// 	}
// 	continue;
//       }
      
//       // check distances
//       epsilon = kmeans_clustering.get_radius();
//       if (! end_anchor->is_within_distance_of (*front_anchor, epsilon)) {
// 	g_statistics->incr_propagation_failures (__c_end_anchor_distance);
// 	continue;
//       }
//       g_statistics->incr_propagation_successes (__c_end_anchor_distance);
      
//       // check orientation
//       epsilon = kmeans_clustering.get_beta();
//       if (! end_anchor->is_within_orientation_of (*front_anchor, epsilon)) {
// 	g_statistics->incr_propagation_failures (__c_end_anchor_orientation);
// 	continue;
//       }
//       g_statistics->incr_propagation_successes (__c_end_anchor_orientation);
      
//       hook_was_found = true;
//       // Retrieve the element of the variable domain associated to this front anchor       
//       reachable_elements_R2L.set (front_anchor->associated_fragment_idx());
//       // Retrieve the clusters form the previous level for which there exists
//       // a path connecting their anchor representatives with the current anchor
//       for (uint p = 0; p < cluster_R2L->numof_parents (); p++) {
// 	alive_clusters_R2L.insert (cluster_R2L->parent (p));
//       }
      
//     }// Aj
    
//     // If at least one compatile front anchor was found, the corresponding end-anchor
//     // is marked and its parents retrieved.
//     if (hook_was_found) {
//       reachable_elements_L2R.set (end_anchor->associated_fragment_idx());
//       for (uint p = 0; p < cluster_L2R->numof_parents (); p++){
// 	alive_clusters_L2R.insert (cluster_L2R->parent (p));   
//       }
//     }

      
//   }// Ai

//   domains_filtered[hook_level_L2R] = reachable_elements_L2R.filp();
//   domains_filtered[(flexible_chain_len-1)-hook_level_R2L] = reachable_elements_R2L.filp();
//   //-

//   /*
//    * Retrieve the alive clusters at each level, backing from the end of the 
//    * flexible chain to the start position.
//    */
//   hook_level_L2R -= 1;
//   for (; hook_level_L2R >= 0; hook_level_L2R--) {
//     ref_var_fragL2R = &(g_logicvars->var_fragment_list[chain_front_aa + hook_level_L2R]);
//     Bitset reachable_elements (ref_var_fragL2R->domain_size());
//     set <Cluster*> awaken_clusters;
//     set<Cluster*>::iterator c_it;
//     for (c_it = alive_clusters_L2R.begin(); c_it != alive_clusters_L2R.end(); ++c_it) {

//       set<Anchor_in_cluster>::iterator a_it;
//       for (a_it = (*c_it)->begin(); a_it != (*c_it)->end(); ++a_it) {      
// 	const Anchor* anchor = &((*a_it).first);
//        	reachable_elements.set (anchor->associated_fragment_idx()); 
//       }
//       for (uint p=0; p < (*c_it)->numof_parents(); p++) {
// 	awaken_clusters.insert ((*c_it)->parent(p));
//       }
//     }
//     domains_filtered[hook_level_L2R] = reachable_elements.filp();
//     alive_clusters_L2R.swap (awaken_clusters);
//   }
//   //-

//   hook_level_R2L -= 1;
//   for (; hook_level_R2L >= 0; hook_level_R2L--) {
//     ref_var_fragR2L = &(g_logicvars->var_fragment_list[chain_end_aa - hook_level_R2L]);
//     Bitset reachable_elements (ref_var_fragR2L->domain_size());
//     set <Cluster*> awaken_clusters;
//     set<Cluster*>::iterator c_it;
//     for (c_it = alive_clusters_R2L.begin(); c_it != alive_clusters_R2L.end(); ++c_it) {
//       set<Anchor_in_cluster>::iterator a_it;
//       for (a_it = (*c_it)->begin(); a_it != (*c_it)->end(); ++a_it) {      
// 	const Anchor* anchor = &((*a_it).first);
//        	reachable_elements.set (anchor->associated_fragment_idx()); 
//       }
//       for (uint p=0; p < (*c_it)->numof_parents(); p++) {
// 	awaken_clusters.insert ((*c_it)->parent(p));
//       }
//     }
//     domains_filtered[(flexible_chain_len-1) - hook_level_R2L] = reachable_elements.filp();
//     alive_clusters_R2L.swap (awaken_clusters);
//   }
//   //-

// // #ifdef STATISTICS
// //   // Compute Filtered search space
// //   // Hack! Create a separate function that we can call prior aborting the search
// //   g_statistics->set_timer (t_statistics);  
// //   long double filtered = 0.0; 

// //   // Count only what we have filtered in this domain,
// //   if (flexible_chain_len > iterations_to_ignore + 1) {
// //     filtered = domains_filtered[0].count() *
// //       g_statistics->get_loop_search_space (1);
// //     g_statistics->incr_filtered_search_space (filtered);
// //   }
// //   else {
// //     // Count all the domains filtered up the leaves
// //     for (uint i = 0; i < domains_filtered.size(); i++) {
// //       filtered = domains_filtered[i].count() * 
// // 	g_statistics->get_loop_search_space (i + 1);
// //       g_statistics->incr_filtered_search_space (filtered);
// //     }
// //   }
// //   g_statistics->stopwatch (t_statistics);
// // #endif
  
//   g_statistics->stopwatch (t_filtering); 
// }
//-
// 

void 
JMConstraint::dump_mem () const {
  cout << "==== MANAGING MEM STRUCTURES ===\n";
  for (int i=0; i<clusters_left_to_right.size(); i++)
    cout << "  clusters_L_to_R [" << i << "]: " 
	 << clusters_left_to_right[i].size() << " / " 
	 << clusters_left_to_right_size[i] << endl;
  for (int i=0; i<clusters_right_to_left.size(); i++)
    cout << "  clusters_R_to_L [" << i << "]: " 
	 << clusters_right_to_left[i].size() << " / " 
	 << clusters_right_to_left_size[i] << endl;
 
  cout << "  size arrays L2R: " << clusters_left_to_right_size.size() << endl
       << "  size arrays R2L: " << clusters_right_to_left_size.size() << endl;
  cout << "  bitset: " << domains_filtered.size() << endl;
  cout << "=================================\n";

}

void 
JMConstraint::dump(bool all) {
  std::cout << "JM constraint (w_" << get_weight()  << ") "
	    << " - [N-O-C]_" 
	    <<  _front_anchor.associated_variable_idx();
  if (_front_anchor_is_fixed && _end_anchor_is_fixed) 
    std::cout << " -->--<-- ";
  else if (_front_anchor_is_fixed) 
    std::cout << " -->-->-- ";
  else if (_end_anchor_is_fixed) 
    std::cout << " --<--<-- ";
  else std::cout <<  "--?--?-- ";
  std::cout << "[N-O-C]_"
	    << _end_anchor.associated_variable_idx() 
	    << std::endl;
}
//-
