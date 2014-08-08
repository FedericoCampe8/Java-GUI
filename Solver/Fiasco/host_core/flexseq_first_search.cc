#include "flexseq_first_search.h"
#include "globals.h"
#include "logic_variables.h"
#include "variable_point.h"
#include "variable_fragment.h"
#include "utilities.h"
#include "constraint_store.h"
#include "trailstack.h"
#include "rmsd.h"
#include "output.h"
#include "statistics.h"
#include "protein.h"

#include <cassert>
#include <limits>
#include <string>
#include <sstream>
#include <stdlib.h> // for atoi

using namespace std;
using namespace Rmsd;
using namespace Utilities;

//#define WRITE_OUTPUT_PDB_FILE
//#define BIDIRECTIONAL
//#define SEARCH_DBG

FlexSeqFirstSearch::FlexSeqFirstSearch (int argc, char* argv[]) 
  : expl_level (0) {
  Rmsd::alloc();
  height = g_target.get_nres();
  curr_labeling.resize(height);
  next_labeling.resize(height);
}//-

void
FlexSeqFirstSearch::reset() {
  expl_level = 0;
  curr_labeling.clear();
  curr_labeling.resize(height);
  next_labeling.clear();
  next_labeling.resize(height);
}//-

FlexSeqFirstSearch::~FlexSeqFirstSearch () {
  Rmsd::dealloc();
}//-

VariableFragment* 
FlexSeqFirstSearch::variable_selection () {
  string dbg = "FlexSeqFirstSearch::vselect_leftmost() - ";
  VariableFragment* var_selected = NULL;
  // HACK -- do it better use intervals!!!
  for (uint i=0; i<height; i++) {
    int ca = i*4+1;
    if (!g_logicvars->var_point_list[ca].is_ground()) {
      var_selected = &(g_logicvars->var_fragment_list[i]);
      var_selected->set_assembly_direction (LEFT_TO_RIGHT);
      break;
    }
  }
  return var_selected;
}
//-


/*
 * Given a loop with front and end anchors on amino acids F and E, 
 * selects the first non-ground variable which minimizes the amino 
 * acid distance to F or to E.
 */
VariableFragment* 
FlexSeqFirstSearch::select_flex_variable (const Loop* loop) {
  string dbg = "Searchengine::select_flex_variable() - ";
  int front_anchor_ca = get_aaidx_from_bbidx (loop->get_bb_bounds().first + 1, CA);
  int end_anchor_ca = get_aaidx_from_bbidx (loop->get_bb_bounds().second - 2, CA);
  real middle  = (end_anchor_ca + front_anchor_ca) / 2;
  VariableFragment* var_selected = NULL;

  // Front and end anchor variables are assumed to be ground already
  // and there is at least one non-ground variable
  while (g_logicvars->var_fragment_list[front_anchor_ca].is_ground()) {front_anchor_ca ++; }
  while (g_logicvars->var_fragment_list[end_anchor_ca].is_ground()) {end_anchor_ca --; }

  if (Math::abs (front_anchor_ca - middle) >= Math::abs (end_anchor_ca - middle)) {
    assert (front_anchor_ca >= 0);
    var_selected = &g_logicvars->var_fragment_list[front_anchor_ca];
    var_selected->set_assembly_direction (LEFT_TO_RIGHT);
  }
  else {
    assert (front_anchor_ca >= 0);
    var_selected = &g_logicvars->var_fragment_list[end_anchor_ca];
    var_selected->set_assembly_direction (RIGHT_TO_LEFT);
  }
  return var_selected;

}//-


/*
 * Select the first loop for which both end anchors are ground, and
 * there exists at least one non ground variable in between.
 * Note that here we are looking for the variables preceeding the 
 * front and end anchors to be ground.
 */
const Loop* 
FlexSeqFirstSearch::select_flexible_sequence () {
  string dbg = "Searchengine::select_flexible_sequence() - ";
  for (size_t loop = 0; loop < g_target.numof_loops(); loop++) {
    pair<uint, uint> loop_bb = g_target.get_loop(loop)->get_bb_bounds(); // N and O
    int front_anchor_ca = loop_bb.first + 1; 
    int end_anchor_ca   = loop_bb.second - 2;
      
    if (front_anchor_ca-4 >= 0 && g_logicvars->var_point_list[front_anchor_ca-4].is_ground() &&
	end_anchor_ca+4 <= g_target.get_bblen() && g_logicvars->var_point_list[end_anchor_ca+4].is_ground()) {

      for (int ca = front_anchor_ca; ca <= end_anchor_ca; ca +=4 ) { 
	if (!g_logicvars->var_point_list[ca].is_ground()) {
	  return g_target.get_loop (loop);
	}
      }
    }
    
  }// forall loops
  return NULL;
}
//-


// Check if every element of the domain for current variable
// has been explored
bool
FlexSeqFirstSearch::labeling (VariableFragment *v) { 
  string dbg = "FlexSeqFirstSearch::labeling() - ";
  if (v->labeling()) {
    curr_labeling[expl_level] = v->get_label();
    next_labeling[expl_level] = v->get_next_label(); 
    
    // Put the chaged variable_fragment into the TRAILSTACK before modifying it
    g_trailstack.trail_variable_fragment (v, v->domain_info, 
					  (size_t) g_trailstack.size());
    // This will activate propagation+consistency later
    v->set_changed (true);
    g_constraintstore.upd_changed (v);
    return true;
  }

  return false;       
}
//-


void 
FlexSeqFirstSearch::goto_next_level(const VariableFragment *v){
  uint d_idx = v->get_label(); //curr_labeling[expl_level];
  expl_level += v->domain[d_idx].nres();
  // manage bundles
  if (v->is_in_bundle()) {
    uint m_idx = v->domain_info[d_idx].frag_mate_idx;
    uint other_var_idx = v->domain_info[d_idx].frag_mate_info[m_idx].first;
    int  other_d_idx   = v->domain_info[d_idx].frag_mate_info[m_idx].second;
    expl_level += 
      (g_logicvars->var_fragment_list[other_var_idx].domain[other_d_idx].nres()); 
  }
}
//-


void 
FlexSeqFirstSearch::goto_prev_level(const VariableFragment *v) {
  uint d_idx = v->get_label(); //curr_labeling[expl_level];
  expl_level -= v->domain[d_idx].nres();
  // manage bundles
  if (v->is_in_bundle()) {
    uint m_idx = v->domain_info[d_idx].frag_mate_idx;
    uint other_var_idx = v->domain_info[d_idx].frag_mate_info[m_idx].first;
    int  other_d_idx   = v->domain_info[d_idx].frag_mate_info[m_idx].second;
    expl_level -= 
      (g_logicvars->var_fragment_list[other_var_idx].domain[other_d_idx].nres());
  }
}
//-


void
FlexSeqFirstSearch::search () {
  string dbg = "FlexSeqFirstSearch::search() - ";

  size_t trailtop=0, continuation=0, continuation_bundle=0;
  VariableFragment *var = NULL;
  int vidx = 0;
  vector <bool> vdom_explored;
  
  if(expl_level == 0) {
#ifdef VERBOSE
    std::cout << dbg << "FIASCO Search Engine started\n";
#endif
  }

  if(expl_level >= height) {
    // Process the backtrack_trailtop
    if (g_constraintstore.check_cardinality_constraints (backjump_trailtop)) {
      process_solution();
    }
    return;
  }
    
  if (backjump_trailtop == NO_BACKJUMPING) {
    // No Backjump is enforced - proceed normally in the recursive 
    // step of the search procedure.
    trailtop = g_trailstack.size();

#ifdef BIDIRECTIONAL
    const Loop* flex = select_flexible_sequence ();
    if (flex)
      var = select_flex_variable (flex);
    else
#else
      var = variable_selection ();
#endif

    assert (var != NULL);
    vidx = var->get_idx();
    
    // Save the current status/choices -- to restore at  backtrack level!
    // HACK! use static array to improve efficiency
    var->get_domain_explored (vdom_explored);
  
    // Set Current Fragment Choice (in VAR_FRAGMENT domain)
    curr_labeling[expl_level] = next_labeling[expl_level] = 0;
    
    while( labeling(var) ) {
#ifdef SEARCH_DBG 
      std::cout << dbg << "Labeling: V_" << vidx << "(" << var->get_label() << ")\n";
#endif
      if (g_statistics->timeout_searchtime_only() || 
	  g_statistics->timeout()) {
	SearchEngine::abort();
      }
      if (SearchEngine::aborted()) return;
      
#ifdef STATISTICS
      g_statistics->set_timer (t_statistics);
      g_statistics->incr_search_space_explored();
#ifdef VERBOSE
      if (g_statistics->get_search_space_explored() % 1000000 == 0) {
        g_statistics->stopwatch(t_search);
        cout << dbg << (real) g_statistics->get_search_space_explored()/1000000 
	     << " ML of nodes expanded (" 
	     << g_statistics->get_solutions_found() 
	     << " sol. found) in "
	     << g_statistics->get_timer(t_search) << " s. "
	     << " - TABLE time: " 
	     << g_statistics->get_timer(t_table) << " s. "
	     << endl;
      }
#endif
      g_statistics->stopwatch (t_statistics);
#endif
      
      continuation_bundle = 0;
      g_trailstack.reset_continuation();
      
      if(g_constraintstore.propagate(trailtop)) {
#ifdef SEARCH_DBG 
	cout << dbg << "propagation ok\n";
#endif
	continuation_bundle = g_trailstack.get_continuation();
	// Recursive call
	goto_next_level(var);
       	search();
	goto_prev_level(var);
	//-----
      }
      
      // check if next labeling choice will still be inside this bucket
      if (next_labeling[expl_level] == curr_labeling[expl_level]) {
	if (continuation_bundle > 0 && trailtop < continuation_bundle) {
	  continuation = continuation_bundle;
	}
      }
      else {
	continuation = trailtop;
      }
      
      // Backtrack all modifications (except for the fragment choice)
      g_trailstack.backtrack (continuation);
      g_statistics->incr_backtracks();
#ifdef SEARCH_DBG
      std::cout << dbg << "Backtrack!\n";
#endif

      if (continuation == backjump_trailtop) {
	backjump_trailtop = NO_BACKJUMPING;
      }
      else if (backjump_trailtop != NO_BACKJUMPING) {
#ifdef SEARCH_DBG
	std::cout << dbg << "Backjump enforced (stop here)!\n";
#endif
	break;
      }
    }//-end-labeling-step
  }//-end-backjump-check
  else 
    std::cout << dbg << "Backjump enforced!\n";
 
 
  // Restore previous Domain values (after all tries)
  g_logicvars->var_fragment_list[vidx].reset_label();
  g_logicvars->var_fragment_list[vidx].reset_domain();  
  g_logicvars->var_fragment_list[vidx].set_ground(false);
  g_logicvars->var_fragment_list[vidx].set_domain_explored(vdom_explored);
  //-
  return;
}
//-

void
FlexSeqFirstSearch::process_solution() {
  string dbg = "FlexSeqFirstSearch::process_leaf() - ";
  g_statistics->incr_soluions_found();

#ifdef STATISTICS
  g_statistics->set_timer (t_statistics);
#endif
  
  // Compute RMSD
  real rmsd_loop=1000;
  Loop *flexible_chain_to_print = g_target.get_loop("flexible_chain_to_print");
  Loop *flexible_chain = g_target.get_loop("jm_loop");
  rmsd_loop = Rmsd::rmsd_compare
    (flexible_chain->get_bb_bounds().first+1, flexible_chain->get_bb_bounds().second-2);
  g_statistics->set_rmsd( p_loop, rmsd_loop );

  g_statistics->set_best_rmsd(p_loop, rmsd_loop);
    
#ifdef WRITE_OUTPUT_PDB_FILE
  //if (rmsd_loop == g_statistics->get_rmsd(p_loop)) {
  //g_output->store_results();
    g_output->store_results(flexible_chain_to_print->get_bb_bounds().first-28, 
     			    flexible_chain_to_print->get_bb_bounds().second+40, 
     			    rmsd_loop);
    //  }
    if (g_statistics->get_solutions_found() % 10000 == 0) {
      std::cout << "stored sol " << g_statistics->get_solutions_found() << endl;
      g_output->dump();
    }
#endif

#ifdef STATISTICS
#ifdef VERBOSE
  if (g_statistics->get_solutions_found() % 10000 == 0) {
    g_statistics->stopwatch (t_search);
    std::cout << dbg << g_statistics->get_solutions_found() << " Solutions Found."
	      << "[time: " << g_statistics->get_timer (t_search)
	      << " s.] / Best RMSD: "
	      << g_statistics->get_rmsd(p_loop) << endl;
  }
  g_statistics->stopwatch (t_statistics);
#endif
#endif

  if (max_numof_solutions > 0 && 
      g_statistics->get_solutions_found()  >= max_numof_solutions)
    SearchEngine::abort();
}//-


void
FlexSeqFirstSearch::dump_statistics (std::ostream &os) {
  if (SearchEngine::aborted())
    os << "Search aborted, timeout of numof solutions limit reached.";
  else 
    os << "Compleately explored the search space. ";

  std::vector<real> rmsds = g_statistics->get_rmsd_ensemble();
  real avg = 0.0;
  for(int i=0; i<rmsds.size(); i++) avg += rmsds[i];
  avg /= rmsds.size();

  os << "[Numof Ensembles Generated: " 
     << g_statistics->get_solutions_found()
     << " | Seach time: "
     << g_statistics->get_total_timer (t_search)
    //<< " s.| Best (Loop) Rmsd: " << g_statistics->get_rmsd (p_loop) << "A]\n";

     << "s.| Rmsd Best: " << rmsds.front() << " Avg: " << avg << " Worst: " 
     << rmsds.back() << endl;

}//-
