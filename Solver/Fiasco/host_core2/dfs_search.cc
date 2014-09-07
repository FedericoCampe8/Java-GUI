#include "dfs_search.h"
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
#include "energy.h"

#include <cassert>
#include <limits>
#include <string>
#include <sstream>
#include <stdlib.h> // for atoi
#include <cmath>
using namespace std;
using namespace Rmsd;
using namespace Utilities;

#define WRITE_OUTPUT_PDB_FILE
//#define SEARCH_DBG

DepthFirstSearchEngine::DepthFirstSearchEngine (int argc, char* argv[]) 
  : expl_level(0) {
  for (int narg=0; narg < argc; narg++) {
    if (!strcmp ("--ensembles",argv[narg])) {
      SearchEngine::max_numof_solutions = atoi(argv[narg + 1]);
      break;
    }
  }

  Rmsd::alloc();
  height = g_target.get_nres();
  curr_labeling.resize(height);
  next_labeling.resize(height);
}//-


void
DepthFirstSearchEngine::reset() {
  expl_level = 0;
  curr_labeling.clear();
  curr_labeling.resize(height);
  next_labeling.clear();
  next_labeling.resize(height);
}//-


DepthFirstSearchEngine::~DepthFirstSearchEngine () {
  Rmsd::dealloc();
}//-


VariableFragment* 
DepthFirstSearchEngine::variable_selection() {
  string dbg = "DepthFirstSearchEngine::vselect_leftmost() - ";
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
}//-


/* Check if every element of the domain for current variable
 * has been explored
 */
bool
DepthFirstSearchEngine::labeling (VariableFragment *v) { 
  string dbg = "DepthFirstSearchEngine::labeling() - ";
#ifdef SEARCH_DBG
  cout << dbg << "Labeling V_ " << v->get_idx();
#endif
  if (v->labeling()) {
    curr_labeling[expl_level] = v->get_label();
    next_labeling[expl_level] = v->get_next_label(); 
#ifdef SEARCH_DBG
    cout << " curr: " << curr_labeling[expl_level]
	 << " next: " << next_labeling[expl_level]
	 << endl;
#endif
    // Put the chaged variable_fragment into the TRAILSTACK before modifying it
    g_trailstack.trail_variable_fragment (v, v->domain_info, 
					  (size_t) g_trailstack.size());
    // This will activate propagation+consistency later
    v->set_changed (true);
    g_constraintstore.upd_changed (v);
    return true;
  }
#ifdef SEARCH_DBG
  cout << " Cannot label - ret FALSE\n";
#endif

  return false;       
}//-


void 
DepthFirstSearchEngine::goto_next_level(const VariableFragment *v){
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
}//-


void 
DepthFirstSearchEngine::goto_prev_level(const VariableFragment *v) {
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
DepthFirstSearchEngine::search () {
  string dbg = "DepthFirstSearchEngine::depth_first_search() - ";
  
  size_t trailtop=0, continuation=0, continuation_bundle=0;
  VariableFragment *var = NULL;
  int vidx = 0;
  vector <bool> vdom_explored;
  
  if(expl_level == 0) {
    std::cout << dbg << "FIASCO Depth First Search Engine started\n";
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

    var = variable_selection ();

    assert (var != NULL);
    vidx = var->get_idx();
    
    // Save the current status/choices -- to restore at  backtrack level!
    // HACK! use static array to improve efficiency
    var->get_domain_explored (vdom_explored);
  
    // Set Current Fragment Choice (in VAR_FRAGMENT domain)
    curr_labeling[expl_level] = next_labeling[expl_level] = 0;

    while( labeling (var) ) {
#ifdef SEARCH_DBG 
      cout << dbg << "Labeling: V_" << vidx << "(" << var->get_label() << ")\n";
#endif
      if (g_statistics->timeout_searchtime_only() || 
	  g_statistics->timeout()) {
	SearchEngine::abort();
      }
      if (SearchEngine::aborted()) return;
      
#ifdef STATISTICS
      g_statistics->set_timer (t_statistics);
      g_statistics->incr_search_space_explored();
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
      g_statistics->stopwatch (t_statistics);
#endif
      
      continuation_bundle = 0;
      g_trailstack.reset_continuation();
      
      if (g_constraintstore.propagate(trailtop)) {
#ifdef SEARCH_DBG 
	cout << dbg << "propagation ok\n";
#endif
	continuation_bundle = g_trailstack.get_continuation();
	// Recursive call
	goto_next_level (var);
       	search ();
	goto_prev_level (var);
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
DepthFirstSearchEngine::process_solution() {
  string dbg = "DepthFirstSearchEngine::process_leaf() - ";
  g_statistics->incr_soluions_found();

  int nres  = g_target.get_nres();
  
  int n_threads = 32;
  while ( n_threads < nres ) n_threads += 32;
  n_threads = n_threads*2 + 32;
  real energy = get_energy ( g_params.secondary_s_info,
                             g_params.h_distances, g_params.h_angles,
                             g_params.contact_params, g_params.aa_seq,
                             g_params.tors, g_params.tors_corr,
                             8, 22, 7,
                             0, (5 * nres) - 1,
                             nres, 0, nres-1,
                             0, 1, n_threads );
  
#ifdef STATISTICS
  g_statistics->set_timer ( t_statistics );
  // Compute RMSD
  real curr_rmsd = Rmsd::rmsd_compare( 0, g_target.get_bblen() - 1 );
  g_statistics->set_best_rmsd( protein, curr_rmsd );
#endif  

#ifdef WRITE_OUTPUT_PDB_FILE
  /// Store every result
  //g_output->store_results();
  /// Store just the best result
  if ( g_statistics->rmsd_is_improved() ) g_output->store_best_results( -1, energy );
  if ( g_statistics->get_solutions_found() % 10000 == 0 ) {
    g_output->dump();
  }
#endif

#ifdef STATISTICS
  if (g_statistics->get_solutions_found() % 10000 == 0) {
    g_statistics->stopwatch ( t_search );
    std::cout << dbg << g_statistics->get_solutions_found() << " Solutions Found."
	      << "[time: " << g_statistics->get_timer (t_search)
	      << " s.] / Best RMSD: "
	      << g_statistics->get_rmsd(protein) << endl;
  }
  g_statistics->stopwatch (t_statistics);
#endif

  if ( (max_numof_solutions > 0) &&
       (g_statistics->get_solutions_found() >= max_numof_solutions) )
    SearchEngine::abort();
}//-


void
DepthFirstSearchEngine::dump_statistics (std::ostream &os) {
  if (SearchEngine::aborted())
    os << "Search aborted, timeout of numof solutions limit reached.";
  else 
    os << "Compleately explored the search space. ";
  
  os << "[Numof Ensembles Generated: " 
     << g_statistics->get_solutions_found()
     << " | Seach time: "
     << g_statistics->get_total_timer (t_search)
     << " s.| Best Rmsd: " << g_statistics->get_rmsd ( protein ) << "A]\n";
}//-
