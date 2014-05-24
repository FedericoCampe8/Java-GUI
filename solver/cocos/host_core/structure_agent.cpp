#include "structure_agent.h"
#include "search_engine.h"
#include "icm.h"
#include "worker_agent.h"
#include "logic_variables.h"
#include "utilities.h"

//#define STR_AGT_SEARCH_DEBUG

using namespace std;

StructureAgent::StructureAgent ( MasAgentDes description, int prot_len ) :
 MasAgent ( description, prot_len ) {
   ostringstream convert;
   convert << _id; 
   _dbg = "#log: Structure_Agent_" + convert.str() + " - ";
   _energy_weights[ 0 ] = gh_params.str_weights[ 0 ];
   _energy_weights[ 1 ] = gh_params.str_weights[ 1 ];
   _energy_weights[ 2 ] = gh_params.str_weights[ 2 ];
   _search_engine = (SearchEngine*) new ICM ( this );
   
   ostringstream scope1, scope2, len_scope;
   scope1 << _scope.first;
   scope2 << _scope.second;
   len_scope << _scope.second - _scope.first + 1;
   Utilities::print_debug ( _dbg, "Created on ["  + scope1.str() + ", " +  scope2.str() + "] L: " +
                            len_scope.str() );
}//-
 
StructureAgent::~StructureAgent () {
}//-

void
StructureAgent::search () {
  Utilities::print_debug ( "*----------------*" );
  Utilities::print_debug ( _dbg, "Search" );
  
  ICM* engine = (ICM*) _search_engine;
  
#ifdef TIME_STATS
  timeval time_stats;
  double time_start, total_time;
  gettimeofday(&time_stats, NULL);
  time_start = time_stats.tv_sec + (time_stats.tv_usec/1000000.0);
#endif
  
  search_alloc ();
  search_init ();
  
  engine->reset ();
  do { 
#ifdef STR_AGT_SEARCH_DEBUG
    static int n_iteration = 0;
    cout << _dbg << "Iteration n_" << ++n_iteration << "\n";
#endif
    
    engine->reset_iteration ();
    engine->search ();
    
  } while ( engine->is_changed() );
  
  /// Set global energy value and structure
  gh_params.minimum_energy = engine->get_local_minimum();
  memcpy ( _current_status, gd_params.curr_str,
          _n_points * sizeof(real) );
  
  /// Free resources and exit
  search_free ();
  end_search ();
  
#ifdef STR_AGT_SEARCH_DEBUG
  g_logicvars.set_point_variables ( _current_status );
  g_logicvars.print_point_variables();
  getchar();
#endif
  
  if ( gh_params.verbose ) {
    cout << _dbg << "Found a minimum:\n";
    cout << "\t - Energy:" << engine->get_local_minimum() << endl;
#ifdef TIME_STATS
    gettimeofday(&time_stats, NULL);
    total_time = time_stats.tv_sec + (time_stats.tv_usec/1000000.0) - time_start;
    cout << "\t - time: " << total_time << " sec.\n";
#endif
    
  }
  
  Utilities::print_debug ( _dbg, "End search" );
  Utilities::print_debug ( "*----------------*" );
}//search

void
StructureAgent::dump () {
  cout << "STRUCTURE Agent_" << _id << " (type " << _agt_type << "):" << endl;
  cout << "SS: " << Utilities::cv_string_to_str_type ( _sec_str_type ) << " P: " <<
  _priority << " Q: " << _quantum << endl;
  cout << "Atom range: [" << _atoms_bds.first << ", " << _atoms_bds.second << "] Scope: [" <<
  _scope.first << ", " << _scope.second << "]\n";
  cout << "AA list:\n[";
  for (int i = 0; i < _vars_list.size()-1; i++)
    cout << _vars_list[i] << ", ";
  cout << _vars_list[_vars_list.size()-1] << "]\n";
}//dump

