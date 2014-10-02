#include "coordinator_agent.h"
#include "search_engine.h"
#include "gibbs.h"
#include "montecarlo.h"
#include "icm.h"
#include "worker_agent.h"
#include "logic_variables.h"
#include "utilities.h"

using namespace std;

CoordinatorAgent::CoordinatorAgent ( MasAgentDes description, int prot_len ) :
 MasAgent ( description, prot_len ) {
   ostringstream convert;
   convert << _id;
   _dbg = "#log: Coordinator_Agent_" + convert.str() + " - ";
   _energy_weights[ 0 ] = gh_params.crd_weights[ 0 ];
   _energy_weights[ 1 ] = gh_params.crd_weights[ 1 ];
   _energy_weights[ 2 ] = gh_params.crd_weights[ 2 ];
   /// Set search strategy
   switch ( _search_strategy ) {
     case icm:
       _search_engine = (SearchEngine*) new ICM ( this );
       break;
     case gibbs:
       _search_engine = (SearchEngine*) new GIBBS ( this );
       break;
     case montecarlo:
       _search_engine = (SearchEngine*) new MONTECARLO ( this );
       break;
     default:
       _search_engine = (SearchEngine*) new MONTECARLO ( this );
       break;
   }
   
   string print_scope = "Created on\n[";
   for ( int i = 0; i < _vars_list.size()-1; i++ ) {
     ostringstream scope;
     scope << _vars_list[i];
     if ((i > 0) && (i % 10 == 0))
       print_scope = print_scope + scope.str() + ",\n ";
     else
       print_scope = print_scope + scope.str() + ", ";
   }
   ostringstream scope;
   scope << _vars_list[_vars_list.size()-1];
   print_scope = print_scope + scope.str() + "] L: ";
   
   ostringstream scope2;
   scope2 << _vars_list.size();
   print_scope = print_scope + scope2.str(); //+ "\n";
   Utilities::print_debug ( _dbg, print_scope );
}//-

CoordinatorAgent::~CoordinatorAgent () {
}//-

void
CoordinatorAgent::search () {
  Utilities::print_debug ( "*----------------*" );
  Utilities::print_debug ( _dbg, "Search" );
  
  SearchEngine* engine;
  switch ( _search_strategy ) {
    case icm:
      engine = (ICM*) _search_engine;
      break;
    case gibbs:
      engine = (GIBBS*) _search_engine;
      break;
    case montecarlo:
      engine = (MONTECARLO*) _search_engine;
      break;
    default:
      engine = (MONTECARLO*) _search_engine;
      break;
  }

  
#ifdef TIME_STATS
  timeval time_stats;
  double time_start, total_time;
  gettimeofday(&time_stats, NULL);
  time_start = time_stats.tv_sec + (time_stats.tv_usec/1000000.0);
#endif
  
  /// Alloc structures
  search_alloc ();
  search_init ();
  /// Sampling
  engine->reset ();
  engine->search ();
  
  HANDLE_ERROR( cudaMemcpy( _current_status, gd_params.curr_str,
                            _n_points * sizeof(real), cudaMemcpyDeviceToHost ) );
  
  /// Set global energy value
  real updated_energy = engine->get_local_minimum();
  if ( updated_energy < MAX_ENERGY ) {
    gh_params.minimum_energy = engine->get_local_minimum();
  }
  /// Free structures
  search_free ();
  end_search ();
  
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
CoordinatorAgent::dump () {
  cout << "COORDINATOR Agent_" << _id << " (type " << _agt_type << "):" << endl;
  cout << "SS: " << Utilities::cv_string_to_str_type ( _sec_str_type ) << " P: " << _priority <<
  " Q: " << _quantum << endl;
  cout << "Atom range: [" << _atoms_bds.first << ", " << _atoms_bds.second << "] Scope: [" <<
  _scope.first << ", " << _scope.second << "]\n";
  cout << "AA list:\n[";
  for (int i = 0; i < _vars_list.size()-1; i++)
    cout << _vars_list[i] << ", ";
  cout << _vars_list[_vars_list.size()-1] << "]\n";
}//dump
