#include "supervisor.h"
#include "logic_variables.h"
#include "structure_agent.h"
#include "coordinator_agent.h"
#include "utilities.h"

//#define SUPERVISOR_SEARCH_DEBUG
//#define SUPERVISOR_SEARCH_DEBUG_TESTING_CODE

using namespace std;

bool sort_priority ( std::pair < int , MasAgent* > p1, std::pair < int , MasAgent* > p2) {
  if ( p1.first == p2.first ) {
    return ( p1.second->get_bounds ( 0 ) < p2.second->get_bounds ( 0 ) );
  }
  else {
    return ( p1.first < p2.first );
  }
}//sort_priority

Supervisor::Supervisor () :
_n_mas_agents ( 0 ),   
_dbg          ( "#log: Supervisor - " ),
_agt_type     ( supervisor ) {
  _current_solution = (real*) malloc ( gh_params.n_points * sizeof (real) );
  memcpy ( _current_solution, g_logicvars.cp_structure, gh_params.n_points * sizeof (real) );
  create_workers ();
  set_agents ();
}//-

Supervisor::~Supervisor() {
  for ( std::map< int, std::pair < MasAgent*, WorkerAgent* > >::iterator it=_wrk_agents.begin();
       it!=_wrk_agents.end(); ++it) {
    delete it->second.second;
  }
  for ( int i = 0; i < _n_mas_agents; i++) {
    delete _mas_agents[ i ].second;
  }
  
  _wrk_agents.clear();
  _mas_agents.clear();
  
  free ( _current_solution );
}//-

void
Supervisor::create_workers () {
  Utilities::print_debug ( _dbg, "Creating WORKER agents..." );
  for (int i = 0; i < gh_params.n_res; i++) {
    WorkerAgent* wrk_agt = new WorkerAgent ( g_logicvars.cp_variables[ i ] );
    _wrk_agents[ g_logicvars.cp_variables[ i ]->get_id() ] = make_pair( (MasAgent *) NULL, wrk_agt );
  }
}//create_workers

void 
Supervisor::set_agents () {
  Utilities::print_debug ( _dbg, "Setting MAS agents..." );
  // @note:
  // For each V_id associated to the agent we add the corresponding worker,
  // and we set the current MasAgent as parent of the WorkerAgent.
  for (int i = 0; i < gh_params.mas_des.size(); i++) {
    if ( gh_params.mas_des[ i ].agt_type == coordinator ) {
      CoordinatorAgent* cr_agt = new CoordinatorAgent( gh_params.mas_des[ i ], gh_params.n_res );
      for ( int ii = 0; ii < cr_agt->var_list_size(); ii++ ) {
        if ( (ii+1 < cr_agt->var_list_size()) &&
             (abs ( cr_agt->get_var_id ( ii ) -  cr_agt->get_var_id ( ii+1 ) ) > 1) ) {
          /// Add the residue associated to the previous structure agent in order to be
          /// more flexible during folding
          /// Add the residue associated to the previous structure agent in order to be
          /// more flexible during folding
          /*
          cr_agt->add_worker( _wrk_agents[ cr_agt->get_var_id ( ii )  + 1 ].second );
          cr_agt->add_worker( _wrk_agents[ cr_agt->get_var_id ( ii+1 ) - 1 ].second );
           */
        }
        
        /// Skip first and last amino acid (tails)
        if ( (cr_agt->get_var_id ( ii ) == 0) ||
             (cr_agt->get_var_id ( ii ) == (gh_params.n_res - 1)) ) {
          continue;
        }
        cr_agt->add_worker( _wrk_agents[ cr_agt->get_var_id ( ii ) ].second );
        _wrk_agents[ cr_agt->get_var_id ( ii ) ].first =  (MasAgent *) cr_agt;
      }
      // Add the MasAgent together with its priority in the queue
      _mas_agents.push_back ( make_pair ( gh_params.mas_des[ i ].priority, (MasAgent*) cr_agt ) );
    }
    else if ( gh_params.mas_des[ i ].agt_type == structure ) {
      StructureAgent* st_agt = new StructureAgent( gh_params.mas_des[ i ], gh_params.n_res );
      for ( int ii = 0; ii < st_agt->var_list_size(); ii++ ) {
        st_agt->add_worker( _wrk_agents[ st_agt->get_var_id ( ii ) ].second );
        _wrk_agents[ st_agt->get_var_id ( ii ) ].first =  (MasAgent *) st_agt;
      }
      // Add the MasAgent together with its priority in the queue
      _mas_agents.push_back ( make_pair ( gh_params.mas_des[ i ].priority, (MasAgent*) st_agt ) );
    }
  }
  // Sort agents based on priority
  std::sort ( _mas_agents.begin(), _mas_agents.end(), sort_priority );
  _n_mas_agents = _mas_agents.size();
}//set_agents

void
Supervisor::search() {
  Utilities::print_debug ( _dbg, "Starting search..." );

  int agt_counter=0;
  while( agt_counter < _n_mas_agents ) {
    for ( int i = 0; i < _n_mas_agents; i++) {
      
      /// Skip agents without time or jobs
      if ( _mas_agents[ i ].second->done() ) { agt_counter++; continue; }
      /// Choose between agents
      if ( _mas_agents[ i ].second->get_agt_type() == coordinator ) {
        /// Coordinator agent: search
        /// Set current solution as global current solution
        g_logicvars.set_point_variables ( _current_solution );
        /// Set the new (partially) folded status
        _mas_agents[ i ].second->set_current_status ( g_logicvars.cp_structure );
        _mas_agents[ i ].second->search ();
        /// Prepare structure for next coordinator agent
        if ( gh_params.n_coordinators > 1 &&
             ((_mas_agents[ i ].second->get_scope_start() > 0) ||
              (_mas_agents[ i ].second->get_scope_end()   < (gh_params.n_res-1))) ) {
          g_logicvars.set_interval_point_variables ( _mas_agents[ i ].second->get_current_status (),
                                                     _mas_agents[ i ].second->get_bb_start (),
                                                     _mas_agents[ i ].second->get_bb_end (),
                                                     _current_solution );
        }
        
        g_logicvars.set_point_variables ( _mas_agents[ i ].second->get_current_status() );
        
#ifdef SUPERVISOR_SEARCH_DEBUG
        cout << "Search on:\n";
        _mas_agents[ i ].second->dump();
        _mas_agents[ i ].second->end_search();
#endif
        
#ifdef SUPERVISOR_SEARCH_DEBUG_TESTING_CODE
        /// Print solution
        g_logicvars.set_point_variables ( _mas_agents[ i ].second->get_current_status() );
        g_logicvars.print_point_variables();
#endif
      }
      else {
        /// Structure agent: search
        _mas_agents[ i ].second->set_current_status ( g_logicvars.cp_structure );
        _mas_agents[ i ].second->search();
         
        g_logicvars.set_interval_point_variables ( _mas_agents[ i ].second->get_current_status (),
                                                   _mas_agents[ i ].second->get_bb_start (),
                                                   _mas_agents[ i ].second->get_bb_end (),
                                                   _current_solution );
        
#ifdef SUPERVISOR_SEARCH_DEBUG
        cout << "Search on:\n";
        _mas_agents[ i ].second->dump();
        _mas_agents[ i ].second->end_search();
        g_logicvars.set_point_variables ( _mas_agents[ i ].second->get_current_status() );
        g_logicvars.print_point_variables();
#endif
      }
      
    }//i
  }
  
  /// Print solution
  g_logicvars.print_point_variables();
  
  Utilities::print_debug ( _dbg, "End search" );
}//search

void
Supervisor::clear_agents() {
  for ( std::map< int, std::pair < MasAgent*, WorkerAgent* > >::iterator it=_wrk_agents.begin();
       it!=_wrk_agents.end(); ++it) {
    delete it->second.second;
  }
  for ( int i = 0; i < _n_mas_agents; i++) {
    delete _mas_agents[ i ].second;
  }
  
  _wrk_agents.clear();
  _mas_agents.clear();
}//clear_agents

void
Supervisor::dump() {
  cout << "SUPERVISOR on MAS-AGENTS:\n";
  for ( int i = 0; i < _mas_agents.size(); i++)
    _mas_agents[ i ].second->dump();
}//dump




/*
 _quantum = 5;
 cout << _dbg << "Search (q: " << _quantum << ")\n";
 
 srand (time(NULL));
 int iRand;
 do {
 iRand = rand() % 8 + 1;
 timeval time_stats;
 double time_start, total_time;
 gettimeofday(&time_stats, NULL);
 time_start = time_stats.tv_sec + (time_stats.tv_usec/1000000.0);
 
 sleep ( rand() % 5 + 1 );
 //search_engine->search();
 
 gettimeofday(&time_stats, NULL);
 total_time = time_stats.tv_sec + (time_stats.tv_usec/1000000.0) - time_start;
 cout << _dbg << "Used time: " << total_time << endl;
 } while ( iRand < _quantum );
 
 int iRand_exit = rand() % 10 + 1;
 if (iRand_exit >= 5) {
 _end_search = true;
 }
 else
 _end_search = false;
 */
