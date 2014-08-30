#include "montecarlo.h"
#include "energy.h"
#include "mathematics.h"
#include "propagator.h"
#include "utilities.h"
#include "rmsd_fast.h"
#include "potential_energy.h"

#include "logic_variables.h"
#include "all_distant.h"

//#define MONTECARLO_DEBUG
//#define MONTECARLO_DEBUG_LABELING
//#define MONTECARLO_USE_RMSD

using namespace std;

MONTECARLO::MONTECARLO ( MasAgent* mas_agt ) :
SearchEngine       ( mas_agt ),
_var_selection     ( 0 ),
_level             ( 0 ),
_height            ( 0 ),
_last_wrk_sel      ( -1 ),
_last_idx_sel      ( -1 ),
_iter_counter      ( 0 ),
_max_iterations    ( 4 ),
_n_of_restarts     ( 0 ),
_max_n_restarts    ( 1 ),
_n_sols            ( 0 ),
_temperature       ( 100.0 ),
_decreasing_factor ( 10.0 ),
_exit_asap         ( false ),
_changed           ( false ),
_forced_labeling   ( false ),
_idx_rand_sel      ( NULL ),
_labeled_vars      ( NULL ),
_dbg               ( "#log: MONTECARLO - " ) {
  srand ( time( NULL ) );
  _local_minimum         = MAX_ENERGY;
  _local_current_minimum = MAX_ENERGY;
  _glb_current_minimum   = MAX_ENERGY;
  _curr_best_str = (real*) malloc ( gh_params.n_points * sizeof( real ) );
  _glb_best_str  = (real*) malloc ( gh_params.n_points * sizeof( real ) );
}//-

MONTECARLO::~MONTECARLO() {
  if ( !_idx_rand_sel )
    delete [] _idx_rand_sel;
  if ( !_labeled_vars )
    delete [] _labeled_vars;
  if ( !start_structure )
    free ( start_structure );
  free ( _curr_best_str );
  free ( _glb_best_str  );
}//-

void
MONTECARLO::set_sequential_scanning () {
  _var_selection = 0;
}//set_sequential_scanning

void
MONTECARLO::set_random_scanning () {
  _var_selection = 1;
}//set_random_scanning

bool
MONTECARLO::is_changed () const {
  return _changed;
}//is_changed

int
MONTECARLO::get_n_ground () const {
  return _height;
}//get_n_ground

bool
MONTECARLO::all_ground () const {
  return ( _height >= _n_vars );
}//is_changed

size_t
MONTECARLO::get_n_sols () {
  return _n_sols;
}//get_n_sol

void
MONTECARLO::reset () {
  _level         = 0;
  _n_sols        = 0;
  _iter_counter  = 0;
  _height        = 0;
  _var_selection = 0;
  _n_vars        = _wrks->size();
  _wrks_it       = _wrks->begin();
  _last_wrk_sel  = -1;
  _local_minimum = MAX_ENERGY;
  _changed       = false;
  if ( !_idx_rand_sel ) _idx_rand_sel = new bool[ _n_vars ];
  memset ( _idx_rand_sel, true, _n_vars * sizeof(bool) );
  if ( !_labeled_vars ) {
    _labeled_vars = new bool[ _n_vars ];
  }
  memset ( _labeled_vars, false, _n_vars * sizeof(bool) );
  /// Set Current Structure
  memcpy( gd_params.curr_str, start_structure, gh_params.n_points * sizeof(real) );
}//reset

WorkerAgent*
MONTECARLO::worker_selection () {
  if ( !_var_selection ) { /// Scanning through variables 
    if ( _level++ > 0 ) advance ( _wrks_it,  1 );
    _last_idx_sel = _level-1;
    return _wrks_it->second;
  }
  else {  /// Random selection
    int r_idx = rand() % ( _n_vars - ( _level++ ) );
    while ( !_idx_rand_sel[ r_idx % _n_vars ] ) { r_idx++; }
    _idx_rand_sel[ r_idx ] = false;
    _wrks_it = _wrks->begin();
    advance ( _wrks_it,  r_idx );
    _last_idx_sel = r_idx;
    return _wrks_it->second;
  }
}//worker_selection

void
MONTECARLO::reset_iteration () {
  /// Set ICM default values for a lopp
  _level   = 0;
  _wrks_it = _wrks->begin();
  _changed = false;
  //_local_current_minimum = MAX_ENERGY;
  memset ( _idx_rand_sel, true, _n_vars * sizeof( bool ) );
  /// Reset domains for the new iteration -> backtrack on validity_solutions
  backtrack ();
}//reset_iteration

void
MONTECARLO::search () {
#ifdef MONTECARLO_DEBUG
  cout << _dbg << " Start search...\n";
#endif
  timeval time_stats;
  double time_start, total_time;
  
  /// Set timer
  if ( gh_params.timer >= 0 ) {
    gettimeofday(&time_stats, NULL);
    time_start = time_stats.tv_sec + (time_stats.tv_usec/1000000.0);
  }
  
  /// Loop until no changes happen:
  /// choose the best structure to set at the end of every loop iteration
  int n_iters = 0, cool_down_times = 0;
  bool unleash_temperature = false, try_again = true /*, cool_it_down = true*/;
  real diff_on_sols, param, fractpart, int_part, previous_int_par = 100000000.0;
  real prev_energy = MAX_ENERGY;
  /// Loop trying to improve energy
  do {
    reset_iteration ();
    WorkerAgent* w;
    while ( _level < _n_vars ) {
      backtrack ();
      w = worker_selection ();
      /// Skip last labeled agent
      //if ( _last_wrk_sel == w->get_id() ) continue;
      /// Propagation
      if ( !(w->propagate ( _constraint_store )) ) { continue; }
      /// Select the label
      choose_label( w );
      /// Check timer
      if ( gh_params.timer >= 0 ) {
        gettimeofday( &time_stats, NULL );
        total_time = time_stats.tv_sec + (time_stats.tv_usec/1000000.0) - time_start;
        /// Exit for timeout
        if ( total_time > gh_params.timer ) {
          cout << _dbg << "Stop search - Timeout reached (" <<
          gh_params.timer << " sec.)\n" ;
          try_again = false;
          break;
        }
      }
    }//while
    
    /// On timeout - break
    if ( (!try_again) && (gh_params.timer >= 0) ) break;
    /// If something is changed -> update solution
    update_solution ();
    //cout << "Number of iterations... " << ++n_iters << endl;
#ifdef MONTECARLO_DEBUG
    if ( ((++n_iters) % 10) == 0) {
      cout << _dbg;
      if ( gh_params.follow_rmsd ) {
        printf("MonteCarlo Iteration: %d with RMSD %.8f\n", n_iters, _local_minimum);
      }
      else {
        printf("MonteCarlo Iteration: %d with energy %.8f\n", n_iters, _local_minimum);
      }
    }
    if (n_iters > 1000) {
      try_again = false;
    }
#endif
    
    /// If no changes and not all variables are ground force the labeling on one variable
    if ( (!_changed) && (!all_ground()) ) force_label ();
    if ( (!_exit_asap) && all_ground() && unleash_temperature ) {
#ifdef MONTECARLO_DEBUG
      cout << "heat it up: " << _temperature << " -> " << _temperature + _decreasing_factor << endl;
#endif
      /// Decrease temperature (increase -> less probability to take a worst structure)
      _temperature += _decreasing_factor;
      if ( _temperature >= 100.0 ) { _temperature = 80.0; _n_of_restarts++; /*cout << "increase n. of restarts\n";*/ }
      /// After _max_n_restarts exit asap
      if ( _n_of_restarts > _max_n_restarts ) _temperature = 100.0;
    }
    if ( !_exit_asap ) {
      
      diff_on_sols = fabs( prev_energy - _local_minimum );
      /// Round up values
      param = diff_on_sols * 1000;
      fractpart = modf ( param , &int_part );
      /// Check if some changes happened
      if ( (int_part == previous_int_par) || (int_part == 0) ) {
        if ( ((++_iter_counter) > _max_iterations) && all_ground() ) { try_again = false; /* cout << "exit asap\n"; */ }
        /// Start to decrease temperature
        if ( _temperature >= 100.0  /*&& cool_it_down*/ ) {
          _temperature = 80.0 + ((100.0/_max_iterations) * ((cool_down_times++) - 1));
          //_temperature = 30.0; cool_it_down = false;

        }
#ifdef MONTECARLO_DEBUG
        cout << _iter_counter << " prev==current -> unleash " << _iter_counter << " t: " << _temperature << endl;
#endif
        unleash_temperature = true;
      }
      else {
        if ( _iter_counter ) _iter_counter = 0;
      }
      /// Set values for next round
      prev_energy      = _local_minimum;
      previous_int_par = int_part;
      
    }
    else {
      /// Exit as soon as possible
      /// _changed: false (no changes) /\ (!all_ground()) = false (all_ground() = true)
      try_again = ( _changed || (!all_ground()) );
    }
  } while ( try_again );

  /// Take the best structure among all possible samplings
  if ( _glb_current_minimum < _local_minimum ) {
    if ( gh_params.verbose ) {
      cout << _dbg << "Solution with value: " <<
      _glb_current_minimum << " instead of " <<
      _local_minimum << endl;
    }
    
    _local_minimum = _glb_current_minimum;
    memcpy ( gd_params.curr_str, _glb_best_str,
             gh_params.n_points * sizeof( real ) );
  }

#ifdef MONTECARLO_DEBUG
  cout << _dbg << "End search\n";
#endif
}//search

void
MONTECARLO::force_label () {
#ifdef MONTECARLO_DEBUG
  cout << "FORCE LABELING" << endl;
#endif
  WorkerAgent* w;
  /// Find the first not labeled var
  int select = -1;
  for (int i = 0; i <= _n_vars; i++) {
    if (i == _n_vars) {
      /// All not labeled variables fail propagation
      cout << "BACKTRACK NEEDED!!!" << endl;
      _changed = false;
      _height  = _n_vars;
      return;
    }
    if ( !_labeled_vars[ i ] ) {
      _wrks_it = _wrks->begin();
      advance ( _wrks_it,  i );
      w = _wrks_it->second;
      //_last_idx_sel = i;
      /// Propagate again on _wrks_it
      backtrack ();
      if ( !(w->propagate ( _constraint_store )) ) {
#ifdef MONTECARLO_DEBUG
        cout << "No admissible labelings for " << i << endl;
#endif
        continue;
      }
#ifdef MONTECARLO_DEBUG
      cout << "Found a labeling for " << i << endl;
#endif
      /// Instead of break -> try to force label to all non ground variables
      select = i;
      break;
    }
  }
  /// Find the label corresponding to its minimum
  _local_current_minimum = MAX_ENERGY;
  _local_minimum         = MAX_ENERGY;
  _forced_labeling = true;
  _changed = false;
  _last_idx_sel = select;
  int label = choose_label( w );
  /// Update the current solution with the new labeling
  update_solution ();
}//force_label

int
MONTECARLO::choose_label ( WorkerAgent* w ) {
  int best_label = 0;
  int n_threads = 32;
  int v_id       = w->get_variable()->get_id();
  int n_blocks   = gh_params.set_size;
  int smBytes    = ( gh_params.n_points + 2 * gh_params.n_res ) * sizeof(real);

  while ( n_threads < gh_params.n_res ) n_threads += 32;
  n_threads = n_threads*2 + 32;
  
#ifdef MONTECARLO_DEBUG
  cout << _dbg << "num. of AA " << _mas_scope_size <<
  " V_id " << v_id << " nres " << gh_params.n_res <<
  " bb start " << _mas_bb_start <<
  " bb end "   << _mas_bb_end <<
  " num. of blocks " << n_blocks <<
  " num. of threads " << n_threads << endl;
  getchar();
#endif

  

  _energy_function->calculate_energy ( gd_params.beam_str, gd_params.beam_energies,
                                       gd_params.validity_solutions, gh_params.n_res,
                                       _mas_bb_start, _mas_bb_end,
                                       _mas_scope_first, _mas_scope_second,
                                       smBytes, n_blocks, n_threads );

  /// Copy Energy Values
  memcpy ( gh_params.beam_energies, gd_params.beam_energies, n_blocks * sizeof( real ) );
  /// Choose best label
  real truncated_number =  Math::truncate_number( gh_params.beam_energies[ best_label ] );
  gh_params.beam_energies[ best_label ] = truncated_number;
  for( int i = 1; i < n_blocks; i++ ) {
    if( Math::truncate_number( gh_params.beam_energies[ i ] ) < truncated_number ) {
      best_label = i;
      truncated_number = gh_params.beam_energies[ best_label ];
      if ( gh_params.sys_job == ab_initio ) { _n_sols++; }
    }
  }


  if ( (gh_params.beam_energies[ best_label ] < _local_current_minimum) ) {
    /// At least one improving structure
    _changed               = true;
    _best_agent            = _last_idx_sel;
    _best_label            = best_label;
    _best_wa               = w;
    _local_current_minimum = gh_params.beam_energies[ best_label ];
    /// Store local best structure
    memcpy ( _curr_best_str,
             &gd_params.beam_str[ best_label * gh_params.n_points ],
             gh_params.n_points * sizeof( real ) );
    /// Take the best structure among all possible samplings
    if ( _local_current_minimum <  _glb_current_minimum ) {
      _glb_current_minimum = _local_current_minimum;
      memcpy ( _glb_best_str, _curr_best_str,
               gh_params.n_points * sizeof( real ) );
    }
  }
  
  /// Try worst choices if not exit asap and everything is ground
  if ( (!_changed) && (!_exit_asap) && all_ground() ) {
    /// We consider a valid structure with energy different from _local_current_minimum
    if ( (gh_params.beam_energies[ best_label ] < MAX_ENERGY) &&
         (gh_params.beam_energies[ best_label ] > _local_current_minimum) ) {
      assign_with_prob ( best_label, w );
    }
    else {
      ///Find another admissible structure (possibly the second best structure)
      int best_label_aux = 0;
      real truncated_number =  Math::truncate_number( gh_params.beam_energies[ best_label_aux ] );
      for( int i = 1; i < n_blocks; i++ ) {
        if( (Math::truncate_number( gh_params.beam_energies[ i ] ) < truncated_number) &&
            (Math::truncate_number( gh_params.beam_energies[ i ] ) > gh_params.beam_energies[ best_label ]) ) {
          best_label_aux = i;
          truncated_number = gh_params.beam_energies[ best_label_aux ];
        }
      }
      /// Set new structure evaluating probability and temperature
      if ( gh_params.beam_energies[ best_label_aux ] < MAX_ENERGY ) {
        assign_with_prob ( best_label_aux, w );
      }
    }
  }//Try worst structure
  
  return 0;
}//choose_label

void
MONTECARLO::assign_with_prob ( int label, WorkerAgent* w, real extern_prob ) {
  real rnd_num = extern_prob;
  if ( rnd_num == 0 ) {
    rnd_num = ( rand () % 101 ) / 100.0;
  }
  
  if ( gh_params.beam_energies[ label ] < _glb_current_minimum ) {
    rnd_num = 1.0;
  }
  
  /// If random is > temperature accept a worst structure
#ifdef MONTECARLO_DEBUG
  cout << _dbg << "rnd_num " << rnd_num << " temp " << (_temperature / 100.0) <<
  " _local_current_minimum " << _local_current_minimum << " new " <<
  gh_params.beam_energies[ label ] << endl;
#endif
  
  if ( rnd_num > (_temperature / 100.0) ) {
    _changed               = true;
    _best_agent            = _last_idx_sel;
    _best_label            = label;
    _best_wa               = w;
    /// Always accept this new worst move
    _local_minimum         = MAX_ENERGY;
    _local_current_minimum = gh_params.beam_energies[ label ];
    /// Store local best structure
    memcpy ( _curr_best_str,
             &gd_params.beam_str[ label * gh_params.n_points ],
             gh_params.n_points * sizeof( real ) );
    /// Take the best structure among all possible samplings
    if ( _local_current_minimum <  _glb_current_minimum ) {
      _glb_current_minimum = _local_current_minimum;
      memcpy ( _glb_best_str, _curr_best_str,
               gh_params.n_points * sizeof( real ) );
    }
  }
}//assign_with_prob

void
MONTECARLO::update_solution () {
  /// Update best structure at the end of one complete scanning through all variables
  if ( _changed || _forced_labeling ) {
    if ( _local_current_minimum < _local_minimum ) {
      /// Update global minimum
      _local_minimum = _local_current_minimum;
      if ( !_labeled_vars[ _best_agent ] ) {
        _height++;
        _labeled_vars[ _best_agent ] = true;
        /// Todo: label the corresponding worker agent
        _best_wa->label ( _best_label );
#ifdef MONTECARLO_DEBUG
        cout << _dbg << "Set ground agt " << _best_agent <<
        " height " << _height << " out of " << _n_vars << endl;
#endif
      }
      _last_wrk_sel = _best_agent;
      memcpy ( gd_params.curr_str, _curr_best_str,
               gh_params.n_points * sizeof( real ) );
//      cout << Utilities::output_pdb_format( _curr_best_str, 0) << endl;
//      exit(2);
    }
    _forced_labeling = false;
  }
}//update_solution

void
MONTECARLO::backtrack () {
  /// Reset domains for the new iteration -> backtrack on validity_solutions
  memcpy (gd_params.validity_solutions, gh_params.validity_solutions,
          gh_params.set_size * sizeof( real ));
}//backtrack_on_bool

void
MONTECARLO::not_labeled ( ) {
  for ( int i = 0; i < _n_vars; i++ )
    if ( !_labeled_vars[ i ] )
      cout << _dbg << "Not labeled: " << i << endl;
}//not_labeled

void
MONTECARLO::dump_statistics ( std::ostream &os ) {
}//dump_statistics
