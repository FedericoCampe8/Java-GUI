#include "gibbs.h"
#include "energy.h"
#include "mathematics.h"
#include "utilities.h"
#include "utilities.h"
#include "constraint.h"
#include "propagator.h"
#include "k_angle_shuffle.h"
#include "all_distant.h"
#include "rmsd_fast.h"

/// FOR TESTING
#include "logic_variables.h"
#include "propagator.h"

//#define GIBBS_DEBUG
//#define GIBBS_DEBUG_ADV
//#define GIBBS_DEBUG_LABELING
//#define GIBBS_USE_RMSD

using namespace std;


GIBBS::GIBBS ( MasAgent* mas_agt, int init_set_size ) :
SearchEngine     ( mas_agt ),
_n_bins          ( 5 ),
_n_samples       ( 50 ),
_set_size        ( init_set_size ),
_iter_swap_bin   ( 50 ),
_prob_to_swap    ( 0.65 ),
_vars_to_shuffle ( NULL ),
_dbg             ( "#log: GIBBS - " ) {
  srand ( time( NULL ) );
  /// Current best structure
  _current_best_str = (real*) malloc ( gh_params.n_res * 15 * sizeof( real ) );
  /// Change default 10 samples
  if ( gh_params.n_gibbs_samples >= 0 ) _n_samples = gh_params.n_gibbs_samples;
  if ( gh_params.n_gibbs_iters_before_swap ) _iter_swap_bin = gh_params.n_gibbs_iters_before_swap;
  _beam_energies_aux = (real*) malloc ( _set_size * sizeof(real) );
  _validity_solutions_aux   = (real*) malloc ( _set_size * sizeof(real) );
  _local_minimum     = MAX_ENERGY;
}//-

GIBBS::~GIBBS () {
  if ( !_current_best_str )         free ( _current_best_str );
  if ( !_validity_solutions_aux )   free ( _validity_solutions_aux );
  if ( !start_structure )           free ( start_structure );
  if ( !_beam_energies_aux )        free ( _beam_energies_aux );
  if ( !_vars_to_shuffle )          free ( _vars_to_shuffle );
  _bin_des.clear();
}//-

void
GIBBS::reset () {
  /// Set Gibbs default values
  _level         = 0;
  _n_vars        = _wrks->size();
  _wrks_it       = _wrks->begin();
  /// Default: current minimum energy to improve
  //_local_minimum = gh_params.minimum_energy;
  _local_minimum = MAX_ENERGY;
  /// Init aux variables (set of updated structures)
  gd_params.beam_str_upd = (real*) malloc ( _set_size * gh_params.n_points * sizeof( real ) );
  /// Set Current Structure
  memcpy ( gd_params.curr_str, start_structure, gh_params.n_points * sizeof(real) );
  _vars_to_shuffle   = (int*) malloc ( _n_vars * sizeof( int ) );
  int i = 0;
  for ( ; _wrks_it != _wrks->end(); ++_wrks_it )
    _vars_to_shuffle[ i++ ] = _wrks_it->first;
  _wrks_it = _wrks->begin();
}//reset

int
GIBBS::get_set_size () const {
  return _set_size;
}//get_set_size

void
GIBBS::search () {
#ifdef GIBBS_DEBUG
  cout << _dbg << "Start search...\n";
#endif
  
  /// Initialize the variables randomly:
  /// set of initial random structures.
  init_variables ();
  /// Create and initialize bins
  create_bins ();
  /// Create the set for the Gibbs sampler
  create_set ();
  /// Create and initialize bins
  create_bins ();
  /// Sample
  sampling ();
  
  free_aux_structures ();
  
#ifdef GIBBS_DEBUG
  cout << _dbg << "End search\n";
#endif
  
}//search

int
GIBBS::choose_label ( WorkerAgent* w ) {
  return 0;
}//choose_label

WorkerAgent*
GIBBS::init_set_worker_selection () {
  if ( _level++ > 0 ) advance ( _wrks_it,  1 );
  return _wrks_it->second;
}//init_set_worker_selection

WorkerAgent*
GIBBS::worker_sample_selection () {
  if ( _level++ > 0 ) advance ( _wrks_it,  1 );
  return _wrks_it->second;
}//worker_sample_selection

void
GIBBS::init_variables () {
#ifdef GIBBS_DEBUG
  cout << _dbg << "Init variables...\n";
#endif
  /// Backtrack on Bool vars
  backtrack ();
  /// Random initial assignment of variables
  k_angle_shuffle( _vars_to_shuffle, _n_vars,
                   gd_params.all_domains, gd_params.all_domains_idx,
                   gd_params.curr_str, gd_params.beam_str,
                   gh_params.set_size, 2, gh_params.n_points * sizeof( real ) );
  /// Check consistency for solutions
  all_distant( gd_params.beam_str,
               gd_params.validity_solutions,
               0, gh_params.set_size, gh_params.n_res, 0 );
}//init_variables

bool
GIBBS::create_set () {
#ifdef GIBBS_DEBUG
  cout << _dbg << "Creating initial set of structures...\n";
#endif
  
  int best_label = 0, n_threads = 32;
  int smBytes    = ( gh_params.n_points + 2 * gh_params.n_res ) * sizeof(real);
  while ( n_threads < gh_params.n_res ) n_threads += 32;
  n_threads = n_threads*2 + 32;
  
  /// Calculate energies on the (initial) set of structures
  _energy_function->calculate_energy ( gd_params.beam_str, gd_params.beam_energies,
                                       gd_params.validity_solutions, gh_params.n_res,
                                       _mas_bb_start, _mas_bb_end,
                                       _mas_scope_first, _mas_scope_second,
                                       smBytes, _set_size, n_threads );

  /// Copy Energy Values
  memcpy ( gh_params.beam_energies, gd_params.beam_energies, _set_size * sizeof( real ) );
  real truncated_number =  Math::truncate_number( gh_params.beam_energies[ best_label ] );
  for( int i = 1; i < _set_size; i++ ) {
    gh_params.beam_energies[ i ] = Math::truncate_number( gh_params.beam_energies[ i ] );
    
    /// Default values for failed structures in the set
    /*
    if ( gh_params.beam_energies[i] == MAX_ENERGY )
      gh_params.beam_energies[i] = gh_params.minimum_energy;
    */
    if( gh_params.beam_energies[ i ]  < truncated_number ) {
      best_label = i;
      truncated_number = gh_params.beam_energies[ best_label ];
    }
  }
  /// Check whether some solution is better than the current one
  if ( gh_params.beam_energies[ best_label ] < _local_minimum ) {
    _local_minimum = gh_params.beam_energies[ best_label ];
#ifdef GIBBS_DEBUG
    cout << _dbg << "Local minimum on set: " << _local_minimum << endl;
#endif
    /// Copy current best solution
    memcpy ( gd_params.curr_str, &gd_params.beam_str[ best_label * gh_params.n_res * 15 ],
             gh_params.n_res * 15 * sizeof( real ) );
  }
  else {
    return false;
  }
  
#ifdef GIBBS_DEBUG_ADV
  cout << Utilities::output_pdb_format( gd_params.curr_str ) << endl;
  getchar();
#endif
  
  /// Initialize set on cuda
  Propagator::prepare_init_set ( gd_params.curr_str, gd_params.beam_str, gd_params.validity_solutions, _set_size );
  return true;
}//create_set

void
GIBBS::create_bins () {
#ifdef GIBBS_DEBUG
  cout << _dbg << "Creating bins...\n";
#endif
  
  int base = _set_size/_n_bins;
  int base_idx = 0;
  int scale_factors[] = { 50, 20, 10, 5, 1 };
  real factor;
  for ( int i = 0; i < _n_bins; i++ ) {
    if (i > 4) factor = 1/((scale_factors[ 4 ] - (1 * i - 4)) * (-1.0));
    else factor = 1/(scale_factors[ i ]*(-1.0));
    pair< int, int > bin_offset ( base_idx, base_idx + base );
    pair < pair< int, int >, real > bin ( bin_offset, factor );
    _bin_des.push_back( bin );
    base_idx += base;
  }
}//create_bins

void
GIBBS::sampling () {
#ifdef GIBBS_DEBUG
  cout << _dbg << "Sampling...\n";
#endif
  
  if ( _n_samples == 0 ) { return; }
  /// Sample
  WorkerAgent* w;
  
#ifdef TIME_STATS
  timeval time_stats;
  double time_start, total_time = 0;
  gettimeofday(&time_stats, NULL);
  time_start = time_stats.tv_sec + (time_stats.tv_usec/1000000.0);
#endif
  
  for ( int t = 0; t < _n_samples; t++ ) {
    //cout << _dbg << "Sample " << t << " out of " << _n_samples << endl;
    /// Reset values from previous iteration
    reset_iteration ();
    while ( _level < _n_vars ) {
      /// Backtrack on Boolean variables
      backtrack ();
      /// Select current variable
      w = worker_sample_selection ();
      /// Failure not checked -> on failure use the previous admissible solution
      w->propagate ( _constraint_store );
      /// Sample the current value from the conditional dependence distribution
      Metropolis_Hastings_sampling ();
    }
    if ( (t > 0) && (t % _iter_swap_bin == 0) && (t < (_n_samples-1)) ) {
      swap_bins ();
    }
  }//t
  
#ifdef TIME_STATS
  gettimeofday(&time_stats, NULL);
  total_time += time_stats.tv_sec + (time_stats.tv_usec/1000000.0) - time_start;
  cout << _dbg << "Sampling avg time per sample: " << total_time/_n_samples << " sec.\n";
#endif
}//sampling

void
GIBBS::Metropolis_Hastings_sampling () {
  /// Calculate current energies
  int best_label = 0, n_threads = 32;
  int smBytes    = ( gh_params.n_points + 2 * gh_params.n_res ) * sizeof( real );
  while ( n_threads < gh_params.n_res ) n_threads += 32;
  n_threads = n_threads*2 + 32;
  
  /// Calculate energies on the updated set of structures
  _energy_function->calculate_energy ( gd_params.beam_str_upd, gd_params.beam_energies,
                                       gd_params.validity_solutions, gh_params.n_res,
                                       _mas_bb_start, _mas_bb_end,
                                       _mas_scope_first, _mas_scope_second,
                                       smBytes, _set_size, n_threads );

  /// Copy previous energy values
  memcpy ( _beam_energies_aux, gh_params.beam_energies, _set_size * sizeof( real ) );
  /// Copy current states
  memcpy ( _validity_solutions_aux, gd_params.validity_solutions, _set_size * sizeof( real ) );
  /// Copy updated energy Values
  memcpy ( gh_params.beam_energies, gd_params.beam_energies, _set_size * sizeof( real ) );
  real upd_energy, probability, rnd_num;
  real truncated_number =  Math::truncate_number( gh_params.beam_energies[ best_label ] );
  for ( int i = 1; i < _set_size; i++ ) {
    upd_energy = Math::truncate_number( gh_params.beam_energies[ i ] );
    if ( upd_energy < truncated_number ) {
      best_label = i;
      truncated_number =  Math::truncate_number( gh_params.beam_energies[ best_label ] );
    }
    
    /// Find bin and accept/reject new value
    for ( int j = 0; j < _n_bins; j++ ) {
      if ( i < _bin_des[ j ].first.second ) {
        if ( upd_energy == MAX_ENERGY ) { continue; }
        if ( _beam_energies_aux[ i ] == upd_energy ) { continue; }
        if ( _beam_energies_aux[ i ] == MAX_ENERGY ) { probability = 1; }
        else {
          rnd_num     = (rand () % 101) / 100.0;
          probability = min ( 1.0,
                             (double) (exp ( (upd_energy * _bin_des[ j ].second) ) /
                             exp ( (_beam_energies_aux[ i ] * _bin_des[ j ].second) )) );
        }
        /*
        if (upd_energy > _beam_energies_aux[ i ] && _beam_energies_aux[ i ] < MAX_ENERGY) {
          cout << i << " bin ub " << _bin_des[ j ].first.second << " scala " << _bin_des[ j ].second << " prob " <<
          (upd_energy * _bin_des[ j ].second) / (_beam_energies_aux[ i ] * _bin_des[ j ].second) <<
          " dove " << upd_energy << " su " << _beam_energies_aux[ i ] <<  " prob " << probability << endl;
          getchar();
        }
        */
        
        /// We use previous structure either if the probability is lower than the ratio or
        /// if the structure is not valid
        if ( (probability <= rnd_num)  ) { //|| (upd_energy == MAX_ENERGY)
          _validity_solutions_aux[ i ] = 0;
        }
        break;
      }
    }//j
  }//i
  /// Check whether some solution is better than the current one
  if ( gh_params.beam_energies[ best_label ] < _local_minimum ) {
    _local_minimum = gh_params.beam_energies[ best_label ];
    
#ifdef GIBBS_DEBUG
    cout << _dbg << "Local minimum on a MH iteration: " << _local_minimum << endl;
#endif
    
    /// Copy current best solution
    memcpy ( gd_params.curr_str, &gd_params.beam_str_upd[ best_label * gh_params.n_res * 15 ],
             gh_params.n_res * 15 * sizeof( real ) );
  }

  /// Copy current updated state
  memcpy ( gd_params.validity_solutions, _validity_solutions_aux,
           _set_size * sizeof(real) );
  /// Update set on cuda
  Propagator::update_set ( gd_params.beam_str, gd_params.beam_str_upd,
                           gd_params.validity_solutions, _set_size );
}//Metropolis_Hastings_sampling

void
GIBBS::swap_bins () {
  real rnd_num, factor;
  for ( int i = 0; i < _n_bins-1; i++ ) {
    rnd_num = (rand () % 101) / 100.0;
    if ( rnd_num > _prob_to_swap ) {
      factor = _bin_des[ i ].second;
      _bin_des[ i ].second   = _bin_des[ i+1 ].second;
      _bin_des[ i+1 ].second = factor;
    }
  }
}//swap_bins

void
GIBBS::reset_iteration () {
  /// Set ICM default values for a lopp
  _level   = 0;
  _wrks_it = _wrks->begin();
}//reset_iteration

void
GIBBS::backtrack () {
  /// Reset domains for the new iteration -> backtrack on bool_states
  memcpy ( gd_params.validity_solutions, gh_params.validity_solutions,
           _set_size * sizeof(real) );
}//backtrack

void
GIBBS::set_fix_propagators () {
  for ( int i = 0; i < g_constraints.size(); i++ ) {
    if ( g_constraints[ i ]->get_type() == c_k_rang )
      g_constraints[ i ]->unset_fix ();
    if ( (g_constraints[ i ]->get_type() == c_all_dist) &&
        (g_constraints[ i ]->get_coeff()[ gh_params.n_res ] ) )
      g_constraints[ i ]->unset_fix ();
    if ( (g_constraints[ i ]->get_type() == c_all_dist) &&
        (!g_constraints[ i ]->get_coeff()[ gh_params.n_res ] ) )
      g_constraints[ i ]->set_fix ();
  }
}//set_fix_propagators

void
GIBBS::free_aux_structures() {
  free ( gd_params.beam_str_upd );
}//free_aux_structures

void
GIBBS::dump_statistics ( std::ostream &os ) {
}//dump_statistics

