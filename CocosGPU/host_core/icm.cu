#include "icm.h"
#include "cuda_energy.h"
#include "mathematics.h"

/// FOR TESTING
#include "logic_variables.h"
#include "utilities.h"
#include "cuda_rmsd.h"

//#define ICM_DEBUG
//#define ICM_DEBUG_LABELING
//#define ICM_USE_RMSD

using namespace std;

ICM::ICM ( MasAgent* mas_agt ) :
SearchEngine   ( mas_agt ),  
_var_selection ( 0 ),
_level         ( 0 ),
_height        ( 0 ),
_last_wrk_sel  ( -1 ),
_last_idx_sel  ( -1 ),
_changed       ( false ),
_idx_rand_sel  ( NULL ),
_labeled_vars  ( NULL ),
_dbg           ( "#log: ICM - " ) {
  _local_minimum = MAX_ENERGY;
  srand ( time( NULL ) );
}//-

ICM::~ICM () {
  if ( !_idx_rand_sel )
    delete [] _idx_rand_sel;
  if ( !_labeled_vars )
    delete [] _labeled_vars;
  if ( !start_structure )
    free ( start_structure );
}//-

void
ICM::set_sequential_scanning () {
  _var_selection = 0;
}//set_sequential_scanning

void
ICM::set_random_scanning () {
  _var_selection = 1;
}//set_random_scanning

bool
ICM::is_changed () const {
  return _changed;
}//is_changed

int
ICM::get_n_ground () const {
  return _height;
}//get_n_ground

bool
ICM::all_ground () const {
  return ( _height >= _n_vars );
}//is_changed

void
ICM::reset () {
  /// Set ICM default values
  _level         = 0;
  _n_vars        = _wrks->size();
  _wrks_it       = _wrks->begin();
  _last_wrk_sel  = -1;
  _local_minimum = MAX_ENERGY;
  _changed       = false;
  if ( !_idx_rand_sel ) _idx_rand_sel = new bool[ _n_vars ];
  memset ( _idx_rand_sel, true, _n_vars * sizeof(bool) );
  if ( !_labeled_vars ) {
    _labeled_vars = new bool[ _n_vars ];
    memset ( _labeled_vars, false, _n_vars * sizeof(bool) );
  }
  /// Set Current Structure
  HANDLE_ERROR( cudaMemcpyAsync( gd_params.curr_str, start_structure,
                                 gh_params.n_points * sizeof(real), cudaMemcpyHostToDevice ) );
}//reset

void
ICM::reset_iteration () {
  /// Set ICM default values for a lopp
  _level   = 0;
  _wrks_it = _wrks->begin();
  _changed = false;
  memset ( _idx_rand_sel, true, _n_vars * sizeof(bool) );
  /// Reset domains for the new iteration -> backtrack on validity_solutions
  backtrack ();
}//reset_iteration

WorkerAgent*
ICM::worker_selection () {
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
ICM::search () {
#ifdef ICM_DEBUG
  cout << _dbg << " Start search...\n";
#endif
  
  int label;
  WorkerAgent* w;
  while ( _level < _n_vars ) {
    w = worker_selection ();
    if ( _last_wrk_sel == w->get_id() ) continue;
    /// Propagation
    backtrack ();
    if ( !(w->propagate ( _constraint_store )) ) continue;
    /// Select the label
    label = choose_label( w );
    /// Labeling
    w->label ( label );
    /// If something is changed -> update solution
    update_solution ( w->get_id(), label );
  }
  
  if ( !(_changed || all_ground()) )
    force_label ();
  
#ifdef ICM_DEBUG
  cout << _dbg << "End search\n";
#endif
  
}//search

void
ICM::force_label () {
  cout << "FORCE LABELING" << endl;
  WorkerAgent* w;
  /// Find the first not labeled var
  for (int i = 0; i <= _n_vars; i++) {
    if (i == _n_vars) {
      /// All not labeled variables fail propagation
    
      //cout << Utilities::output_pdb_format( gd_params.curr_str ) << endl;
      
      cout << "BACKTRACK NEEDED!!!" << endl;
      return;
      //exit(2);
    } 
    if ( !_labeled_vars[ i ] ) {
      _wrks_it = _wrks->begin();
      advance ( _wrks_it,  i );
      w = _wrks_it->second;
      _last_idx_sel = i;
      /// Propagate again on _wrks_it
      backtrack ();
      if ( !(w->propagate ( _constraint_store )) ) {
        cout << "Fail in forcing labeling for V_" << w->get_id() << endl;
        continue;
      }
      break;
    }
  }
  /// Find the label corresponding to its minimum
  ///real current_local_min = _local_minimum;
  _local_minimum = MAX_ENERGY;
  int label = choose_label( w );
  /// Labeling
  w->label ( label );
  /// Update solution
  update_solution ( w->get_id(), label );
  /// The following instruction avoid to label again
  /// variables that have been previously labeled.
  /* _local_minimum = min ( current_local_min, _local_minimum ); */
}//force_label

void
ICM::backtrack () {
  /// Reset domains for the new iteration -> backtrack on validity_solutions
  HANDLE_ERROR( cudaMemcpyAsync( gd_params.validity_solutions, gh_params.validity_solutions,
                                 gh_params.set_size * sizeof(real), cudaMemcpyHostToDevice ) );
}//backtrack_on_bool

int
ICM::choose_label ( WorkerAgent* w ) {
  int best_label = 0, n_threads = 32;
  int v_id       = w->get_variable()->get_id();
  int n_blocks   = w->get_variable()->get_domain_size();
  int smBytes    = ( gh_params.n_points + 2 * gh_params.n_res ) * sizeof(real);

  while ( n_threads < gh_params.n_res ) n_threads += 32;
  n_threads = n_threads*2 + 32;
  
#ifdef ICM_DEBUG
  cout << _dbg << "# AA " << _mas_scope_size <<
  " V_id " << v_id <<
  " bb start " << _mas_bb_start <<
  " bb end "   << _mas_bb_end <<
  " # of blocks " << n_blocks << 
  " # of threads " << n_threads << endl;
#endif
  if ( gh_params.follow_rmsd && false ) {
    int num_of_res = _mas_scope_second - _mas_scope_first + 1;
    cuda_rmsd<<< n_blocks, 1, 12*2*num_of_res*sizeof(real) >>>
    ( gd_params.beam_str, gd_params.beam_energies,
      gd_params.validity_solutions,
      gd_params.known_prot, num_of_res, gh_params.n_res,
      _mas_scope_first, _mas_scope_second,
      gh_params.h_def_on_pdb
     );
  }
  else {
    cuda_energy<<< n_blocks, n_threads, smBytes >>>
    ( gd_params.beam_str, gd_params.beam_energies,
      gd_params.validity_solutions,
      gd_params.secondary_s_info,
      gd_params.h_distances, gd_params.h_angles,
      gd_params.contact_params, gd_params.aa_seq,
      gd_params.tors, gd_params.tors_corr,
      _mas_bb_start, _mas_bb_end, gh_params.n_res,
      _mas_scope_first, _mas_scope_second, structure
   );
  }
  /// Copy Energy Values
  HANDLE_ERROR( cudaMemcpy( gh_params.beam_energies, gd_params.beam_energies,
                            n_blocks * sizeof( real ), cudaMemcpyDeviceToHost ) );

  real truncated_number =  Math::truncate_number( gh_params.beam_energies[ best_label ] );
  for( int i = 1; i < n_blocks; i++ ) { 
    if( Math::truncate_number( gh_params.beam_energies[ i ] ) < truncated_number ) {
      best_label = i;
      truncated_number =  Math::truncate_number( gh_params.beam_energies[ best_label ] );
    }
    //cout << "E: " << gh_params.beam_energies[ i ] << endl;
  }
  
#ifdef ICM_DEBUG
  cout << _dbg << "V_" << v_id << " Best label " << best_label << " Best energy "<< std::setprecision(10) <<
  gh_params.beam_energies[ best_label ] <<
  " Local minimum " << _local_minimum << "\n";
  getchar();
#endif
  
  if ( gh_params.beam_energies[ best_label ] < _local_minimum ) {
#ifdef ICM_DEBUG_LABELING
    cout << _dbg << "Label V_" << v_id << " Energy " <<  gh_params.beam_energies[ best_label ] << endl;
#endif
    _local_minimum = gh_params.beam_energies[ best_label ];
    _changed = true;
    if ( !_labeled_vars[ _last_idx_sel ] ) {
      _height++;
      _labeled_vars[ _last_idx_sel ] = true;
#ifdef ICM_DEBUG_LABELING
      cout << _dbg << "Set as TRUE " << _last_idx_sel << " height " << _height << " out of " << _n_vars << endl;
#endif
    }
    return best_label;
  }
  return -1;
}//choose_label

void
ICM::update_solution ( int w_id, int label ) {
  if ( label >= 0 ) {
    _last_wrk_sel = w_id;
    /// Copy current best solution
    HANDLE_ERROR( cudaMemcpy( gd_params.curr_str, &gd_params.beam_str[ label * gh_params.n_res * 15 ],
                              gh_params.n_res * 15 * sizeof( real ) , cudaMemcpyDeviceToDevice ) );
  }
}//update_solution

void
ICM::not_labeled ( ) {
  for ( int i = 0; i < _n_vars; i++ )
    if ( !_labeled_vars[ i ] )
      cout << _dbg << "Not labeled: " << i << endl;
}//not_labeled

void
ICM::dump_statistics ( std::ostream &os ) {
}//dump_statistics
