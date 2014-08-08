#include "constraint_store.h"
#include "propagator.h"
#include "constraint.h"
#include "aminoacid.h"
#include "logic_variables.h" 

using namespace std;
using namespace Propagator;

//#define CSTORE_DBG_ISOLV

prop_func
func[ c_type_size ] = {
  prop_c_sang,
  prop_c_k_angle_shuffle,
  prop_c_all_dist,
  prop_c_k_rang,
  prop_c_cg,
  prop_c_mang,
  prop_c_dist
};

ConstraintStore::ConstraintStore() :
_dbg      ( "#log: Constraint Store - " ),
_not_init ( true ),
_constraint_queue ( NULL ),
_dom_events ( NULL ) {
}//-

ConstraintStore::~ConstraintStore() {
  if ( !_constraint_queue )     free ( _constraint_queue );
  if ( !_dom_events )           free ( _dom_events );
  if ( !_already_set_cons )     free ( _already_set_cons );
  if ( !_already_set_cons_fix ) free ( _already_set_cons_fix );
}//-

void
ConstraintStore::init() {
  _dom_events           = (int*) malloc( 2 * sizeof(int) );
  _constraint_queue     = (int*) malloc( gh_params.num_cons * sizeof(int) );
  _already_set_cons     = (int*) malloc( gh_params.num_cons * sizeof(int) );
  _already_set_cons_fix = (int*) malloc( gh_params.num_cons * sizeof(int) );
}//init

bool
ConstraintStore::ISOLV ( AminoAcid* v ) {
#ifdef CSTORE_DBG_ISOLV
  cout << _dbg < "ISOLV: enter..." << endl;
#endif
  /// Init constraint store (first time)
  if ( _not_init ) {  init(); _not_init = false; }
  /// Init iteration
  _v_id = v->get_id();
  _first_iteration = true;
  _dom_size = g_logicvars.cp_variables[ _v_id ]->get_domain_size();
  _q_size   = gh_params.constraint_events[ _v_id ][ all_events ].size();
  memset( _already_set_cons,     0, gh_params.num_cons * sizeof(int) );
  memset( _already_set_cons_fix, 0, gh_params.num_cons * sizeof(int) );
  /// Check whether constraints are at their fix-point
  check_memcpy ( _constraint_queue, &gh_params.constraint_events[ _v_id ][ all_events ][ 0 ], &_q_size );
  if ( !_q_size ) return true;
  
  bool more_constraints = true;
  while ( more_constraints ) {
#ifdef CSTORE_DBG_ISOLV
    cout << "ISOLV: while: - # propagators:\t" << n_blocks << endl;
    getchar();
#endif
    
    more_constraints = false;
    if ( !_first_iteration ) memset( _already_set_cons, 0, gh_params.num_cons * sizeof( int ) );
    /// Propagation
    if ( !propagation() ) { return false; }
    /// Update queue
    update_queue ();
  }//while
  
#ifdef CSTORE_DBG_ISOLV
  cout << _dbg << "ISOLV: ...exit" << endl;
#endif
  return true;
}//ISOLVPAR

bool
ConstraintStore::propagation () {
  int c_id, c_idx;
  /// Loop throughout the queue of constraints
  for ( int i = 0; i < _q_size; i++ ) {
    c_id  = _constraint_queue[ i ];
    c_idx = gh_params.constraint_descriptions_idx[ c_id ];
    func[ gh_params.constraint_descriptions[ c_idx ] ] ( _v_id, c_id, c_idx );
  }//i
  /// Copy events on host and check failure: SYNCHRONIZATION POINT
  HANDLE_ERROR( cudaMemcpy( _dom_events, gd_params.domain_events, sizeof(int), cudaMemcpyDeviceToHost ) );
  if ( _dom_events[ 0 ] == failed_event ) {
#ifdef CSTORE_DBG_ISOLV
    cout << _dbg << "ISOLV: fail propagation" << endl;
#endif
    return false;
  }
  return true;
}//propagation

void
ConstraintStore::update_queue () {
  
}//update_queue

void
ConstraintStore::check_memcpy ( int* queue_to, int* queue_from, int* size ) {
  int fix_val, q_idx = 0, length = *size;
  for ( int i = 0; i < length; i++ ) {
    if ( _already_set_cons_fix[ queue_from[ i ] ] ) continue;
    fix_val = g_constraints[ queue_from[ i ] ]->is_fix();
    if ( fix_val == fix_prop ) {
      /// Propagator is at a fix point: do not propagate it
      (*size)--;
      continue;
    }
    if ( fix_val == single_prop ) {
      /// Propagator will be fix in the current run: do not propate it more than this loop
      _already_set_cons_fix[ queue_from[ i ] ] = 1;
    }
    queue_to[ q_idx++ ] = queue_from[ i ];
  }
}//check_memcpy





