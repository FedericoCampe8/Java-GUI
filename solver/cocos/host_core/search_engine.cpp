#include "search_engine.h"

SearchEngine::SearchEngine ( MasAgent* mas_agt ) :
_abort_search      ( false ),
_mas_scope_size    ( mas_agt->var_list_size() ),
_mas_bb_start      ( mas_agt->get_atom_bounds( 0 ) ),
_mas_bb_end        ( mas_agt->get_atom_bounds( 1 ) ),
_mas_scope_first   ( mas_agt->get_scope_start() ),
_mas_scope_second  ( mas_agt->get_scope_end() ),
_constraint_store  ( mas_agt->get_c_store() ),
_wrks              ( mas_agt->get_workers() ),
_wrks_it           ( mas_agt->get_workers()->begin() ) {
  for ( int i = 0; i < en_fields_size; i++ )
    _energy_weights[ i ] = mas_agt->get_energy_weight ( i );
  start_structure = (real* ) calloc ( mas_agt->get_n_points(), sizeof(real) );
}//-

SearchEngine::SearchEngine ( const SearchEngine& other ) {
  _abort_search = other._abort_search;
  _mas_scope_size = other._mas_scope_size;
  _mas_bb_start = other._mas_bb_start;
  _mas_bb_end = other._mas_bb_end;
  _constraint_store = other._constraint_store;
  _wrks = other._wrks;
  _wrks_it = other._wrks_it;
  for ( int i = 0; i < en_fields_size; i++ )
    _energy_weights[ i ] = other._energy_weights[ i ];
  start_structure = (real* ) calloc ( MAX_TARGET_SIZE*15, sizeof(real) );
}//-

SearchEngine::~SearchEngine() {
  free ( start_structure );
}//-

void
SearchEngine::set_status ( real* status, int n ) {
  memcpy ( start_structure, status, n * sizeof(real) );
}//set_status

real
SearchEngine::get_local_minimum () const {
  return _local_minimum;
}//get_local_minumum

void
SearchEngine::reset () {
  _abort_search = false;
}//-

void
SearchEngine::abort () {
  _abort_search = true;
}//-

bool
SearchEngine::aborted () const {
  return _abort_search;
}//-

void
SearchEngine::dump_statistics ( std::ostream &os ) const {
}//-
