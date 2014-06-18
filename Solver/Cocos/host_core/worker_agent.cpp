#include "worker_agent.h"
#include "constraint_store.h"

using namespace std;

WorkerAgent::WorkerAgent ( AminoAcid* v ) :
_scope_size   ( 1 ),
_prop_success ( true ),
_dbg          ( "#log: Worker - " ),
_agt_type     ( worker ) {
  static size_t _g_WRK_COUNTER = 0;
  _id = _g_WRK_COUNTER++;
  _scope.push_back( v );
}//-

WorkerAgent::WorkerAgent ( const WorkerAgent& other ) {
  _id           = other._id;
  _dbg          = other._dbg;
  _scope        = other._scope;
  _agt_type     = other._agt_type;
  _scope_size   = other._scope_size;
  _prop_success = other._prop_success;
}//copy constructor

WorkerAgent::~WorkerAgent () {
  for (int i = 0; i < _scope.size(); i++)
    delete _scope[i];
  _scope.clear();
}//-

uint
WorkerAgent::get_id () const {
  return _id;
}//get_id

uint
WorkerAgent::get_var_id ( int v ) const {
  if (v >= _scope_size) return 0;
  return _scope[ v ]->get_id();
}//get_id

AminoAcid*
WorkerAgent::get_variable ( int v ) {
  if (v >= _scope_size) return NULL;
  return _scope[ v ];
}//get_id

int
WorkerAgent::get_scope_size () const {
  return _scope_size;
}//get_scope_size

int
WorkerAgent::get_dom_size () const {
  int size = 0;
  for (int i = 0; i < _scope_size; i++)
    size += _scope[ i ]->get_domain_size();
  return size;
}

void
WorkerAgent::add_variable ( AminoAcid* v ) {
  _scope_size++;
  _scope.push_back ( v );
}//add_variable

void
WorkerAgent::set_variable ( AminoAcid* v ) {
  _scope_size = 1;
  _scope.clear();
  _scope.push_back( v );
}//add_variable

void
WorkerAgent::clear_scope () {
  _scope_size = 0;
  _scope.clear();
}//clear_scope

void
WorkerAgent::clear_scope_var ( AminoAcid* v ) {
  vector< AminoAcid* > reduced;
  for ( int i = 0; i < _scope_size; i++ ) {
    if ( v->get_id() == _scope[ i ]->get_id() ) continue;
    reduced.push_back( _scope[i] );
  }
  _scope.swap( reduced );
  _scope_size--;
  
  reduced.clear();
}//clear_scope_var

bool
WorkerAgent::propagate ( ConstraintStore* c_store, int v ) {
  _prop_success = c_store->ISOLV ( _scope[ v ] );
  return _prop_success;
}//propagate

bool
WorkerAgent::fail_propagation () {
  return !_prop_success;
}//fail_propagation

void
WorkerAgent::label ( int l, int v ) {
  if ( l < 0 ) return;
  _scope[ v ]->set_singleton ( l );
}//label

void
WorkerAgent::dump() {
  cout << "WORKER Agent " << _id << " on:\n";
  for (int i = 0; i < _scope_size; i++)
    _scope[ i ]->dump();
}//dump
