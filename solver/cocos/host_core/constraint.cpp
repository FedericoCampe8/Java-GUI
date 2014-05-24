#include "constraint.h"
#include "logic_variables.h"
#include "aminoacid.h"

using namespace std;

bool constraint_cmp ( const Constraint* ci, const Constraint* cj ) {
  return ( (int)ci->get_weight() < (int)cj->get_weight() ||
          ((int)ci->get_weight() == (int)cj->get_weight() && ci->get_id() < cj->get_id()) );
}//constraint_cmp

Constraint::Constraint ( constr_type c_type, vector<int> vars, int weight ) :
_weight ( weight ),
_global ( false ),
_fix    ( false ),
_type   ( c_type ) {
  vector<int> dummy_coeffs;
  _id = g_constraints.size();
  _coeffs = dummy_coeffs;
  if ( vars.size() > 2 ) _global = true;
  for ( int i = 0; i < vars.size(); i++ )
    _scope.push_back( g_logicvars.cp_variables[ vars[ i ] ] );
  /// Set constraints already at a fix point based on the type of constraint
  set_init_fix ();
  dummy_coeffs.clear();
}

Constraint::Constraint ( constr_type c_type, vector<int> vars, std::vector<int> as, int weight ) :
_weight ( weight ),
_global ( false ),
_fix    ( false ),
_type   ( c_type ) {
  _id = g_constraints.size();
  _coeffs = as;
  if ( vars.size() > 2 ) _global = true;
  for ( int i = 0; i < vars.size(); i++ )
    _scope.push_back( g_logicvars.cp_variables[ vars[ i ] ] );
  /// Set constraints already at a fix point based on the type of constraint
  set_init_fix ();
}//-

Constraint::~Constraint () {
  _scope.clear();
  _coeffs.clear();
}//-

Constraint::Constraint ( const Constraint& other ) {
  _id       = other._id;
  _type     = other._type;
  _weight   = other._weight;
  _global   = other._global;
  _fix      = other._fix;
  _scope    = other._scope;
  _coeffs   = other._coeffs;
}//-

Constraint&
Constraint::operator= ( const Constraint& other ) {
  if (this != &other) {
    _id       = other._id;
    _type     = other._type;
    _weight   = other._weight;
    _global   = other._global;
    _fix      = other._fix;
    _scope    = other._scope;
    _coeffs   = other._coeffs;
  }
  return *this;
}//-

bool
Constraint::operator== ( const Constraint& other ) {
  return ( _id == other._id );
}//-

bool
Constraint::operator() ( Constraint& ci, Constraint& cj ) {
  return ( ci.get_weight() < cj.get_weight() );
}//-

int
Constraint::get_weight () const {
  return _weight;
}//get_weight

bool
Constraint::is_global () const {
  return _global;
}//is_global

constr_type
Constraint::get_type() const {
  return _type;
}//-

size_t
Constraint::get_id () const {
  return _id;
}//-

size_t
Constraint::scope_size() const {
  return _scope.size();
}//-

AminoAcid*
Constraint::get_scope_var ( int idx ) {
  assert( idx < _scope.size() );
  return _scope[ idx ];
}//-

vector<int>
Constraint::get_coeff () {
  return _coeffs;
}//get_coeff

void
Constraint::set_fix () {
  _fix = true;
}//set_fix

void
Constraint::unset_fix () {
  _fix = false;
}//set_fix

void
Constraint::set_fix ( bool fix ) {
  _fix = fix;
}//set_fix

void
Constraint::set_init_fix () {
//  if ( _type == c_k_rang ) {
//    _fix = true;
//  }
//  else if ( _type == c_all_dist && _coeffs[ gh_params.n_res ] ) {
//    _fix = true;
//  }
//  else {
//    _fix = false;
//  }
}//set_init_fix

int
Constraint::is_fix() {
  if ( _fix ) return fix_prop;
  if ( _type == c_sang ) return single_prop;
  return events_size;
}//is_fix

int
Constraint::get_num_of_events () const {
  switch ( _type ) {
    case c_sang:
      return 1;
    case c_k_rang :
      return 1;
    case c_all_dist:
      return 1;
    case c_cg:
      return 1;
    default:
      return 1;
  }
}//get_num_of_events

constr_events
Constraint::get_event( int i ) {
  switch ( _type ) {
    case c_sang:
      return sing_event;
    case c_k_rang:
      return sing_event;
    case c_all_dist:
      return sing_event;
    case c_cg:
      return sing_event;
    default:
      return sing_event;
  }
}//get_event

void
Constraint::dump() {
  cout << "C_" << get_id() << " Type " << get_type() << " " << endl;
  cout << "Vars involved: ";
  for (int i = 0; i < _scope.size(); i++)
    cout << "AA_" << _scope[ i ]->get_id() << " ";
  cout << endl;
  if ( _coeffs.size() > 0 ) {
    cout << "Coeffs: ";
    for (int i = 0; i < _coeffs.size(); i++) {
      cout << _coeffs[i] << " ";
    }
    cout << endl;
  }
  cout << "FIX: ";
  if ( _fix )
    cout << "YES" << endl;
  else
    cout << "NO" << endl;
}//dump
