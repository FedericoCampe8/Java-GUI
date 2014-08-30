#include "aminoacid.h"
#include "utilities.h"
#include "logic_variables.h"

using namespace std;

/// Current Representation of an amino acid:
/// first and last three atoms needed for
/// overlapping planes.
/// (C - O - H -) N - Ca - C - O - H - N
AminoAcid::AminoAcid ( ss_type aa_type ) :
_label    ( -1 ),
_assigned ( false ),
_dom_size ( 0 ),
_aa_type  ( aa_type ) {
  static int _g_CP_VARIABLE_COUNTER = 0;
  _id = _g_CP_VARIABLE_COUNTER++;
  
  fill_domain ( gh_params.domain_angles[ _id ] );
  fill_backbone ();
  domain = new Domain( _dom_size );
  domain->set_variable_ptr( this );
}//-

AminoAcid::AminoAcid ( const AminoAcid& other ) {
  _id            = other._id;
  _label         = other._label;
  _assigned      = other._assigned;
  _dom_size      = other._dom_size;
  _backbone      = other._backbone;
  _domain_values = other._domain_values;
}//-

AminoAcid::~AminoAcid () {
  _backbone.clear();
  _domain_values.clear();
  delete domain;
}//-

Atom
AminoAcid::operator[] ( const int index ) {
  return _backbone[ index ];
}//-

// DFS Labeling
bool
AminoAcid::labeling () {
  _label++;
  if (_label < 0) _label = 0;
  while ( ( _label <= (_dom_size-1) ) &&
          ( !domain->is_valid( _label ) ) ) {
    _label++;
  }
  if ( _label > (_dom_size-1) ) return false;
  
  /* domain.set_singlet ( label ); */
  
  return true;
}//labeling

int
AminoAcid::get_label () const {
  return _label;
}//get_label

int
AminoAcid::get_domain_size () const {
  return _dom_size;
}//get_domain_size

vector< pair < real, real >  >*
AminoAcid::get_domain_values () {
  return &_domain_values;
}//get_domain_values

uint
AminoAcid::get_id() const {
  return _id;
}//get_id

void
AminoAcid::set_assigned() {
  if ( !_assigned ) _assigned = true;
}//set_assigned

void
AminoAcid::unset_assigned() {
  _assigned = false;
}//unset_assigned

bool
AminoAcid::is_assigned() const {
  return _assigned;
}//is_assigned

void
AminoAcid::set_label (int l) {
  _label = l;
}//set_label

void
AminoAcid::reset_label () {
  _label = -1;
}//reset_label

void
AminoAcid::set_singleton ( size_t pos ) {
  set_assigned ();
  set_label ( pos );
  domain->unset();
  domain->set ( pos );
}//set_singleton

void
AminoAcid::unset_singleton ( size_t pos ) {
  unset_assigned ();
  domain->unset( pos );
}//unset_singleton

void
AminoAcid::set_unique_singleton() {
  set_assigned();
}//set_unique_singleton

uint*
AminoAcid::get_dom_state() {
  return domain->get_state();
}//get_dom_state

void
AminoAcid::set_dom_state( uint* other_state ) {
  domain->set_state ( other_state );
}//set_dom_state

real
AminoAcid::get_phi() const {
  if ( _label < 0 )
    return -180.0;
  else
    return _domain_values[_label].first;
}//get_phi

real
AminoAcid::get_psi() const {
  if ( _label < 0 )
    return 180.0;
  else
    return _domain_values[_label].second;
}//get_psi

bool
AminoAcid::is_singleton () {
  return _label >= 0 ;
}//is_singleton

void
AminoAcid::fill_domain ( vector< vector< real > >& angles ) {
  /// Set domain by partitioning the interval [-180, +180]
  if ( gh_params.set_angles > -1 ) {
    real degs = gh_params.set_angles == 0 ? 1 : gh_params.set_angles;
    set_angles( degs );
    
    if ( _domain_values.size() > MAX_DOM_SIZE )
      _domain_values.resize( MAX_DOM_SIZE );
    _dom_size = _domain_values.size();
    
    return;
  }
  /// Set domain by reading the db files
  assert( angles[0].size() == angles[1].size() );
  for (uint i = 0; i < angles[0].size(); i++) {
    if ( _aa_type == helix ) {
      if ( (angles[2][i] == _aa_type) && (angles[1][i] < - 19) )
        _domain_values.push_back ( make_pair( 180.0 + angles[0][i], 180.0 + angles[1][i] ) );
    }
    else if ( _aa_type == sheet ) {
      /*
       ( (angles[0][i] <= -95) && (angles[0][i] >= -156) ) &&
       ( (angles[1][i] <= 160) && (angles[1][i] >=  110) )
       */
      if  ( angles[2][i] == _aa_type )
        _domain_values.push_back ( make_pair( 180.0 + angles[0][i], 180.0 + angles[1][i] ) );
    }
    else if ( _aa_type == turn ) {
      if ( angles[2][i] == _aa_type ) 
        _domain_values.push_back ( make_pair( 180.0 + angles[0][i], 180.0 + angles[1][i] ) );
    }
    else if ( _aa_type == coil ) {
      if ( angles[2][i] == _aa_type )
        _domain_values.push_back ( make_pair( 180.0 + angles[0][i], 180.0 + angles[1][i] ) );
    }
    else {
      // other
      if ( (angles[2][i] == turn) || (angles[2][i] == coil) )
        _domain_values.push_back ( make_pair( 180.0 + angles[0][i], 180.0 + angles[1][i] ) );
    }
  }

  /// Next Try
  if ( _domain_values.size()  == 0 ) {
    cout << "Something went wrong on loading angles for AminoAcid " << _id << endl;
    cout << "...used default angles.\n";
    cout << "Check the offset for the Secondary Structure elements in the input file!\n";
  }
  
  ss_type type_oux = _aa_type;
  if ( _domain_values.size() == 0 ) {
    _aa_type = turn;
    for (uint i = 0; i < angles[0].size(); i++) {
       if  ( angles[2][i] == _aa_type )
      _domain_values.push_back ( make_pair( 180.0 + angles[0][i], 180.0 + angles[1][i] ) );
    }
  }
  _aa_type = type_oux;
  
  assert ( _domain_values.size() > 0 );
  
  if ( _domain_values.size() > MAX_DOM_SIZE )
    _domain_values.resize( MAX_DOM_SIZE );
  _dom_size = _domain_values.size();
}//fill_domain

void
AminoAcid::set_angles ( real deg ) {
  for ( int i = -180; i <= 180; i += deg ) {
    for ( int j = -180; j <= 180; j += deg ) {
      _domain_values.push_back ( make_pair( i, j ) );
    }
  }
}//set_angles

void
AminoAcid::fill_backbone () {
  bool dir;
  real bb_points[27];
  atom_type types[9] = { CB, O, H, N, CA, CB, O, H, N };
  
  ( _id % 2 ) ? dir = true : dir = false;
  Utilities::calculate_aa_points ( dir, bb_points );

  for (int i = 0; i < 9; i++) {
    Atom atm ( bb_points[ i*3 + 0 ], bb_points[ i*3 + 1 ], bb_points[ i*3 + 2 ], types[ i ], _id );
    _backbone.push_back( atm );
  }
}//fill_backbone

void
AminoAcid::dump_domain() {
  cout << "V_" << _id << " ";
  domain->dump();
}//dump_domain

void 
AminoAcid::dump() {
  cout << "V_" << _id << " ";
  cout << "L: " << _label << "\t";
  cout << "| D | = " << _dom_size << " ";
  if ( is_singleton() ) cout << " ASSIGNED\n";
  else cout << " NOT ASSIGNED\n";
  cout << "Type: ";
  switch ( _aa_type ) {
    case helix:
      cout << "HELIX" << endl;
      break;
    case turn:
      cout << "TURN" << endl;
      break;
    case coil:
      cout << "COIL" << endl;
      break;
    case sheet:
      cout << "SHEET" << endl;
      break;
    default:
      cout << "OTHER" << endl;
      break;
  }
  cout << "Phi: " << get_phi() << "\tPsi: " << get_psi() << endl;
  if ( _id )
    cout << "BB start/end: ["<< (_id * 5 - 3) << ", "<< (_id * 5 + 5) << "]" << endl;
  else
    cout << "BB start/end: ["<< 0 << ", "<< (_id * 5 + 5) << "]" << endl;
  
}//dump
