#include "domain.h"
#include "aminoacid.h"

using namespace std;

Domain::Domain ( int d_size ) :
 _size ( d_size ),
 _min_value ( 0 ),
 _max_value ( d_size-1 ),
 _v_ptr ( NULL ) {
   state   = ( unsigned int* ) calloc( MAX_DIM, sizeof( unsigned int ) );
   r_state = ( unsigned int* ) calloc( MAX_DIM, sizeof( unsigned int ) );
   for (int i = 0; i < _size; i++) SETBIT( state, i );
   memcpy( r_state, state, MAX_DIM * sizeof( unsigned int )) ;
}//-

Domain::Domain ( const Domain& other ) {
  _size = other._size;
  _min_value = other._min_value;
  _max_value = other._max_value;
  _v_ptr = other._v_ptr;
  state   = ( unsigned int* ) calloc( MAX_DIM, sizeof( unsigned int ) );
  r_state = ( unsigned int* ) calloc( MAX_DIM, sizeof( unsigned int ) );
  memcpy( state, other.state, MAX_DIM * sizeof( unsigned int )) ;
  memcpy( r_state, state, MAX_DIM * sizeof( unsigned int )) ;
}//-

Domain&
Domain::operator= (const Domain& other) {
  if ( this != &other ) {
    _size = other._size;
    _min_value = other._min_value;
    _max_value = other._max_value;
    _v_ptr = other._v_ptr;
    state   = ( unsigned int* ) calloc( MAX_DIM, sizeof( unsigned int ) );
    r_state = ( unsigned int* ) calloc( MAX_DIM, sizeof( unsigned int ) );
    memcpy( state, other.state, MAX_DIM * sizeof( unsigned int ) ) ;
    memcpy( r_state, other.state, MAX_DIM * sizeof( unsigned int ) ) ;
  }
  return *this;
}//-

Domain::~Domain () {
  free( state );
  free( r_state );
}//-

void
Domain::set_variable_ptr( AminoAcid* v ) {
  _v_ptr = v;
}//set_variable_ptr

unsigned int*
Domain::get_state () {
  return state;
}//get_state

void 
Domain::set_state ( unsigned int* other_state ) {
  memcpy ( state, other_state, MAX_DIM * sizeof( unsigned int ) );
}//set_state

void  
Domain::set( size_t pos, bool val) {
  if( val ) {
    SETBIT( state, pos );
  }
  else {
    CLEARBIT( state, pos );
  }
}//set

void
Domain::set_singlet( size_t pos ) {
  for (int i = 0; i < MAX_DIM; i++)
    state[ i ] = 0;
  SETBIT( state, pos );
}//set_singlet

void
Domain::unset () {
  memcpy( state, r_state, MAX_DIM * sizeof( unsigned int ) ) ;
}//unset

void
Domain::unset ( size_t pos ) {
  SETBIT( state, pos );
}//unset

int
Domain::get_min_value () const {
  return _min_value;
}//get_min_value

int
Domain::get_max_value () const {
  return _max_value;
}//get_max_value

bool
Domain::is_valid () const {
  for ( int i = _min_value; i <= _max_value; i++ )
    if ( ISBITSET( state, i ) )
      return true;
  return false;
}//is_valid

bool
Domain::is_failed () const {
  return ( !is_valid() );
}//is_failed

bool
Domain::is_valid (size_t pos) const {
  return ( ISBITSET( state, pos ) );
}//is_valid

size_t
Domain::size(){
  return _size;
}//size

void
Domain::dump() {
  assert( _v_ptr );
  int i = 0;
  int l = _v_ptr->get_label();
  cout << "{";
  for (; i < _size; i++) {
    if ( (i % 8 == 7) && (ISBITSET( state, i ) )) cout << endl;
    if ( i == l ) {
      cout << "[(" << (_v_ptr->get_domain_values())->operator[]( i ).first <<
      ", " << (_v_ptr->get_domain_values())->operator[]( i ).second << ")], ";
    }
    else if ( ISBITSET( state, i ) )
      cout << "(" <<
     (_v_ptr->get_domain_values())->operator[]( i ).first<< ", " <<
     (_v_ptr->get_domain_values())->operator[]( i ).second << "), ";
  }
  cout << "}\n";
  
}//dump

