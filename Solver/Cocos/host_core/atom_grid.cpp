#include "atom_grid.h"
#include "utilities.h"
#include "mathematics.h"
#include "atom.h"

#include <cassert>
#include <cmath>

//#define QUERY_DBG

using namespace std;
using namespace Math;
using namespace Utilities;

AtomGridCell::AtomGridCell() : size ( 0 ) {
}//-

AtomGridCell::AtomGridCell ( const AtomGridCell& other ) {
  size = other.size;
  atom_list = other.atom_list;
}//-

AtomGridCell&
AtomGridCell::operator= (const AtomGridCell& other) {
  if (this != &other) {
    size = other.size;
    atom_list = other.atom_list;
  }
  return *this;
}//-

AtomGrid::AtomGrid( int maxdist, real epsilon ) :
  _grid_max_dist ( maxdist ),
  _epsilon ( epsilon * 100 ){
  space.resize( GRID_EDGE * GRID_EDGE * GRID_EDGE );
}//-

AtomGrid::AtomGrid ( const AtomGrid& other ) {
  _grid_max_dist = other._grid_max_dist;
  space = other.space;
}//-

AtomGrid& 
AtomGrid::operator= ( const AtomGrid& other ) {
  if ( this != &other ) {
    _grid_max_dist = other._grid_max_dist;
    space = other.space;
  }
  return *this;
}//-

void 
AtomGrid::reset() {
  for ( uint i = 0; i < space.size(); i++ ) {
    space[i].size = 0;
    space[i].atom_list.clear();
  }
}//reset

void
AtomGrid::remove( int idx ) {
   space.at( idx ).size--;
}//remove

void
AtomGrid::fill_grid ( std::string path ) {

  ifstream protein_file ( path.c_str() );
  if ( !protein_file.is_open() ) {
    cout << "#log: AtomGrid - Unable to open \"" << path << "\"" << endl;
    return;
  }
  
  const string atom  = "ATOM  ";
  char ok = 0;
  bool set_chain = false;
  string chain = "A";
  string line, token, buf, proper_chain;
  aminoacid c;
  real x, y, z;
  atom_type type      = X;
  atom_type last_type = X;
  
  while ( protein_file.good() ) {
    getline( protein_file, line );
    
    token = line.substr(0, 6);
    
    if ( token == atom ) { // Atom found
      ok = 0;
      buf = line.substr( 12, 5 );
      
      type = Utilities::get_atom_type ( line.substr ( line.length() - 5, line.length() ) );
      
      buf = line.substr ( 16, 1 );
      if(buf == " " || buf == "A") { ok = 1; }
      
      // Protein chain
      buf = line.substr( 21, 1 ); // chain
      
      if ( !set_chain ) {
        proper_chain = line.substr( 21, 1 );
        if ( proper_chain.compare( chain ) != 0 ) chain = proper_chain;
        set_chain = true;
      }
      
      ok *= ( buf == chain ) ? 1 : 0;
      if ( !ok ) type = X;
      
      // coordinates
      x = atof( line.substr( 30, 8 ).c_str() );
      y = atof( line.substr( 38, 8 ).c_str() );
      z = atof( line.substr( 46, 8 ).c_str() );
      
      if ( ok ) {
        if ( type == N  || type == CA  ||  type == CB  || type == O || type == H ) {
          last_type = type;
          add ( x, y, z, type );
        }//type
      }//ok
    }
  }
  
  protein_file.close();
}//fill_grid


void
AtomGrid::add ( real x, real y, real z ) {
  point p;
  p[ 0 ] = x;
  p[ 1 ] = y;
  p[ 2 ] = z;
  add ( p, X, -1 );
}//add

void
AtomGrid::add ( real x, real y, real z, atom_type type ) {
  point p;
  p[ 0 ] = x;
  p[ 1 ] = y;
  p[ 2 ] = z;
  add ( p, type, -1 );
}//add

void 
AtomGrid::add ( point p, atom_type type, int ref_aa ) {
  size_t idx = convert_pos_to_key ( p );
  if ( space.at( idx ).size >= space.at( idx ).atom_list.size() ) {
    allocate_more_atoms ( idx );
  }
  int k = space.at( idx ).size - 1;
  space[ idx ].atom_list[ k ].set_position( p );
  space[ idx ].atom_list[ k ].ref_aa = ref_aa;
  space[ idx ].atom_list[ k ].set_type( type );
}//add

void
AtomGrid::allocate_more_atoms ( int idx ){
  Atom a;
  space.at( idx ).atom_list.push_back ( a );
  space[ idx ].size++;
}//-

real
AtomGrid::query ( real x, real y, real z ) {
  real position[ 3 ];
  position[ 0 ] = x;
  position[ 1 ] = y;
  position[ 2 ] = z;
  return query ( position, X, -1 );
}//query

real
AtomGrid::query ( real x, real y, real z, atom_type type ) {
  real position[ 3 ];
  position[ 0 ] = x;
  position[ 1 ] = y;
  position[ 2 ] = z;
  return query ( position, type, -1 );
}//query

real
AtomGrid::query ( const point& vp, atom_type type ) {
  return query ( vp, type, -1 );
}

real
AtomGrid::query ( const Atom& a ) {
  return query ( a.position, a.type, a.ref_aa );
}//query

//@todo: Investigate on Hash function
real
AtomGrid::query ( const point& vp, atom_type type, int ref_aa, int rad ) {
  int x = (int)floor ( vp[ 0 ] / GRID_SIDE );
  int y = (int)floor ( vp[ 1 ] / GRID_SIDE );
  int z = (int)floor ( vp[ 2 ] / GRID_SIDE );
  int d = _grid_max_dist;

#ifdef QUERY_DBG
  cout << "#log: AtomGrid::query - Query point " <<
  x << " " << y << " " << z << " " << " dist " << d << endl;
#endif
  
  size_t a = 0;
  size_t idx, natoms;
  real dist, tot_dist = 0;
  real limit;
  real distx, disty, distz;
  int radius = (int) Utilities::get_atom_radii ( type );
  if ( rad > 0 ) { radius = rad; }

  // Look in the neighborhood 
  for ( int i = x-d; i <= x+d; i++ )
    for ( int j = y-d; j <= y+d; j++ )
      for ( int k = z-d; k <= z+d; k++ ) {
        idx = convert_cell_to_key ( i, j, k );
        
        /* 
         * For each atom in the cell: 
         * test distance the minimum intra-distance
         * between itself and its neighborhood
         */
        natoms = space.at( idx ).size;
        for ( a = 0; a < natoms; a++ ) {
          int other_ref_aa = space[ idx ].atom_list[ a ].ref_aa;

          if( (abs(other_ref_aa - ref_aa) > 2) ||
              (other_ref_aa < 0) ||
              (ref_aa < 0) ) {
            // Calculate distance between the two atoms
            distx = space[ idx ].atom_list[ a ][ 0 ] - vp[ 0 ];
            disty = space[ idx ].atom_list[ a ][ 1 ] - vp[ 1 ];
            distz = space[ idx ].atom_list[ a ][ 2 ] - vp[ 2 ];
            dist  = sqrt( distx*distx + disty*disty + distz*distz );
            if ( dist == 0 ) dist = CLOSE_TO_ZERO_VAL;
            
            atom_type other_type =
              space[idx].atom_list[a].type;
            atom_radii other_radius = 
              space[idx].atom_list[a].radius;
	    
            limit = (type == CG || other_type == CG) ? 
                    ((radius + other_radius) - _epsilon)/2 :
                    (radius  + other_radius) - _epsilon;
            
            // If distance is less than the threshold ->
            // fail distance constraint: sum the distances
            if ( dist * ( 100 ) < limit ) {
              tot_dist += dist;
            }
            else {
              tot_dist += 0;
            }
          }
        }
      }
  return tot_dist;
}//query

/***************************************************
 *                                                 *
 *              Auxiliary functions                *
 *                                                 *
 ***************************************************/
size_t
AtomGrid::convert_cell_to_key( int x, int y, int z ) {
  return 
  ( x + GRID_EDGE/2 )+
  ( y + GRID_EDGE/2 ) * GRID_EDGE +
  ( z + GRID_EDGE/2 ) * GRID_EDGE * GRID_EDGE;
}//-

size_t
AtomGrid::convert_pos_to_key(point p) {
  int x, y, z;
  x = (int)floor( p[ 0 ] / GRID_SIDE );
  y = (int)floor( p[ 1 ] / GRID_SIDE );
  z = (int)floor( p[ 2 ] / GRID_SIDE );
  return convert_cell_to_key( x , y, z );
}//-
