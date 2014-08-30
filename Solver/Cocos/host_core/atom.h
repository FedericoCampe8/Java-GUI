#ifndef COCOS_ATOM__
#define COCOS_ATOM__

#include "typedefs.h"

class Atom {
public:
  real position[3];  /// x,y,z coordinates
  int  ref_aa;       /// Reference to its residue
  int  ref_pos;      /// Reference to its position in the sequence
  atom_type type;
  atom_radii radius;

  Atom() {};
  Atom( real x, real y, real z, atom_type t, int idx=-1 );
  Atom( real* p, atom_type t );
  Atom( const Atom& );
  ~Atom(){}; 
    
  Atom& operator= ( const Atom& );
  real operator[] ( const int index ) const;
  
  bool is_type ( atom_type t ) const;
  void set_type ( atom_type t );
  void set_radius ( atom_type t );
  void set_position( real* p );
  void set_position( real x, real y, real z );
    
  void dump ();
};

#endif
