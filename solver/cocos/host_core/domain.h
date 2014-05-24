#ifndef COCOS_DOMAIN_H
#define COCOS_DOMAIN_H

#include "globals.h"

class AminoAcid;

class Domain {
 private:
  unsigned int * state;
  unsigned int * r_state;
  size_t _size;
  int _min_value;
  int _max_value;
  AminoAcid* _v_ptr;
  
 public:
  Domain ( int len );
  Domain ( const Domain& other );
  Domain& operator= ( const Domain& other );
  ~Domain ();

  void set_variable_ptr( AminoAcid* v );
  
  unsigned int* get_state ();
  void  set_state ( unsigned int* other_state );
  
  int get_min_value ( ) const;
  int get_max_value ( ) const;
  
  void set ( size_t pos, bool val = true );
  void set_singlet ( size_t pos );
  void unset ( );
  void unset ( size_t pos );

  bool is_valid () const;
  bool is_failed () const;
  bool is_valid ( size_t pos ) const;

  size_t size();
  void dump ();
};

#endif
