#ifndef COCOS_AMINOACID__
#define COCOS_AMINOACID__

#include "globals.h"
#include "atom.h"
#include "domain.h"

class AminoAcid {
private:
  /// FD Variable fields
  uint  _id;
  int   _label;
  bool  _assigned;
  uint  _dom_size;
  /// AminoAcid fields
  ss_type _aa_type;
  std::vector< Atom > _backbone;
  std::vector< std::pair < real, real > > _domain_values;
  
  void fill_domain ( std::vector< std::vector< real > >& domain_angles );
  void fill_backbone();
  
public:
  /// Bit mask domain
  Domain* domain;
  
  AminoAcid ( ss_type aa_type=other );
  AminoAcid ( const AminoAcid& );
  ~AminoAcid ();
  
  Atom operator[]  ( const int index );
  
  /// DFS Labeling
  bool labeling ();  
  uint get_id() const;
  real get_phi() const;
  real get_psi() const;
  void set_assigned();
  void unset_assigned();
  bool is_assigned() const;
  bool is_singleton();
  int  get_label() const;
  void set_label ( int l );
  void reset_label();
  void set_singleton (size_t idx);
  void unset_singleton ( size_t idx );
  void set_unique_singleton();
  int   get_domain_size() const;
  uint* get_dom_state();
  void  set_dom_state( uint* other_state  );
  std::vector< std::pair < real, real > >* get_domain_values();
  /* void trail_back ( TrailVariable& tv ); */
  
  void dump ();
  void dump_domain ();
};

#endif
