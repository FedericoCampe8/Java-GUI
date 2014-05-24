/*********************************************************************
 * Constraint
 * Constraint base class implementation.
 *********************************************************************/
#ifndef COCOS_CONSTRAINT__
#define COCOS_CONSTRAINT__

#include "globals.h"

class AminoAcid;

class Constraint {
protected:
  size_t _id;
  int  _weight;
  bool _global;
  bool _fix;
  constr_type _type;
  
  std::vector < AminoAcid* > _scope;
  std::vector <int> _coeffs;
  
  void set_init_fix ();
public:
  Constraint ( constr_type c_type, std::vector<int> vars, int weight=0 );
  Constraint ( constr_type c_type, std::vector<int> vars, std::vector<int> as, int weight=0 );
  ~Constraint();
  
  Constraint ( const Constraint& other );
  Constraint& operator= ( const Constraint& other );
  
  bool operator== ( const Constraint& other ); 
  bool operator() ( Constraint& ci, Constraint& cj );
  
  size_t get_id() const;
  constr_type get_type() const;
  int get_weight() const;
  int get_num_of_events() const;
  constr_events get_event ( int i );
  bool is_global()  const;
  size_t scope_size() const;
  AminoAcid* get_scope_var ( int idx );
  std::vector<int> get_coeff();
  void set_fix ();
  void unset_fix ();
  void set_fix ( bool fix );
  int is_fix ();
  
  void dump ();
};//-

#endif
