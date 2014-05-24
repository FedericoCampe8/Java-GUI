#ifndef COCOS_MAS_AGENT__
#define COCOS_MAS_AGENT__

#include "globals.h"

class SearchEngine;
class ConstraintStore;
class WorkerAgent;

class MasAgent {
protected:
  int  _id;
  int  _n_res;
  int  _n_points;
  int  _quantum;
  int  _priority;
  int  _sum_dom_size;
  int  _max_dom_size;
  bool _end_search;
  std::string _dbg;
  agent_type _agt_type;
  ss_type _sec_str_type;
  search_type _search_strategy;
  real _energy_weights[ en_fields_size ];
  std::vector< int > _vars_list;
  std::pair< int, int > _atoms_bds;
  std::vector< std::pair< int, int > > _vars_bds;
  std::pair< int, int > _scope;
  std::map< int, WorkerAgent* > _wrk_agt; /// V_id, WrkAgt*
  
  real* _current_status;
  
  SearchEngine*    _search_engine;
  ConstraintStore* _constraint_store;
  
public:
  MasAgent ( MasAgentDes description, int prot_len );
  MasAgent ( const MasAgent& other );
  MasAgent& operator= ( const MasAgent& other );
  virtual ~MasAgent ();
  
  int        get_quantum () const;
  void       set_quantum ( int q );
  int        get_priority () const;
  void       inc_priority ();
  void       set_priority ( int p );
  int        get_n_res () const;
  int        get_n_points () const;
  void       set_energy_weight ( int en_field, real w );
  real       get_energy_weight ( int en_field );
  void       end_search ();
  bool       done () const;
  
  void       clear_data();
  void       add_worker   ( WorkerAgent* wrk );
  int        var_list_size () const;
  int        get_var_id ( int v ) const;
  int        get_bounds ( int b ) const;
  int        get_atom_bounds ( int b ) const;
  int        get_bb_start () const;
  int        get_bb_end () const;
  int        get_aa_start () const;
  int        get_aa_end () const;
  int        get_scope_start () const;
  int        get_scope_end () const;
  agent_type get_agt_type () const;
  
  std::map< int, WorkerAgent* >* get_workers();
  ConstraintStore* get_c_store();
  
  void set_current_status  ( real* c_status );
  real* get_current_status ();
  void dump_general_info ();
  
  void search_alloc ( int max_beam_size=0 );
  void search_init ();
  void search_free ();
  
  virtual void search () = 0;
  virtual void dump () = 0;
};

#endif

