#ifndef COCOS_WORKER_AGENT__
#define COCOS_WORKER_AGENT__

#include "globals.h"
#include "mas_agent.h"
#include "aminoacid.h"

class ConstraintStore;

class WorkerAgent {
private:
  uint _id;
  int _scope_size;
  bool _prop_success;
  std::string _dbg;
  agent_type _agt_type;
  std::vector< AminoAcid* > _scope;
  
public:
  WorkerAgent ( AminoAcid* v );
  WorkerAgent ( const WorkerAgent& );
  ~WorkerAgent();

  uint get_id () const;
  uint get_var_id ( int v=0 ) const;
  AminoAcid* get_variable ( int v=0 );
  int get_dom_size () const;
  int get_scope_size() const;
  void clear_scope ();
  void add_variable ( AminoAcid* v );
  void set_variable ( AminoAcid* v );
  void clear_scope_var ( AminoAcid* v );
  
  bool propagate ( ConstraintStore* c_store, int v=0 );
  bool fail_propagation();
  void label ( int l, int v=0 );
  
  void dump();
};

#endif

