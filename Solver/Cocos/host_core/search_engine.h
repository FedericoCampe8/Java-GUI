#ifndef COCOS_SEARCH_ENGINE__
#define COCOS_SEARCH_ENGINE__

#include "globals.h"
#include "mas_agent.h"
#include "constraint_store.h"
#include "worker_agent.h"
#include "energy.h"

/*
 * SearchEngine: interface for strategies (i.e., search algorithms).
 * See "Strategy" design pattern.
 * @note: context (i.e., the common interface used by the client) not used here.
 *        Clients use SearchEngine interface directly.
 */
class SearchEngine {
protected:
  bool _abort_search;
  int  _mas_scope_size;
  int  _mas_bb_start;
  int  _mas_bb_end;
  int  _mas_scope_first;
  int  _mas_scope_second;
  real _local_minimum;
  real _energy_weights[ en_fields_size ];
  
  ConstraintStore* _constraint_store;
  std::map< int, WorkerAgent* >* _wrks;
  std::map< int, WorkerAgent* >::iterator _wrks_it;
  
  SearchEngine ( MasAgent* mas_agt ) ;
  SearchEngine ( const SearchEngine& );
  
  /// Energy function: obj function
  Energy * _energy_function;
public:
  real * start_structure;
  
  /// Interface
  virtual ~SearchEngine () = 0;
  virtual void reset    () = 0;
  virtual void search   () = 0;
  virtual int  choose_label ( WorkerAgent* w ) = 0;
  
  void set_status ( real* status, int n );
  
  void abort ();
  bool aborted () const;
  real get_local_minimum () const ;
  
  void dump_statistics ( std::ostream &os = std::cout ) const;
};

#endif
