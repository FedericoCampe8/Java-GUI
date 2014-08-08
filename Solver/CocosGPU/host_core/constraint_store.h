#ifndef COCOS_CONSTRAINT_STORE_
#define COCOS_CONSTRAINT_STORE_

#include "globals.h"
class AminoAcid;

class ConstraintStore {
private:
  std::string _dbg;
  int   _not_init;
  int   _v_id;
  int   _dom_size;
  int   _q_size;
  int * _dom_events;
  int * _constraint_queue;
  int * _already_set_cons;
  int * _already_set_cons_fix;
  bool _first_iteration;
  
  bool propagation ();
  void update_queue ();
  void check_memcpy ( int* queue_to, int* queue_from, int* size );
public:
  int backtrack_action;
  int curr_labeled_var;
  int first_singlet;
  
  ConstraintStore ();
  ~ConstraintStore ();
  
  void init();
  bool ISOLV ( AminoAcid* aa_var );
  
  /// Preprocessing
  void init_states_on_gpu ();
  
  void dump_domains ();
};//-

#endif
