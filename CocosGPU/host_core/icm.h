/*********************************************************************
 * This search engine implements the Iterated Conditional Modes,
 * an iterative technique that finds a local maximum of the 
 * joint probability in a Markov Random Field model.
 *********************************************************************/
#ifndef COCOS_ICM__
#define COCOS_ICM__

#include "search_engine.h"

class ICM : public SearchEngine {
private:
  int _n_vars;
  /// Variable selection strategy: 0 -> seq, 1 -> random
  int _var_selection;
  /// Number of variables that have been selected for labeling
  int _level;
  /// Number of variables that have been labeled
  int _height;
  /// Idx of the last selected worker
  int _last_wrk_sel;
  /// Idx in the list of workers of the last selected worker
  int _last_idx_sel;
  bool _changed;
  bool* _idx_rand_sel;
  bool* _labeled_vars;
  std::string _dbg;
  
  WorkerAgent* worker_selection ();
  void force_label ();
  void backtrack ();
public:
  ICM ( MasAgent* mas_agt );
  ~ICM ();
  
  void set_sequential_scanning ();
  void set_random_scanning ();
  bool is_changed () const;
  int  get_n_ground () const;
  bool all_ground () const;
  void reset ();
  void reset_iteration ();
  void not_labeled ();
  
  void search ();
  int choose_label ( WorkerAgent* w );
  void update_solution ( int w_id, int label );
  
  void dump_statistics (std::ostream &os = std::cout);
};

#endif
