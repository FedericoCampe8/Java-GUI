/*********************************************************************
 * This search engine implements a MonteCarlo random sampling of the,
 * search space.
 *********************************************************************/
#ifndef COCOS_MOTECARLO__
#define COCOS_MOTECARLO__

#include "search_engine.h"

class MONTECARLO : public SearchEngine {
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
  /// Current values and parameters (changed at each sampling step)
  int _best_agent;
  real _local_current_minimum;
  real * _curr_best_str;
  real _glb_current_minimum;
  real * _glb_best_str;
  bool _forced_labeling;
  int _best_label;
  WorkerAgent* _best_wa;
  /// Number of iteration before stop
  int _iter_counter;
  int _max_iterations;
  int _n_of_restarts;
  int _max_n_restarts;
  /// Temperature
  real _temperature;
  real _decreasing_factor;
  /// Exit as soon as no changes happen (otherwise try more random samples)
  bool _exit_asap;
  
  bool _changed;
  bool* _idx_rand_sel; 
  bool* _labeled_vars;
  std::string _dbg;
  
  WorkerAgent* worker_selection ();
  void force_label ();
  void assign_with_prob ( int label, WorkerAgent* w );
  void update_solution ();
  void backtrack ();
public:
  MONTECARLO ( MasAgent* mas_agt );
  ~MONTECARLO ();
  
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
//  void update_solution ( int w_id, int label );
  
  void dump_statistics (std::ostream &os = std::cout);
};

#endif
