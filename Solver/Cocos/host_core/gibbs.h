/*********************************************************************
 * This search engine implements the Gibbs sampling algorithm,
 * a Markov chain Monte Carlo algorithm, special case of the
 * Metropolis- Hastings algorithm.
 * We start with a random assignement to z_1, ..., z_n, then 
 * we sample z_1^t+1 from p(z1 | z_2^t, ..., z_n^t) and so on.
 * We repeat this procedure for T samples.
 * The sample z_i^t+1 is done by a one-dimensional 
 * Metropolis–Hastings step.
 *********************************************************************/
#ifndef COCOS_GIBBS__
#define COCOS_GIBBS__

#include "search_engine.h"

class GIBBS : public SearchEngine {
private:
  /// Number of variables that have been selected for labeling
  int    _level;
  int    _n_vars;
  int    _n_bins;
  int    _n_samples;
  int    _set_size;
  int    _iter_swap_bin;
  int *  _vars_to_shuffle;
  real   _prob_to_swap;
  real * _validity_solutions_aux;
  real * _beam_energies_aux;
  real * _current_best_str;
  /// Vector of pairs ((start bin, end bin), factor scale)
  std::vector< std::pair< std::pair< int, int >, real > > _bin_des;
  std::string _dbg;

  void init_variables ();
  bool create_set ();
  void create_bins ();
  void set_fix_propagators ();
  void swap_bins ();
  void sampling ();
  void reset_iteration ();
  void backtrack ();
  void free_aux_structures ();
  void Metropolis_Hastings_sampling ();
  WorkerAgent* worker_sample_selection ();
  WorkerAgent* init_set_worker_selection ();
public:
  GIBBS ( MasAgent* mas_agt, int init_set_size=MAX_GIBBS_SET_SIZE );
  ~GIBBS ();
  /*
  void set_sequential_scanning ();
  void set_random_scanning ();
  real get_local_minimum () const ;
  bool is_changed () const;
  int  get_n_ground () const;
  bool all_ground () const;
  void reset_iteration ();
  void not_labeled ();
  */
  
  void reset ();
  void search ();
  int choose_label ( WorkerAgent* w );
  
  int get_set_size () const;

  void dump_statistics (std::ostream &os = std::cout);
};

#endif
