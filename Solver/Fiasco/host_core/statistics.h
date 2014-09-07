#ifndef FIASCO_STATISTICS_H
#define FIASCO_STATISTICS_H

#include "globals.h"
#include "typedefs.h"

#include <iostream>

class Statistics {
 private:
  size_t backtracks;
  size_t propagation_failures[constr_type_size];
  size_t propagation_successes[constr_type_size];
 
  bool result_is_improved;
  long double loop_search_space;
  long double search_space_explored;
  long double filtered_search_space;
  long double numof_possible_conformations;
  real_exp avg_filtered_domain_elements;
  real_exp clustering_avg_distance_error;
  
  size_t solutions_found;
  size_t solutions_to_file;
  real energy;
  real rmsd[prot_struct_size]; 

  timeval time_stats[t_stat_size];  
  double t_search_limit;
  double t_total_limit;
  double time_start[t_stat_size];
  double time[t_stat_size];
  double total_time[t_stat_size];
  
  /* array containing the size of the loop search space (i.e. 
   * its number of nodes). Position i-th of the loop_search_space_dim
   * contains the size of search space i->loop_le  
   */
  long double *loop_search_space_dim;
  
  /* This structure is used as support to compute the filtered
   * search space. Every row contains the information of an element
   * in a particular domain, if it was explored or not.
   */
  std::vector< std::vector <bool> > filtered_domains;
  // for debugging purpose
  bool intersection;  

  std::vector<real> _RMSD_ensemble;
  
 public:
  Statistics(int argc, char* argv[]);
  ~Statistics();

  void reset(); 
  void new_loop_search_space (uint vf_s, uint vf_e);
  void new_loop_search_space_depr (uint vf_s, uint vf_e);
  long double get_loop_search_space (uint lev);

  bool rmsd_is_improved();
  bool energy_is_improved();
  void incr_soluions_found(uint n=1);
  size_t get_solutions_found();
  void incr_solutions_tofile(uint n=1);
  size_t get_solutions_tofile();

  void set_intersection() {intersection = true;}  
  bool get_intersection() {return intersection;}

  void incr_backtracks(uint n=1);
  void incr_propagation_successes(constr_type c);
  size_t get_propagation_successes (constr_type c) const {return propagation_successes[c]; }
  void incr_propagation_failures(constr_type c);
  size_t get_propagation_failures (constr_type c) const {return propagation_failures[c]; }
  void incr_filtered_search_space(long double n);
  void decr_search_space_filtered(long double n);
  long double get_filtered_search_space () const {return filtered_search_space; }
  void incr_search_space_explored(size_t n=1);
  void decr_search_space_explored(size_t n=1);
  size_t get_search_space_explored() const {return search_space_explored; }
  void set_loop_search_space_size(long double s);
  long double get_loop_search_space_size () const {return loop_search_space; }

  void set_numof_possible_conformations (long double s);
  long double get_numof_possible_conformations () const {return numof_possible_conformations; };

  void set_rmsd (prot_struct_type t, real r);
  void set_best_rmsd (prot_struct_type t, real r);
  void set_best_energy ( real r );
  real get_rmsd (prot_struct_type t);
  real get_energy ();
  std::vector<real> get_rmsd_ensemble ();

  void set_search_timeout (double sec);
  void set_total_timeout (double sec);
  void set_timer (t_stats t);
  void force_set_time(t_stats t);

  double get_timer (t_stats t);
  double get_total_timer (t_stats t);
  void stopwatch (t_stats t);
  bool timeout ();  
  bool timeout_searchtime_only ();
  void incr_clustering_avg_distance_error (real clusters_avg_distance);
  real get_clustering_avg_distance_error ();
  void incr_avg_filtered_domain_elements (real avg_element_filtered);
  real get_avg_filtered_domain_elements ();
  size_t getMaxRSS() const; 
  void dump (std::ostream &os=std::cout);

};

#endif
