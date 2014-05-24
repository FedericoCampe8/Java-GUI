#ifndef COCOS_GLOBALS_H
#define COCOS_GLOBALS_H

/* Common dependencies */
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <string.h>
#include <unistd.h>

/* Input/Output */
#include <iomanip>
#include <iostream>
#include <fstream>
#include <sstream>

/* Arithmetic and Assertions */
#include <cassert>
#include <cmath>
#include <limits>

/* STL dependencies */
#include <algorithm>
#include <iterator>
#include <map>
#include <memory>
#include <queue>
#include <stack>
#include <set>
#include <string>
#include <vector>
#include <utility>

/* Cuda */
//#include "cuda.h"
//#include <curand.h>
//#include <curand_kernel.h>

/* Other */
#include "typedefs.h"
#include "protein.h"

class LogicVariables;
class Constraint;


// GLOBAL STRUCTURES
typedef struct {
  int quantum;
  int priority;
  agent_type agt_type;
  ss_type sec_str_type;
  search_type search_strategy;
  std::vector< int > vars_list;
  std::vector< std::pair< int, int > > vars_bds;
  std::pair< int, int > scope;
} MasAgentDes;

typedef struct {
  /// Input options
  bool follow_rmsd;
  bool centroid;
  bool gibbs_as_default;
  bool verbose;
  real str_weights[3];
  real crd_weights[3];
  int  n_coordinators;
  int  set_size;
  int  n_gibbs_samples;
  int  n_gibbs_iters_before_swap;
  int  timer;
  /// Output options
  std::string output_file;
  /// Variables and domains
  int n_res;
  int n_points;
  uint * domain_states;
  //uint * bool_states;
  real * validity_solutions;
  /// Constraints info
  int num_cons;
  int max_scope_size;
  std::vector< int > constraint_descriptions;
  std::vector< int > constraint_descriptions_idx;
  std::vector< std::vector< std::vector<int> > > constraint_events;
  /// Secondary Structure and MAS Agents Descriptions
  std::vector< MasAgentDes > mas_des;
  std::vector< std::vector< std::vector< real > > > domain_angles;
  /// Known and target protein
  bool h_def_on_pdb;
  Protein* known_protein;
  Protein* target_protein;
  real * known_bb_coordinates;
  /// Energy
  real   minimum_energy;
  real * h_distances;
  real * h_angles;
  real * contact_params;
  real * tors_params;
  real * tors;
  real * tors_corr;
  real * beam_energies;
} H_GLB_params;

typedef struct {
  /// Target Protein
  real * known_prot;
  /// Variables
  real * curr_str;
  real * beam_str;
  real * beam_str_upd;
  uint * domain_states;
  //uint * bool_states;
  real * validity_solutions;
  /// Domains
  real * all_domains;
  int  * all_domains_idx;
  int  * domain_events;
  /// Energy 
  real * h_distances;
  real * h_angles;
  real * contact_params;
  real * tors_params;
  real * tors;
  real * tors_corr;
  aminoacid * aa_seq;
  ss_type* secondary_s_info;
  real * beam_energies;
} D_GLB_params;

extern H_GLB_params gh_params;
extern D_GLB_params gd_params;

extern LogicVariables g_logicvars;
extern std::vector < Constraint* > g_constraints;

#endif
