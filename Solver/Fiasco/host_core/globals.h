#ifndef FIASCO_GLOBALS_H
#define FIASCO_GLOBALS_H

/* Common dependencies */
#include <stdlib.h>
#include <stdio.h>
#include <cstdio>
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
#include <cstring>
#include <vector>
#include <utility>

// Other
#include "typedefs.h"
#include "mathematics.h"

class Protein;
class TrailStack;
class ConstraintStore;
class LogicVariables;
class AtomGrid;
class Constraint;
class Output;
class Statistics;
class Fragment;

typedef struct {
  /// Energy
  real * h_distances;
  real * h_angles;
  real * contact_params;
  real * tors_params;
  real * tors;
  real * tors_corr;
  aminoacid * aa_seq;
  ss_type* secondary_s_info;

  /// GUI params
  bool fix_fragments; // fixes special fragments to their original position
} GLB_params;


extern AtomGrid g_grid;
extern std::vector<Fragment> g_assembly_db;
extern Protein g_target;
extern Protein g_known_prot;
extern TrailStack g_trailstack;
extern ConstraintStore g_constraintstore;
extern ReferenceSystem g_reference_system;

extern LogicVariables* g_logicvars;
extern std::vector<Constraint*> g_constraints;
extern Output* g_output;
extern Statistics* g_statistics;

// temp sol
extern size_t g_jm_cluster_size;
extern std::pair<size_t, int> g_mem;

extern size_t fails;
extern size_t table_head;
extern size_t body_head;
extern size_t tables;

extern GLB_params g_params;

#endif

