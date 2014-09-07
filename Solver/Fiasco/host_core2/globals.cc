#include "atom_grid.h"
#include "constraint_store.h"
#include "fragment.h"
#include "mathematics.h"
#include "logic_variables.h"
#include "output.h"
#include "protein.h"
#include "statistics.h"
#include "trailstack.h"

#include <vector>

AtomGrid g_grid(1);
std::vector<Fragment> g_assembly_db;
Protein g_target;
Protein g_known_prot;
TrailStack g_trailstack;
ConstraintStore g_constraintstore;
ReferenceSystem g_reference_system;

LogicVariables* g_logicvars;
std::vector<Constraint*> g_constraints;
Output* g_output;
Statistics* g_statistics;

GLB_params g_params;

// debug
size_t g_jm_cluster_size;
std::pair<size_t, int> g_mem (0,0);
size_t fails = 0;
size_t table_head = 0;
size_t body_head = 0;
size_t tables = 0;
