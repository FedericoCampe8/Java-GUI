/*********************************************************************
 * CP-Logic Variables 
 * 
 * Lists the logic variables set employed by FIASCO.
 *********************************************************************/
#ifndef FIASCO_LOGIC_VARIABLES__
#define FIASCO_LOGIC_VARIABLES__

#include "typedefs.h"

#include <vector>

class VariablePoint;
class VariableFragment;
class Atom;
class Protein;
class Fragment;


class LogicVariables {
 public:
  std::vector <VariablePoint>     var_point_list;
  std::vector <VariableFragment>  var_fragment_list;
  std::vector <Atom>              var_cg_list;
  
  real energy;   // energy built during branch exploration
  real en_cca;
  real en_ccg;
  real en_ori;
  real en_tor;
  real en_cor;

  LogicVariables (int argc, char* argv[]);

  void populate_fragment_variables
    (Protein target, const std::vector<Fragment>& assembly_db);
  void populate_fragment_variables_md
    (Protein target, const std::vector< std::vector<Fragment> >& assembly_db, 
     int numMd, std::vector<Fragment>& assembly_db_out);
  void reset_allvars_changed();
  void reset();
  int natom_ground (int atomType) const;
  void dump();

};

#endif
