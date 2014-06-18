#ifndef COCOS_LOGIC_VARIABLES__
#define COCOS_LOGIC_VARIABLES__

#include "globals.h"
#include "aminoacid.h"

using namespace std;

class LogicVariables {
private:
  std::string _dbg;
public:
  real* cp_structure;
  std::vector < AminoAcid* >  cp_variables;

  LogicVariables();
  ~LogicVariables();
  
  void populate_logic_variables ();
  void populate_point_variables ();
  void init_logic_variables ();
  void set_point_variables ( real* pt_vars );
  void set_interval_point_variables ( real* pt_vars, int bb_start, int bb_end, real* dest=NULL );
  
  void clear_variables ();
  
  void dump();
  void print_point_variables ();
  void print_point_variables ( int start_aa, int end_aa );
};

#endif
