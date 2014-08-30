/***************************************
 *           Energy interface          *
 ***************************************/
#ifndef COCOS_ENERGY__
#define COCOS_ENERGY__
 
#include "globals.h"

class Energy {
public:
  virtual ~Energy () {};
  virtual void calculate_energy ( real* setOfStructures, real* setOfEnergies,
                                  real* validStructures, int n_res,
                                  int bb_start, int bb_end,
                                  int scope_start, int scope_end,
                                  int n_bytes, int n_blocks, int n_threads ) = 0;
};

#endif


