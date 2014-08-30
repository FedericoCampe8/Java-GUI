/***************************************
 *       Energy calculated given       *
 *       by RMSD values.               *
 ***************************************/
#ifndef COCOS_RMSD_ENERGY__
#define COCOS_RMSD_ENERGY__
 
#include "globals.h"
#include "energy.h"

class RmsdEnergy : public Energy {
protected:
  /// Known protein to compare with
  real * _known_prot;
public:
  RmsdEnergy () {};
  virtual ~RmsdEnergy () {};
  
  virtual void calculate_energy ( real* setOfStructures, real* setOfEnergies,
                                  real* validStructures, int n_res,
                                  int bb_start, int bb_end,
                                  int scope_start, int scope_end,
                                  int n_bytes, int n_blocks, int n_threads );
  
  RmsdEnergy * set_parameters ( real* known_prot );
  
};

#endif


