/****************************************
 *  Energy calculated by                *
 *  considering contacts between atoms. *
 *  This energy function is used for    *
 *  docking analysis.                   *
 ****************************************/
#ifndef COCOS_CONTACT_ENERGY__
#define COCOS_CONTACT_ENERGY__
 
#include "globals.h"
#include "energy.h"

class ContactEnergy : public Energy {
protected:
  aminoacid * _aa_seq;
public:
  ContactEnergy () {};
  virtual ~ContactEnergy () {};
  
  virtual void calculate_energy ( real* setOfStructures, real* setOfEnergies,
                                  real* validStructures, int n_res,
                                  int bb_start, int bb_end,
                                  int scope_start, int scope_end,
                                  int n_bytes, int n_blocks, int n_threads );
  
  ContactEnergy * set_parameters ( aminoacid * aa_seq );
  
};

#endif


