/****************************************
 *  Energy calculated by                *
 *  considering contacts between atoms. *
 *  This energy function is used for    *
 *  docking analysis.                   *
 *  It is different from contact energy *
 *  since it considers a decay factor   *
 *  that depends on the distance        *
 *  between a specific pair of atoms.   *
 ****************************************/
#ifndef COCOS_CONTACT_DECAY_ENERGY__
#define COCOS_CONTACT_DECAY_ENERGY__
 
#include "globals.h"
#include "energy.h"

class ContactDecayEnergy : public Energy {
protected:
  /// AA sequence for contact energy
  aminoacid * _aa_seq;
  /// Index of the peptide atom to check
  int _atom_idx;
  /// Coordinates of the atoms to check for distance decadement
  point _atom_coordinates;
public:
  ContactDecayEnergy () {};
  virtual ~ContactDecayEnergy () {};
  
  virtual void calculate_energy ( real* setOfStructures, real* setOfEnergies,
                                  real* validStructures, int n_res,
                                  int bb_start, int bb_end,
                                  int scope_start, int scope_end,
                                  int n_bytes, int n_blocks, int n_threads );
  
  ContactDecayEnergy * set_parameters ( aminoacid * aa_seq, point atom, int atom_idx );
  
};

#endif


