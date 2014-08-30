/***************************************
 *       Energy calculated given       *
 *       potential fields              *
 ***************************************/
#ifndef COCOS_POTENTIAL_ENERGY__
#define COCOS_POTENTIAL_ENERGY__
 
#include "globals.h"
#include "energy.h"

class PotentialEnergy : public Energy {
protected:
  /// Energy tables and structure parameters
  ss_type   * _secondary_s_info;
  real      * _h_distances;
  real      * _h_angles;
  real      * _contact_params;
  real      * _tors;
  real      * _tors_corr;
  aminoacid * _aa_seq;
  /// Energy weights
  real _hydrogen_weight;
  real _contact_weight;
  real _correlation_weight;
  /*
   * Energy fields: function
   * @Todo: create interface for fields and array of energy
   * fields to use with a visitor pattern.
   */
  void hydrogen_energy    ( real * structure, real * h_values,
                            int bb_start, int bb_end, int n_res, int idx );
  void contact_energy     ( real * structure, real * c_values,
                            int bb_start, int bb_end, int n_res, int idx );
  void contact_energy_cg  ( real * structure, int first_cg_idx, int second_cg_idx,
                            real* c_energy );
  void correlation_energy ( real * structure, real * corr_val,
                            int bb_start, int bb_end, int n_res, int v_id=0 );
  /// Auxiliary function
  int get_h_distance_bin ( real distance );
  int get_h_angle_bin    ( real angle );
  int get_corr_aa_type   ( aminoacid );
public:
  PotentialEnergy () {};
  virtual ~PotentialEnergy () {};
  
  virtual void calculate_energy ( real* setOfStructures, real* setOfEnergies,
                                  real* validStructures, int n_res,
                                  int bb_start, int bb_end,
                                  int scope_start, int scope_end,
                                  int n_bytes, int n_blocks, int n_threads );
  
  PotentialEnergy * set_parameters ( ss_type* secondary_s_info,
                                     real* h_distances, real* h_angles,
                                     real * contact_params, aminoacid * aa_seq,
                                     real * tors, real * tors_corr,
                                     real hydrogen_w, real contact_w, real correlation_w );
  
};

#endif


