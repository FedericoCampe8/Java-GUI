#ifndef COCOS_PROPAGATOR__
#define COCOS_PROPAGATOR__

#include "globals.h"

typedef void (*prop_func) ( int v_id, int c_id, int c_idx );

void check_failure ( real* domain_states, int* events, int n );

namespace Propagator{
  void print_beam ( real* beam, int s, int e );
  void prepare_init_set ( real* current_str, real* beam_str, real* states, int n_blocks );
  void update_set ( real* current_beam, real* upd_beam, real* states, int n_blocks );
  
  void prop_c_sang            ( int v_id, int c_id, int c_idx );
  void prop_c_cg              ( int v_id, int c_id, int c_idx );
  void prop_c_all_dist        ( int v_id, int c_id, int c_idx );
  void prop_c_k_rang          ( int v_id, int c_id, int c_idx );
  void prop_c_mang            ( int v_id, int c_id, int c_idx );
  void prop_c_dist            ( int v_id, int c_id, int c_idx );
}

#endif