#ifndef COCOS_PROPAGATOR__
#define COCOS_PROPAGATOR__

#include "globals.h"

typedef void (*prop_func) ( int v_id, int c_id, int c_idx );

namespace Propagator{
  void prop_c_sang            ( int v_id, int c_id, int c_idx );
  void prop_c_cg              ( int v_id, int c_id, int c_idx );
  void prop_c_all_dist        ( int v_id, int c_id, int c_idx );
  void prop_c_k_angle_shuffle ( int v_id, int c_id, int c_idx );
  void prop_c_k_rang          ( int v_id, int c_id, int c_idx );
  void prop_c_mang            ( int v_id, int c_id, int c_idx );
  void prop_c_dist            ( int v_id, int c_id, int c_idx );
}

#endif