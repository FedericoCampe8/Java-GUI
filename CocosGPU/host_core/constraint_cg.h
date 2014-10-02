/*********************************************************************
 * Authors: Federico Campeotto (campe8@nmsu.edu)                     *
 *                                                                   *
 * (C) Copyright 2012-2013                                           *
 *                                                                   *
 * This file is part of COCOS (COncurrent system with COnstraints    *
 * for protein Structure prediction).                                *
 *                                                                   *
 * COCOS is free software; you can redistribute it and/or            *
 * modify it under the terms of the GNU General Public License       *
 * as published by the Free Software Foundation;                     *
 *                                                                   *
 * COCOS is distributed WITHOUT ANY WARRANTY; without even the       *
 * implied  warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR  *
 * PURPOSE. See the GNU General Public License for more details.     *
 *                                                                   *
 * You should have received a copy of the GNU General Public License *
 * along with this program; if not, see http://www.gnu.org/licenses. *
 *                                                                   *
 *********************************************************************/
#ifndef COCOS_CONSTRAINT_CG__
#define COCOS_CONSTRAINT_CG__

#include "constraint_shell.h"
#include "typedefs.h"

using namespace std;

/*
 * Used to set a minimum distance between atoms.
 * This is done in order to avoid steric clashes.
 */
class ConstraintCG : public Constraint {
private:
  string dbg;
  int d_size;
  int n_tertiary_atoms;
  int bb_start;
  int bb_end;
  
  bool * no_good_str;
  
  real * cuda_set_strs;
  bool * cuda_no_good_str;
  short int * cuda_conflicts;
  
  void host_init_data();
  void memcopy_host_to_device();
  void memcopy_device_to_host();
  
public:
  ConstraintCG(Fragment *f_ptr) {
    weight = 3;
    type = __c_cg;
    vfrag.push_back(f_ptr);
    dbg = "#log: Constraint CG - ";
  }//-
  
  ~ConstraintCG() { vfrag.clear(); }

  bool propagate();
  bool consistency();
  
  void dump();
};//ConstraintCG

#endif
