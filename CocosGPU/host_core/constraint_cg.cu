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
#include "constraint_cg.h"
#include "cuda_cg.h"

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <stdlib.h>
#include <algorithm>

#include "globals.h"
#include "fragment.h"
#include "statistics.h"
#include "protein.h"
#include "structure_agent.h"
#include "worker_agent.h"

using namespace std;
using namespace Utilities;

//#define CG_DEBUG

bool
ConstraintCG::propagate() {
  return true;
}//propagate

bool
ConstraintCG::consistency () {
  if (vfrag[0]->worker_agent->structure_agent->get_type() != OTHER) return true;
  
  int blocks, threads;
  int smBytes;
  
  n_tertiary_atoms = (int) g_target.get_bblen();
  d_size = vfrag[0]->worker_agent->domain_size;
  smBytes = 3 * n_tertiary_atoms * sizeof(real);
  
  int num_of_str = vfrag[0]->worker_agent->structure_agent->num_of_str;
  int num_threads = min(512, (num_of_str/d_size));
  if (num_threads*d_size > MAX_NUM_STR) num_threads = (MAX_NUM_STR / d_size);
  if (MAX_NUM_STR < d_size) num_threads = 1;
  d_size =  d_size * num_threads;

  blocks = d_size;
  threads = g_target.get_nres()-2;
  
#ifdef CG_DEBUG
  cout << "All distant blocks " << d_size << endl;
#endif
  
  //Init data on host
  host_init_data();

#ifdef CUDA_STATISTICS
  g_statistics.set_timer(t_cg);
#endif

  /* Kernel call */
  cuda_cg_consistency<<<blocks, threads, smBytes>>>(cuda_set_strs, cuda_no_good_str, cuda_g_aa_seq,
                                                    cuda_g_type_seq, bb_start, bb_end, n_tertiary_atoms);
  
  cudaDeviceSynchronize();
//  getchar();
#ifdef CUDA_STATISTICS
  g_statistics.stopwatch(t_cg);
#endif

  //Copy results on local vector
  memcopy_device_to_host();
  
#ifdef CG_DEBUG
  cout << "Checking all distant CG for:" << endl;
  cout << vfrag[0]->get_id() << endl;
  cout << "from " << bb_start << " to " << bb_end << endl;
  for (int i = 0; i < blocks; i++) {
    cout << "i " << i << ": ";
    if (no_good_str[i])
      cout << "OK" << endl;
    else
      cout << "NO" << endl;
  }
  getchar();
#endif
  
  //If !at_least_one valid -> backtrack
  bool at_least_one = false;
  for (int i = 0; i < blocks; i++) {
    at_least_one = at_least_one || no_good_str[i];
  }

  if (!at_least_one) return false;
  
  return true;
}//consistency

void
ConstraintCG::host_init_data() {
  
  /*
   * @todo:
   * Do this only one time, when I create the constraint.
   * Nothe: pay attention to shared fragments.
   */
  no_good_str = vfrag[0]->worker_agent->structure_agent->no_good_strs;
  
  cuda_set_strs    = vfrag[0]->worker_agent->structure_agent->cuda_structures;
  cuda_no_good_str = vfrag[0]->worker_agent->structure_agent->cuda_no_good_strs;
  
  bb_start = 0;
  bb_end = 4*g_target.get_nres();
}//host_init_data

void
ConstraintCG::memcopy_host_to_device() {
  cudaMemcpy(cuda_no_good_str, no_good_str, d_size * sizeof(bool),
             cudaMemcpyHostToDevice);
}//memcopy_host_to_device

void
ConstraintCG::memcopy_device_to_host() {
  cudaMemcpy(no_good_str, cuda_no_good_str, d_size * sizeof(bool),
             cudaMemcpyDeviceToHost);
}//memcopy_device_to_host

void 
ConstraintCG::dump() {
  cout << "Constraint Id  :" << get_id() << endl;
  cout << "           Type: CG" << endl;
  cout << "           Vars:";
  for (uint i = 0; i < vfrag.size(); i++)
    cout << " vF_" << vfrag.at(i)->get_id();
  cout << endl;
}//dump