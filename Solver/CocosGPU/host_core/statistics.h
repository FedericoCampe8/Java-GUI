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
#ifndef COCOS_STATISTICS_H
#define COCOS_STATISTICS_H

#include "globals.h"
#include "typedefs.h"

class Statistics {
 private:
  size_t solutions_found;
  size_t computed_solutions;
  timeval time_stats[t_stat_size];  
  double t_search_limit;
  double t_total_limit;
  double time_start[t_stat_size];
  double time[t_stat_size];
  double total_time[t_stat_size];
  cudaEvent_t start_kernel, stop_kernel;
  long double cuda_calls;
  float cuda_partial_time;
  
 public:
  Statistics();
  ~Statistics();

  void reset(); 
  
  void incr_soluions_found(uint n = 1);
  void incr_computed_solutions(uint n = 1);
  void incr_cuda_calls(long double n = 1);
  
  size_t get_solutions_found();
  size_t get_computed_solutions();
  long double get_cuda_calls();

  void set_search_timeout (double sec);
  void set_total_timeout (double sec);
  void set_timer (t_stats t);
  void force_set_time(t_stats t);
  void set_time_cuda();
  float get_time_cuda();

  double get_timer (t_stats t);
  double get_total_timer (t_stats t);
  void stopwatch (t_stats t);
  bool timeout ();  
  void dump();
};

#endif
