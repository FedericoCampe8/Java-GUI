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
#include "typedefs.h"
#include "statistics.h"
#include <sys/time.h>
#include <cmath>

Statistics::Statistics() :
  solutions_found(0),
  computed_solutions(0),
  cuda_calls(0),
  t_search_limit(0),
  t_total_limit(0),
  cuda_partial_time(-1) {
    for (uint i = 0; i < t_stat_size; i++) {
      time_start[i] = 0;
      time[i] = 0;
      total_time[i] = 0;
    }
    cudaEventCreate(&start_kernel);
    cudaEventCreate(&stop_kernel);
}//-  

Statistics::~Statistics() {
  cudaEventDestroy(start_kernel);
  cudaEventDestroy(stop_kernel);
}//-

void 
Statistics::reset () {
  solutions_found = 0;
  computed_solutions = 0;
  cuda_calls = 0;
  for (uint i = 0; i < t_stat_size; i++) {
    time_start[i] = 0;
    time[i] = 0;
    total_time[i] = 0;
  }
}//reset


void 
Statistics::incr_soluions_found(uint n) {
  solutions_found += n;
}//incr_soluions_found

void
Statistics::incr_computed_solutions(uint n) {
  computed_solutions += n;
}//incr_soluions_found

void
Statistics::incr_cuda_calls(long double n) {
  cuda_calls += n;
}//incr_cuda_calls

size_t 
Statistics::get_solutions_found() {
  return solutions_found;
}//get_solutions_found

size_t
Statistics::get_computed_solutions() {
  return computed_solutions;
}//get_solutions_found

long double
Statistics::get_cuda_calls() {
  return cuda_calls;
}//get_cuda_calls

void 
Statistics::set_search_timeout (double sec) {
  t_search_limit = sec;
}//set_search_timeout 

void 
Statistics::set_total_timeout (double sec) {
  t_total_limit = sec;
}//set_total_timeout

void 
Statistics::set_timer (t_stats t) {
  if (t == t_cuda || t == t_all_distant || t == t_cg || t == t_fragment_prop || t == t_energy || t == t_rmsd) {
    cuda_calls += 1;
    cudaEventRecord(start_kernel,0);
  }
  else {
    gettimeofday(&time_stats[t], NULL);
    time_start[t] = time_stats[t].tv_sec + (time_stats[t].tv_usec/1000000.0);
  }
}//set_timer

void
Statistics::force_set_time(t_stats t) {
  time[t] = t;
}//force_set_time

void
Statistics::set_time_cuda() {
  cudaEventRecord(start_kernel,0);
}//set_time_cuda

float
Statistics::get_time_cuda() {
  float elapsed_time;
  cudaEventRecord(stop_kernel, 0);
  cudaEventSynchronize(stop_kernel);
  cudaEventElapsedTime(&elapsed_time, start_kernel, stop_kernel);
  return elapsed_time;
}//get_time_cuda

double 
Statistics::get_timer (t_stats t) {
  return time[t]; 
}//get_timer

double 
Statistics::get_total_timer (t_stats t) { 
  return total_time[t];
}//get_total_timer

void 
Statistics::stopwatch (t_stats t) {
  if (t == t_cuda || t == t_all_distant || t == t_cg || t == t_fragment_prop || t == t_energy || t == t_rmsd) {
    cuda_partial_time = 0;
    cudaEventRecord(stop_kernel, 0);
    cudaEventSynchronize(stop_kernel);
    cudaEventElapsedTime(&cuda_partial_time, start_kernel, stop_kernel);
    time[t] += cuda_partial_time;
    if (t != t_cuda) time[t_cuda] += cuda_partial_time;
  }
  else {
    gettimeofday(&time_stats[t], NULL);
    time[t] = time_stats[t].tv_sec + (time_stats[t].tv_usec/1000000.0) - time_start[t];
    if (t == t_search) // for t_search we use stopwatch without setting the initial time
      total_time[t] = time[t];
    else
      total_time[t] += time[t];
  }
}//stopwatch

bool 
Statistics::timeout () {
  stopwatch(t_search);
  if (total_time[t_search] >= t_total_limit)
    return true;
  return false;
}//timeout

void 
Statistics::dump() {
  std::cout << "***************** Statistics *****************" << std::endl;
  std::cout << "- Total Search time: " << time[t_search] << " sec." << std::endl;
  std::cout << "- CUDA calls: " << cuda_calls << " sec." << std::endl;
  std::cout << "- CUDA total time: " << time[t_cuda] / 1000 << " sec." << std::endl;
  std::cout << "- CUDA Frg. prop time: " << time[t_fragment_prop] / 1000 << " sec." << std::endl;
  std::cout << "- CUDA Alldistand time: " << time[t_all_distant] / 1000 << " sec." << std::endl;
  std::cout << "- CUDA CG time: " << time[t_cg] / 1000 << " sec." << std::endl;
  std::cout << "- CUDA Energy time: " << time[t_energy] / 1000 << " sec." << std::endl;
  std::cout << "- CUDA RMSD time: " << time[t_rmsd] / 1000 << " sec." << std::endl;
  std::cout << "**********************************************" << std::endl;
}//-
