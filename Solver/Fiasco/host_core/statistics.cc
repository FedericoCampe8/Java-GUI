#include "typedefs.h"
#include "statistics.h"
#include "globals.h"
#include "logic_variables.h"
#include "variable_fragment.h"
#include "protein.h"

#include <algorithm>
#include <sys/time.h>
#include <sys/types.h>
#include <cmath>
#include <stdlib.h> // for atoi
#include <stdio.h>
#include <unistd.h>

Statistics::Statistics(int argc, char* argv[]) 
  : backtracks                   ( 0 ),
    loop_search_space            ( 0.0 ),
    search_space_explored        ( 0.0 ),
    filtered_search_space        ( 0.0 ),
    numof_possible_conformations ( 0.0 ),
    solutions_found              ( 0 ),
    solutions_to_file            ( 0 ),
    t_search_limit               ( 0 ),
    t_total_limit                ( 0 ),
    energy                       ( 1000 ),
    result_is_improved           ( true ) {
  
  /* Process the input parameters */
  for (int narg = 0 ; narg < argc  ; narg++) {
    if (!strcmp ("--timeout-search",argv[narg])) {
      t_search_limit= atoi(argv[narg + 1]);      
    }//-
    if (!strcmp ("--timeout-total",argv[narg])) {
      t_total_limit= atoi(argv[narg + 1]);      
    }//-
  }

  avg_filtered_domain_elements.first = 0.0;
  avg_filtered_domain_elements.second = 0;
  clustering_avg_distance_error.first = 0.0;
  clustering_avg_distance_error.second = 0;

  rmsd[protein] = rmsd[p_loop] = 1000;
  
  for (uint i=0; i<constr_type_size; i++) {

    propagation_successes[i] = 0;
    propagation_failures[i] = 0;
  }
  for (uint i=0; i<prot_struct_size; i++)
    rmsd[i] = 1000;
  for (uint i=0; i<t_stat_size; i++) {
    time_start[i] = 0;
    time[i] = 0;
    total_time[i] = 0;
  }
  intersection = false;  
}//-  

Statistics::~Statistics() {
  if (loop_search_space_dim)
    delete[] loop_search_space_dim;
}//-

bool
Statistics::energy_is_improved () {
  return rmsd_is_improved();
}

bool
Statistics::rmsd_is_improved () {
  if ( result_is_improved ) {
    result_is_improved = false;
    return true;
  }
  else {
    return result_is_improved;
  }
}

void 
Statistics::reset () {
  backtracks = 0;
  search_space_explored = 0;
  //solutions_found = 0;
  //solutions_to_file = 0;
  filtered_search_space = 0;
  //filtered_search_space.second = 0;
  //loop_search_space.first = 0;
  //loop_search_space.second = 0;
  for (uint i=0; i<constr_type_size; i++) {
    propagation_successes[i] = 0;
    propagation_failures[i] = 0;
  }
  for (uint i=0; i<prot_struct_size; i++)
    rmsd[i] = 1000;
  for (uint i=0; i<t_stat_size; i++) {
    time_start[i] = 0;
    time[i] = 0;
    total_time[i] = 0;
  }
  intersection = false;  
}//-


// ok!
void
Statistics::new_loop_search_space (uint vf_s, uint vf_e) {
  uint dim = vf_e - vf_s +1; 
  loop_search_space_dim = new long double[dim+1];
  loop_search_space_dim[dim] = 1;
  numof_possible_conformations = 1;

  for (uint lidx = 0; lidx < dim; lidx++) {
    long double loop_search_space_size = 1;
    loop_search_space_dim[lidx] = 1;
     for (uint i = vf_s+lidx; i <= vf_e; i++) {
       loop_search_space_size *= g_logicvars->var_fragment_list[i].domain.size();
       loop_search_space_dim[lidx] += loop_search_space_size;
     }
     if (lidx == 0)
       numof_possible_conformations = loop_search_space_size;
  }
  
  loop_search_space = loop_search_space_dim[0];
}//-

long double 
Statistics::get_loop_search_space(uint lev) {
  return loop_search_space_dim[lev];
}//-

void 
Statistics::incr_soluions_found(uint n) {
  solutions_found += n;
}//-

size_t 
Statistics::get_solutions_found() {
  return solutions_found;
}//-

void 
Statistics::incr_solutions_tofile(uint n) {
  solutions_to_file += n;
}//-

size_t 
Statistics::get_solutions_tofile() {
  return solutions_to_file;
}//-
  
void 
Statistics::incr_backtracks(uint n) {
  backtracks += n;
}//-

void 
Statistics::incr_propagation_successes(constr_type c) {
  propagation_successes[c]++;
}//-

void 
Statistics::incr_propagation_failures(constr_type c) {
  propagation_failures[c]++;
}//-

// This function is expecting a number that have already been trasnfomed
// in the reduced loop search space measure /= pow(10, loop_search_space.second)
void 
Statistics::incr_filtered_search_space (long double reduced_increment) {
  filtered_search_space += reduced_increment;
}//-

void 
Statistics::decr_search_space_filtered (long double reduced_increment) {
  filtered_search_space -= reduced_increment; 
}//-

void 
Statistics::incr_search_space_explored(size_t n) {
  search_space_explored += n;
}//-

void 
Statistics::decr_search_space_explored(size_t n) {
  search_space_explored -= n;
}//-

void 
Statistics::set_loop_search_space_size(long double n) {
  loop_search_space  = n;
}//-

void 
Statistics::set_numof_possible_conformations(long double n) {
  numof_possible_conformations  = n;
}//-

void 
Statistics::set_rmsd (prot_struct_type t, real r) {
  rmsd[t] = r;
  _RMSD_ensemble.push_back( r );
}//-

void
Statistics::set_best_energy ( real e ) {
  if ( e < energy ) {
    result_is_improved = true;
    energy = e;
    stopwatch (t_search);
  }
}

void
Statistics::set_best_rmsd ( prot_struct_type t, real r ) {
  if ( r < rmsd[t] ) {
    result_is_improved = true;
    rmsd[t] = r;
    stopwatch (t_search);
    /*
    std::cout << "STATISTICS - New Best Loop RMSD found "
	      << "[time: " << get_timer (t_search)
	      << " s.] / Best Loop RMSD: " << get_rmsd(p_loop) 
	      << std::endl;
     */
  }
}//-

real 
Statistics::get_rmsd (prot_struct_type t) {
  return rmsd[t];
}//-

real
Statistics::get_energy () {
  return energy;
}//-

std::vector<real>
Statistics::get_rmsd_ensemble ( ) 
{
  std::sort( _RMSD_ensemble.begin(), _RMSD_ensemble.end() );
  return _RMSD_ensemble;
}

void 
Statistics::set_search_timeout (double sec) {
  t_search_limit=sec;
}//-

void 
Statistics::set_total_timeout (double sec) {
  t_total_limit=sec;
}//-

void 
Statistics::set_timer (t_stats t) {
  gettimeofday(&time_stats[t], NULL);
  time_start[t] = time_stats[t].tv_sec+(time_stats[t].tv_usec/1000000.0);
}//-

void
Statistics::force_set_time(t_stats t) {
  time[t] = t;
}

double 
Statistics::get_timer (t_stats t) { 
  return time[t]; 
}//-

double 
Statistics::get_total_timer (t_stats t) { 
  if (t != t_statistics)
    return (total_time[t] - total_time[t_statistics]);
  else
    return total_time[t];
}//-

void 
Statistics::stopwatch (t_stats t) {
  gettimeofday(&time_stats[t], NULL);
  time[t] = time_stats[t].tv_sec+(time_stats[t].tv_usec/1000000.0) - time_start[t]; 
  if (t == t_search) // for t_search we use stopwatch without setting the initial time
    total_time[t] = time[t];
  else
    total_time[t] += time[t];
}//-

bool 
Statistics::timeout () {
  if (t_total_limit <= 0) 
    return false;
  stopwatch(t_search);
  if (total_time[t_search] >= t_total_limit)
    return true;
  return false;
}//-

bool 
Statistics::timeout_searchtime_only () {
  if (t_search_limit <= 0) 
    return false;
  stopwatch(t_search);
  if ( (total_time[t_search] - total_time[t_jm]) >= t_search_limit)
    return true;
  return false;
}//-

void
Statistics::incr_clustering_avg_distance_error (real clusters_avg_distance) {
  clustering_avg_distance_error.first += clusters_avg_distance;
  clustering_avg_distance_error.second ++;
}//-

real 
Statistics::get_clustering_avg_distance_error () {
  return (clustering_avg_distance_error.first / clustering_avg_distance_error.second); 
}//-

void
Statistics::incr_avg_filtered_domain_elements (real avg_element_filtered) {
  avg_filtered_domain_elements.first  += avg_element_filtered;
  avg_filtered_domain_elements.second ++;
}//-

real 
Statistics::get_avg_filtered_domain_elements () {
  return (avg_filtered_domain_elements.first / avg_filtered_domain_elements.second); 
}//-


size_t
Statistics::getMaxRSS() const {
  int len = 0; 
  int srtn = 0;
  char procf[257] = { "" };
  FILE *fp = NULL;
  char line[2001] = { "" };
  char crap[2001] = { "" };
  char units[2001] = { "" };
  size_t maxrss = 0L;
  size_t maxrsskb = 0L;

  sprintf(procf,"/proc/%d/status",getpid());
  
  fp = fopen(procf, "r");
  if(fp == NULL){
    return -1;
  }
  
  while(fgets(line, 2000, fp) != NULL){
    if(strncasecmp(line,"VmPeak:",7) == 0){
      len = (int)strlen(line);
      line[len-1] = '\0';
      srtn = sscanf(line,"%s%ld%s",crap,&maxrss,units);
      if(srtn == 2){
        maxrsskb = maxrss / 1024L;
      }else if(srtn == 3){
        if( (strcasecmp(units,"B") == 0) || (strcasecmp(units,"BYTES") == 0) ){
          maxrsskb = maxrss / 1024L;
        }else if( (strcasecmp(units,"k") == 0) || (strcasecmp(units,"kB") == 0) ){
          maxrsskb = maxrss * 1L;
        }else if( (strcasecmp(units,"m") == 0) || (strcasecmp(units,"mB") == 0) ){
          maxrsskb = maxrss * 1024L;
        }else if( (strcasecmp(units,"g") == 0) || (strcasecmp(units,"gB") == 0) ){
          maxrsskb = maxrss * 1024L * 1024L;
        }else{
          maxrsskb = maxrss * 1L;
        }
      }
      break;
    }
  }
  
  fclose(fp);
  
  return maxrsskb;
}



void 
Statistics::dump (std::ostream &os) {

  os << "\t============ FIASCO statistics ============\n";
  os << "Protein Name: "
     << g_target.get_name() << std::endl;
  os << "COMPUTATION TIME" << std::endl
     << "  JMf Propagation Time            :  " 
     << get_timer (t_jm)        << " s.\n"
     << "  Avg clustering time/1000 nodes  :  "
     << get_timer (t_jm)/1000   << " s.\n"
     << "  Search Time                     :  "
     << get_timer (t_search) - get_timer (t_jm)  << " s.\n"
     << "  Total Time                      :  " 
     << get_timer (t_search) << " s.\n";
  
  os << "SEARCH SPACE PROERTIES" << std::endl
     << "  No. nodes [explorable/explored] : [" 
     << get_loop_search_space_size()  << " / " 
     << get_search_space_explored()         <<  "]" << std::endl 
     << "  No. nodes Filtered (JMf)        :  " 
     << get_loop_search_space_size() - get_solutions_found() <<  std::endl
     // << get_filtered_search_space()   << std::endl
     << "  Ratio_space_filtered (JMf)      :  "
     << (get_loop_search_space_size() - get_solutions_found()) / get_loop_search_space_size()  << std::endl;
     // << "  Avg domain size filtered        :  "
     // << get_avg_filtered_domain_elements()  << std::endl
     // << "  Avg clustering distance error   :  "
     // << get_clustering_avg_distance_error() << std::endl;

  os << "CONFORMATIONS " << std::endl 
     /*
     << "  Best Loop RMSD                  :  "
     << get_rmsd (p_loop) << std::endl
     << "  Best Prot RMSD                  :  "
     << get_rmsd (protein) << std::endl
     */
     << "  Best Energy                     :  "
     << get_energy() << std::endl
     << "  No. leaves [possible/reached]   : [" 
     << get_numof_possible_conformations() << " / "
     << get_solutions_found() << "]" << std::endl;

#ifdef VERBOSE 
  for (int i=0; i<constr_type_size; i++) {
    os << "\tpropagation_successes ["<<i<<"]: " <<propagation_successes[i]<< std::endl;
  }
  for (int i=0; i<constr_type_size; i++) {
    os << "\tpropagation_failures ["<<i<<"]: " <<propagation_failures[i] << std::endl;
  }
#endif
  os << std::endl;
}//-
