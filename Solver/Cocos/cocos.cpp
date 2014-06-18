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
#include "globals.h"
#include "input_data.h"
#include "logic_variables.h"
#include "supervisor.h"
#include "utilities.h"

using namespace std;
using namespace Utilities;

int main ( int argc, char* argv[] ) {
  string dbg = "#log: Main - ";

  /***************************************
   *         INIT DATA STRUCTURES        *
   ***************************************/
  cout << dbg << "Initialize Data...\n";
  Input_data i_data( argc, argv );
  
  /***************************************
   *           LOGIC VARIABLES           *
   ***************************************/
  cout << dbg <<
  "Generating Logic Variables...\n";
  g_logicvars.init_logic_variables ();
  
  /***************************************
   *           CREATING AGENTS           *
   ***************************************/
  Supervisor supervisor_agt;
  
  /***************************************
   *             CONSTRAINTS             *
   ***************************************/
  cout << dbg <<
  "Setting Constraints on Variables..." << endl;
  /// CONSTRAINT: SANG/RANG
  set_search_labeling_strategies ();
  /// CONSTRAINT: ALL_DISTANT
  set_all_distant_constraint ();
  /// CONSTRAINT: DISTANCE
  //set_distance_constraint ();
  /// CONSTRAINT: CG
  //set_centroid_constraint();
  /// Alloc constraints
  i_data.alloc_constraints();

  /***************************************
   *               LABELING              *
   ***************************************/
  timeval time_stats;
  double time_start, total_time;
  gettimeofday(&time_stats, NULL);
  time_start = time_stats.tv_sec + (time_stats.tv_usec/1000000.0);
  
  cout << dbg << "Start Search..." << "\n";
  supervisor_agt.search();
  cout << dbg << "...end Search" << "\n";
  
  gettimeofday(&time_stats, NULL);
  total_time = time_stats.tv_sec + (time_stats.tv_usec/1000000.0) - time_start;
  /// Print info about search
  if (gh_params.follow_rmsd) {
    cout << dbg << "RMSD       : " << gh_params.minimum_energy << "\n";
  }
  else {
    cout << dbg << "Energy     : " << gh_params.minimum_energy << "\n";
  }
  cout << dbg << "Search time: " << total_time << " sec.\n";
  
  /***************************************
   *           CLEAR AND EXIT            *
   ***************************************/
  cout << dbg << "Freeing memory" << endl;
  i_data.clear_data();
  /*
  g_logicvars.clear_variables();
  for ( int i = 0; i < g_constraints.size(); i++ )
    delete g_constraints[i];
   g_constraints.clear();
   supervisor_agt.~Supervisor();
  */
  cout << dbg << "Memory freed." << endl;
  cout << dbg << "Exit from COCOS... \n";

  return 0;
}//main
