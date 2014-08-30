#include "docking.h"
#include "energy.h"
#include "mathematics.h"
#include "propagator.h"
#include "utilities.h"
#include "logic_variables.h"
#include "rmsd_fast.h"

using namespace std;

DOCKING::DOCKING ( MasAgent* mas_agt ) :
MONTECARLO ( mas_agt ),
_energy_value ( 0 ),
_n_seeds ( 0 ),
_n_total_sols ( 0 ) {
  srand ( time( NULL ) );
}//-

DOCKING::~DOCKING() {
}//-

void
DOCKING::set_parameters ( std::vector <  std::vector < real > >& coords ) {
  _centers_coords = coords;
}//set_parameters

void
DOCKING::set_parameters ( real x, real y, real z, real radius, real height ) {
  vector < real > coords;
  coords.push_back( x );
  coords.push_back( y );
  coords.push_back( z );
  coords.push_back( radius );
  coords.push_back( height );
  
  _centers_coords.push_back ( coords );
}//set_parameters

void
DOCKING::search () {
  string dbg = "#log: DOCKING::search - ";
  timeval time_stats;
  double time_start, total_time;
  int partitions;
  point starting_point;
  for ( auto& coords: _centers_coords ) {
    _center_x = coords[ 0 ];
    _center_y = coords[ 1 ];
    _center_z = coords[ 2 ];
    _radius   = coords[ 3 ];
    _oc_tree_height = coords[ 4 ];       /// 8^_oc_tree_height samplings
    _side           = 1.41421 * _radius; /// 2R = L * sqrt(2)
    partitions      = pow ( 2.0, (double)_oc_tree_height );
    _oc_tree_side   = _side / partitions;
    
    // Force a single computation when TRANSLATE constraint has been activated
    if ( gh_params.translate_str ) {
      partitions = 1;
      _oc_tree_side = 0;
    }
    
#ifdef DOCKING_SEARCH_DBG
    cout << dbg<< "Radius: " << _radius << " Height " << _oc_tree_height
    << " Side " << _side << " Oc-Tree Side " << _oc_tree_side << " Partitions " << partitions << endl;
    getchar();
#endif
    /// Explore the cube
    starting_point[ 0 ] = _center_x - (_side / 2);
    starting_point[ 1 ] = _center_y - (_side / 2);
    starting_point[ 2 ] = _center_z - (_side / 2);
    for ( int i = 0; i < partitions; i++ ) {
      for ( int j = 0; j < partitions; j++ ) {
        for ( int z = 0; z < partitions; z++ ) {
          _n_seeds++;
          
          gettimeofday ( &time_stats, NULL );
          time_start = time_stats.tv_sec + (time_stats.tv_usec/1000000.0);
          
          
          /// Reset search to (re)start with MonteCarlo Sampling
          MONTECARLO::reset();
          _to_print = false;
          _failed_constraints = MAX_ENERGY;
          
#ifdef DOCKING_SEARCH_DBG
          cout << dbg << "Translate Peptide into point: "
          << starting_point[ 0 ] + (i * _oc_tree_side) + (_oc_tree_side/2) << " "
          << starting_point[ 1 ] + (j * _oc_tree_side) + (_oc_tree_side/2) << " "
          << starting_point[ 2 ] + (z * _oc_tree_side) + (_oc_tree_side/2) << endl;
#endif
          /// Translate peptide in the new center
          Utilities::translate_structure ( gd_params.curr_str,
                                           1,
                                           starting_point[ 0 ] + (i * _oc_tree_side) + (_oc_tree_side/2),
                                           starting_point[ 1 ] + (j * _oc_tree_side) + (_oc_tree_side/2),
                                           starting_point[ 2 ] + (z * _oc_tree_side) + (_oc_tree_side/2),
                                           gh_params.n_res * 5 );
          /// MonteCarlo sampling for finding new configurations inside the dock
          MONTECARLO::search();
          /// Set minimum and print if an improving structure has been found
          if ( SearchEngine::get_local_minimum() < _energy_value ) {
            _energy_value = SearchEngine::get_local_minimum();
            // If verbose, print best solutions
            if ( _to_print || gh_params.verbose  ) {
              cout << dbg << "Best solution " << _energy_value << " (" << _failed_constraints << "):\n";
              /// Set result as global result
              //g_logicvars.set_point_variables ( gd_params.curr_str );
              /// Print solution
              g_logicvars.print_point_variables();
            }
          }
          gettimeofday( &time_stats, NULL );
          total_time = time_stats.tv_sec + (time_stats.tv_usec/1000000.0) - time_start;
          if ( gh_params.verbose ) {
            cout << dbg
            << MONTECARLO::get_n_sols() << " solutions in " << total_time << " sec.\n";
          }
          _n_total_sols += MONTECARLO::get_n_sols();
        }//z
      }//j
    }//i
  }//coord
  if ( gh_params.verbose ) {
    cout << dbg
    << "Total of " << _n_seeds << " seeds and " << _n_total_sols << " solutions\n";
  }
}//search


int
DOCKING::choose_label ( WorkerAgent* w ) {
  int best_label = -1;
  int n_threads  = 32;
  int v_id       = w->get_variable()->get_id();
  int n_blocks   = gh_params.set_size;
  int smBytes    = ( gh_params.n_points + 2 * gh_params.n_res ) * sizeof(real);
  
  while ( n_threads < gh_params.n_res ) n_threads += 32;
  n_threads = n_threads*2 + 32;
  
  _energy_function->calculate_energy ( gd_params.beam_str, gd_params.beam_energies,
                                      gd_params.validity_solutions, gh_params.n_res,
                                      _mas_bb_start, _mas_bb_end,
                                      _mas_scope_first, _mas_scope_second,
                                      smBytes, n_blocks, n_threads );
  
  /// Copy Energy Values
  memcpy ( gh_params.beam_energies, gd_params.beam_energies, n_blocks * sizeof( real ) );
  /// Select best label
  real truncated_number = MAX_ENERGY;
  bool improving_consistent_str = false;
  for( int i = 0; i < n_blocks; i++ ) {
    if( Math::truncate_number( gh_params.beam_energies[ i ] ) < truncated_number ) {
      /* Verify hard contact constriants
       * before setting the current solution as best solution found so far. */
      best_label       = i;
      truncated_number = gh_params.beam_energies[ best_label ];
      improving_consistent_str = true;
      bool hard_cons = verify_conditions ( i );
      
      if ( gh_params.known_protein != nullptr ) {
        real rmsd_value = Rmsd_fast::get_rmsd( &gd_params.beam_str[ i * gh_params.n_points ],
                                               gh_params.known_bb_coordinates,
                                               gh_params.n_res, 0, gh_params.n_res - 1 );
        if ( rmsd_value <= 1.2 ) {
          cout << "RMSD: " << rmsd_value << endl;
          g_logicvars.set_point_variables ( &gd_params.beam_str[ i * gh_params.n_points ] );
          // Print solution
          g_logicvars.print_point_variables();
          getchar();
        }
        
      }
      
      if ( (gd_params.validity_solutions[ i ] < 0.13) && hard_cons ) {
        _to_print = true;
        /*
          cout << "All_distant " << gd_params.validity_solutions[ i ] <<
          " Val " << _local_current_minimum << "(" << gh_params.beam_energies[ best_label ] << ")" << endl;
        */
        _n_sols++;
        // Set current structure
        if ( gd_params.validity_solutions[ i ] < _failed_constraints ) {
             _failed_constraints = gd_params.validity_solutions[ i ];
             g_logicvars.set_point_variables ( &gd_params.beam_str[ i * gh_params.n_points ] );
            // Print solution
            //g_logicvars.print_point_variables();
          }
        }
    }
    // Print "intermediate" structures (i.e., ith structure) when verbose
    print_str_step_by_step ( i );
  }//i
  
  if ( (gh_params.beam_energies[ best_label ] < _local_current_minimum) && improving_consistent_str ) {
    cout << "Best " << gh_params.beam_energies[ best_label ] <<  " local " << _local_current_minimum << endl;
    
    /// At least one improving structure
    _changed               = true;
    _best_agent            = _last_idx_sel;
    _best_label            = best_label;
    _best_wa               = w;
    _local_current_minimum = gh_params.beam_energies[ best_label ];
    /// Store local best structure
    memcpy ( _curr_best_str,
             &gd_params.beam_str[ best_label * gh_params.n_points ],
             gh_params.n_points * sizeof( real ) );
    /// Take the best structure among all possible samplings
    if ( _local_current_minimum <  _glb_current_minimum ) {
      _glb_current_minimum = _local_current_minimum;
      memcpy ( _glb_best_str, _curr_best_str,
              gh_params.n_points * sizeof( real ) );
    }
  }
  
  /// Try worst choices if not exit asap and everything is ground
  if ( (!_changed) && (!_exit_asap) && all_ground() ) {
    // All ground: select random variables to label (next loop iteration)
    _var_selection = 1;
    
    ///Find another admissible structure
    int best_label_aux = 0;
    
    real truncated_number =  Math::truncate_number( gh_params.beam_energies[ best_label_aux ] );
    for( int i = 1; i < n_blocks; i++ ) {
      if( (Math::truncate_number( gh_params.beam_energies[ i ] ) < truncated_number) &&
          (Math::truncate_number( gh_params.beam_energies[ i ] ) > gh_params.beam_energies[ best_label ]) ) {
        best_label_aux = i;
        truncated_number = gh_params.beam_energies[ best_label_aux ];
      }
    }
    
    /*
    cout << "-----------> best_label_aux " << best_label_aux <<  " val "
    << gh_params.beam_energies[ best_label_aux ] << endl;
    */
    
    /// Set new structure evaluating probability and temperature
    if ( gh_params.beam_energies[ best_label_aux ] < MAX_ENERGY ) {
      assign_with_prob ( best_label_aux, w, 1.1 );
    }
    
  }//Try worst structure
  
  return 0;
}//choose_label

void
DOCKING::print_str_step_by_step ( int pos ) {
  if ( ( (gh_params.verbose) && (gh_params.sys_job == ab_initio) ) ||
      (  (gh_params.verbose) && (gh_params.sys_job == docking) && verify_conditions( pos ) ) )  {
    /// Increment number of solutions
    _n_sols++;
    /// Set current structure
    g_logicvars.set_point_variables ( &gd_params.beam_str[ pos * gh_params.n_points ] );
    /// Print solution
    g_logicvars.print_point_variables();
  }
}//print_str_step_by_step

bool
DOCKING::verify_conditions ( int pos ) {
  /// Check if there is a minimum number of contacts.
#ifdef MONTECARLO_DEBUG
  if ( Math::truncate_number( gh_params.beam_energies[ pos ]) < -gh_params.min_n_contacts ) {
    cout << "Num of contacts " << Math::truncate_number( gh_params.beam_energies[ pos ]) <<
    " Required " << -gh_params.min_n_contacts << endl;
  }
#endif

  bool is_good = true;
  if ( is_good && (gh_params.force_contact.size() > 0) ) {
    // Check hard constraints
    // Points
    point pep_point;
    point dock_point;
    // Distances
    real min_bound, max_bound, distance;
    // Dock atom
    int atom_idx;
    
    for ( int idx = 0; idx < gh_params.force_contact.size(); idx++ ) {
      min_bound = gh_params.force_contact[ idx ][ 0 ];
      max_bound = gh_params.force_contact[ idx ][ 1 ];
      atom_idx  = (int) gh_params.force_contact[ idx ][ 2 ];
      
      pep_point[ 0 ]  =  gd_params.beam_str[ pos * gh_params.n_points + (( atom_idx - 1) * 3) + 0 ];
      pep_point[ 1 ]  =  gd_params.beam_str[ pos * gh_params.n_points + (( atom_idx - 1) * 3) + 1 ];
      pep_point[ 2 ]  =  gd_params.beam_str[ pos * gh_params.n_points + (( atom_idx - 1) * 3) + 2 ];
      
      dock_point[ 0 ] = gh_params.force_contact[ idx ][ 3 ];
      dock_point[ 1 ] = gh_params.force_contact[ idx ][ 4 ];
      dock_point[ 2 ] = gh_params.force_contact[ idx ][ 5 ];
      
      distance = Math::eucl_dist( pep_point, dock_point );
      
#ifdef MONTECARLO_DEBUG
      cout << "min " << min_bound << " max " << max_bound << " idx " << atom_idx << " x: " <<
      pep_point[ 0 ] << " y: " << pep_point[ 1 ] << " z: " << pep_point[ 2 ] <<
      " vs x: " <<
      dock_point[ 0 ] << " y: " << dock_point[ 1 ] << " z: " << dock_point[ 2 ] << endl;
      cout << "distance " << distance << endl;
      getchar();
#endif
      
      // If structure is not good, return asap
      if ( !( (distance >= min_bound) &&
              (distance <= max_bound) ) ) {
        is_good = false;
        break;
      }
    }//idx
  }
  
  // Return checks
  return is_good;
}//verify_conditions
