#include "input_data.h"

#include "logic_variables.h"
#include "constraint.h"
#include "tors_corr_bmf.h"
#include "tors_bmf.h"
#include "utilities.h"
#include "atom.h"

//#define INPUT_DATA_DBG

using namespace std;
using namespace Utilities;

Input_data::Input_data ( int argc, char* argv[] ) :
_dbg ( "#log: Input_data - " ),
_know_prot ( false ),
_in_file ( "" ),
_out_file ( "" ),
_known_prot_file ( "" ),
_target_prot_file ( "" ) ,
_target_sequence ( "" ) {
  //Default values
  gh_params.gibbs_as_default          = false;
  gh_params.follow_rmsd               = false;
  gh_params.verbose                   = false;
  gh_params.centroid                  = false;
  gh_params.h_def_on_pdb              = true;
  gh_params.n_gibbs_samples           = -1;
  gh_params.n_coordinators            = 1;
  gh_params.set_size                  = MAX_GIBBS_SET_SIZE;
  gh_params.n_gibbs_iters_before_swap = 0;
  gh_params.timer                     = -1;
  gh_params.vars_to_shuffle           = NULL;
  gd_params.vars_to_shuffle           = NULL;
  
  // Process input
  int c;
  bool auto_allign = false;
  while ( (c = getopt(argc, argv, "i:o:c:g:t:sarhevdq")) != -1 ) {
    switch ( c ) {
      case 'i':
        /// Input file name
        _in_file = optarg;
        break;
      case 'o':
        /// Output file name (otherwise default one is used)
        _out_file = optarg;
        break;
      case 'c':
        /// Set clock timer for Montecarlo sampling
        gh_params.timer = atoi ( optarg );
      case 'r':
        /// Use RMSD ad objective function
        gh_params.follow_rmsd = true;
        break;
      case 'g':
        /// Set number of Gibbs samples
        gh_params.n_gibbs_samples = atoi ( optarg );
        break;
      case 'q':
        /// Use RMSD ad objective function
        gh_params.gibbs_as_default = true;
        break;
      case 'e':
        /// Automagically create an input file!
        gh_params.centroid = true;
        break;
      case 's':
        /// Set "set" size
        gh_params.set_size = atoi ( optarg );
        break;
      case 'd':
        /// Device info
        print_gpu_info();
        exit( 0 );
      case 't':
        /// Set iters before swapping bins
        gh_params.n_gibbs_iters_before_swap = atoi ( optarg );
        break;
      case 'a':
        /// Automagically create an input file!
        auto_allign = true;
        break;
      case 'v':
        /// Verbose
        gh_params.verbose = true;
        break;
      case 'h':
        /// Print help
        print_help();
        exit( 0 );
      default:
        print_help();
        exit( 0 );
    }
  }
  
  /// Cuda flags
  cudaSetDeviceFlags ( cudaDeviceScheduleBlockingSync );
  /// Parse and get info from the input file
  if ( _in_file == "" ) {
    cout << "Use \"-i\" for input file\n";
    exit( 0 );
  }
  
  if ( auto_allign ) {
    create_input_file ();
    _in_file = "alignment.txt";
  }
  
  read_file ();
  /// Init data
  init_data ();
  /// Print Global minimum
  if ( _know_prot ) {
    cout << _dbg << "Native Protein global minimum: " <<
    gh_params.known_protein->get_minium_energy() << endl;
  }
}//-

Input_data::~Input_data () {
}//-

void
Input_data::clear_data () {
  // Free memory on Host
  Utilities::print_debug ( _dbg, "Free memory on host" );
  for ( int i = 0; i < gh_params.mas_des.size(); i++ ) {
    gh_params.mas_des[ i ].vars_list.clear();
    gh_params.mas_des[ i ].vars_bds.clear();
  }
  gh_params.mas_des.clear();
  gh_params.constraint_descriptions.clear();
  gh_params.constraint_descriptions_idx.clear();
  gh_params.constraint_events.clear();
  gh_params.domain_angles.clear();
  if ( _know_prot )
    gh_params.known_protein->~Protein();
  gh_params.target_protein->~Protein();
  
  HANDLE_ERROR( cudaFreeHost( gh_params.validity_solutions ) );
  HANDLE_ERROR( cudaFreeHost( gh_params.random_array ) );
  
  free ( gh_params.domain_states );
  free ( gh_params.h_distances );
  free ( gh_params.h_angles );
  free ( gh_params.contact_params );
  free ( gh_params.tors );
  free ( gh_params.tors_corr );
  free ( gh_params.beam_energies );
  if ( gh_params.follow_rmsd )
    free ( gh_params.known_bb_coordinates );
  
  /// Free memory aux
  free_dt ();
}//clear_data

void
Input_data::create_input_file () {
  /// Allign sequence
  char * cstr = new char [_in_file.length()+1];
  char * jent =
  strcpy (cstr, _in_file.c_str());
  
  pid_t pid=fork();
  if (pid==0) { /// child process
    static char *argv[]={"jnet","-p", cstr, NULL};
    execv("./bin_jnet/jnet",argv);
    exit(127); /// only if execv fails
  }
  else { /// pid!=0; parent process
    waitpid(pid,0,0); /// wait for child to exit
  }
  /// File alignment.txt created: open it and read the line
  ifstream inputFile;
  string buffer, line_fasta, line_allign, allign_input = "alignment.txt";
  char * fname = (char*) malloc ( (allign_input.size() + 1) * sizeof(char) );
  bool first_line = false;
  strcpy( fname, allign_input.c_str() );
  inputFile.open( fname );
  if( inputFile.is_open() ){
    getline ( inputFile, line_allign );
    /*
     while ( inputFile.good() ){
     getline ( inputFile, line_allign );
     }
     */
  }
  else {
    cout << _dbg << "unable to open " << allign_input << " " << endl;
    free( fname );
    exit( 1 );
  }
  free( fname );
  inputFile.close ();
  
  /* Old version - looks like ain't work with the rcsb.org files
  char * fname2 = (char*) malloc ( (_in_file.size() + 1) * sizeof(char) );
  strcpy( fname2, _in_file.c_str() );
  inputFile.open( fname2 );
  if( inputFile.is_open() ){
    while ( inputFile.good() ){
      getline ( inputFile, line_fasta );
    }
  }
  */

  char * fname2 = (char*) malloc ( (_in_file.size() + 1) * sizeof(char) );
     

  strcpy( fname2, _in_file.c_str() );
  inputFile.open( fname2 );
  if( inputFile.is_open() ){
    while ( inputFile.good() ){
      getline ( inputFile, buffer );
      if (buffer.length() != 0){
        if (buffer.compare( 0, 1, ">" ) != 0 ){
          first_line = true;
          line_fasta += buffer;
        }
        if ((buffer.compare( 0, 1, ">" ) == 0 ) && (first_line)){
          break;
        }
      }
    }
  }
  else {
    cout << _dbg << "unable to open " << _in_file << " " << endl;
    free( fname2 );
    exit( 1 );
  }
  free( fname2 );
  inputFile.close ();
  
  FILE *fid;
  /// Rewrite the input file for cocos
  fid = fopen ( allign_input.c_str(), "w" ); //Use append "a"
  if (fid < 0){
    printf( "Cannot open %s to write!\n", allign_input.c_str() );
    exit( 1 );
  }
  
  fprintf( fid, ">FASTA SEQUENCE\n" );
  fprintf( fid, "%s", line_fasta.c_str() );
  fprintf( fid, "\n");
  
  /// Analyze alligned string "line_allign"
  bool first_found = false, start_counting = false;
  int start_aa;
  int previous_sec = -1;
  int counter = 0;
  for ( int i = 0; i < line_allign.length(); i++ ) {
    counter++;
    if ( line_allign.compare( i, 1, "-") == 0 ) {
      if ( !first_found ) {
        counter = 0;
        first_found = true;
      }
      if ( start_counting ) {
        if ( previous_sec == 1 ) {
          fprintf( fid, "H %d %d\n", start_aa, counter-1 );
        }
        else if ( previous_sec == 2 ) {
          fprintf( fid, "S %d %d\n", start_aa, counter-1 );
        }
      }
      previous_sec = -1;
      start_counting = false;
      continue;
    }
    if ( line_allign.compare( i, 1, "H") == 0 ) {
      if ( !start_counting ) {
        start_counting = true;
        start_aa = counter;
        previous_sec = 1;
      }
      else {
        if ( previous_sec == 2 ) {
          fprintf( fid, "S %d %d\n", start_aa, counter-1 );
          start_aa = counter;
          previous_sec = 1;
        }
      }
    }
    else if ( line_allign.compare( i, 1, "E") == 0 ) {
      if ( !start_counting ) {
        start_counting = true;
        start_aa = counter;
        previous_sec = 2;
      }
      else {
        if ( previous_sec == 1 ) {
          fprintf( fid, "H %d %d\n", start_aa, counter-1 );
          start_aa = counter;
          previous_sec = 2;
        }
      }
    }
    else {
      //cout << " Not done yet!\n";
      continue;
    }
  }
  fprintf( fid, "\n");
  ///Close file
  fclose(fid);
}//create_input_file

void
Input_data::read_file () {
  string line;
  string token;
  char * pch;
  int value, start = 12;   // Value taken from the format of the input file
  int lw_priority = LOWER_PRIORITY/2;
  int energy_parameters_read = 0;
  
  ifstream inputFile;
  char * fname = (char*) malloc ( (_in_file.size() + 1) * sizeof(char) );
  strcpy( fname, _in_file.c_str() );
  inputFile.open( fname );
  if( inputFile.is_open() ){
    while ( inputFile.good() ){
      getline ( inputFile, line );
      /// Target on FASTA format
      if ( line.compare( 0, 1, ">" ) == 0 ) {
        getline ( inputFile, line );
        set_target_sequence ( line );
      }
      /// Target on pdb
      else if ( line.compare( 0, 10, "KNOWN_PROT" ) == 0 ) {
        _known_prot_file = line.substr( start, line.size() - start );
        _known_prot_file += ".pdb";
        _know_prot = true;
      }
      else if ( line.compare( 0, 11, "TARGET_PROT" ) == 0 ) {
        _target_prot_file = line.substr( start, line.size() - start );
        _target_prot_file += ".pdb";
      }
      /// Energy Tables
      else if ( line.compare( 0, 10, "COULOMBPAR" ) == 0 ) {
        _energy_charges = line.substr( start, line.size() - start );
        energy_parameters_read++;
      }
      else if ( line.compare( 0, 11, "LJPARAMETER" ) == 0 ) {
        _lj_params = line.substr( start, line.size() - start );
        energy_parameters_read++;
      }
      else if ( line.compare( 0, 11, "HDPARAMETER" ) == 0 ) {
        _h_distances = line.substr( start, line.size() - start );
        energy_parameters_read++;
      }
      else if ( line.compare( 0, 11, "HAPARAMETER" ) == 0 ) {
        _h_angles = line.substr( start, line.size() - start );
        energy_parameters_read++;
      }
      else if ( line.compare( 0, 7, "CONTACT" ) == 0 ) {
        _contact_params = line.substr( start, line.size() - start );
        energy_parameters_read++;
      }
      else if ( line.compare( 0, 7, "TORSPAR" ) == 0 ) {
        _tors_params = line.substr( start, line.size() - start );
        energy_parameters_read++;
      }
      else if (line.compare( 0, 6, "ANGLES" ) == 0 ) {
        _angles_file = line.substr(start, line.size() - start);
        energy_parameters_read++;
      }
      /// Secondary Structure Descriptions, Agents, and Priorities
      else if ( line.compare( 0, 2, "H " ) == 0 ||
               line.compare( 0, 2, "S " ) == 0 ||
               line.compare( 0, 2, "E " ) == 0 ||
               line.compare( 0, 2, "C " ) == 0 ||
               line.compare( 0, 2, "T " ) == 0 ||
               line.compare( 0, 2, "A " ) == 0 ) {
        MasAgentDes agt_description;
        char* char_line = (char*) malloc ( (line.size() + 1) * sizeof(char) );
        strcpy( char_line, line.c_str() );
        pch = strtok (char_line, " ,.-[]");
        
        int  first_v = -1;
        int  first_s = -1;
        agt_description.priority = -1;
        agt_description.quantum = MAX_QUANTUM;
        bool p_found = false; // Priority
        bool s_found = false; // Scope
        bool q_found = false; // Quantum of time
        ss_type  t_found = other;
        while ( pch != NULL ) {
          if ( !p_found && !s_found && !q_found ) {
            if ( pch[0] > 47 && pch[0] < 58 ) {
              value = atoi( pch );
              if ( first_v == -1 ) {
                first_v = value;
              }
              else {
                agt_description.vars_bds.push_back ( make_pair ( first_v, value ) );
                first_v = -1;
              }
            }
          }
          else if ( p_found ) {
            if ( pch[0] > 47 && pch[0] < 58 ) {
              agt_description.priority = atoi( pch );
              // At most one priority value
              p_found = false;
            }
          }
          else if ( q_found ) {
            if ( pch[0] > 47 && pch[0] < 58 ) {
              agt_description.quantum = atoi( pch );
              // At most one quantum value
              q_found = false;
            }
          }
          else if ( s_found ) {
            if ( pch[0] > 47 && pch[0] < 58 ) {
              value = atoi( pch );
              if ( first_s == -1 ) {
                first_s = value;
              }
              else {
                agt_description.scope.first  = first_s;
                agt_description.scope.second = value;
              }
            }
          }
          
          if ( pch[0] == 'p') p_found = true;
          if ( pch[0] == 's') s_found = true;
          if ( pch[0] == 'q') q_found = true;
          if ( pch[0] == 'T' ) t_found = turn;
          if ( pch[0] == 'C' ) t_found = coil;
          
          pch = strtok (NULL, " ,.-[]");
        }//while
        
        if ( (!p_found) && (agt_description.priority < 0) ) {
          agt_description.priority = lw_priority;
          lw_priority++;
        }
        if ( !s_found ) {
          agt_description.scope.first  = -1;
          agt_description.scope.second = -1;
        }
        
        if ( line.compare( 0, 2, "H " ) == 0 ) {
          agt_description.sec_str_type = helix;
          agt_description.agt_type = structure;
        }
        if ( (line.compare( 0, 2, "S " ) == 0) ||
            (line.compare( 0, 2, "E " ) == 0) ) {
          agt_description.sec_str_type = sheet;
          agt_description.agt_type = structure;
        }
        if ( line.compare( 0, 2, "C " ) == 0 ) {
          agt_description.sec_str_type = coil;
          agt_description.agt_type = coordinator;
        }
        if ( line.compare( 0, 2, "T " ) == 0 ) {
          agt_description.sec_str_type = turn;
          agt_description.agt_type = coordinator;
        }
        if ( line.compare( 0, 2, "A " ) == 0 ) {
          agt_description.sec_str_type = t_found;
          agt_description.agt_type = coordinator;
          gh_params.n_coordinators++;
        }
        for ( int i = 0; i < agt_description.vars_bds.size(); i++ ) {
          int first_aa = agt_description.vars_bds[ i ].first;
          int last_aa  = agt_description.vars_bds[ i ].second;
          for ( int j = first_aa; j <= last_aa ; j++ )
            agt_description.vars_list.push_back ( j );
        }
        /// Search strategy
        std::string key_icm ("icm");
        std::string key_montecarlo ("montecarlo");
        std::string key_gibbs ("gibbs");
        std::string key_complete ("complete");
        size_t found_icm        = line.rfind( key_icm );
        size_t found_montecarlo = line.rfind( key_montecarlo );
        size_t found_gibbs      = line.rfind( key_gibbs );
        size_t found_complete   = line.rfind( key_complete );
        bool something_found = false;
        if ( (found_icm != std::string::npos) && (!something_found) ) {
          agt_description.search_strategy = icm;
          something_found = true;
        }
        if ( (found_montecarlo != std::string::npos) && (!something_found) ) {
          agt_description.search_strategy = montecarlo;
          something_found = true;
        }
        if ( (found_gibbs != std::string::npos) && (!something_found) ) {
          agt_description.search_strategy = gibbs;
          something_found = true;
        }
        if ( (found_complete != std::string::npos) && (!something_found) ) {
          agt_description.search_strategy = complete;
          something_found = true;
        }
        
        if ( agt_description.agt_type == structure ) {
          agt_description.search_strategy = icm;
        }
        /// Default search strategy for coordinator agent: Montecarlo sampling
        if ( (!something_found) && (agt_description.agt_type == coordinator) ) {
          if ( gh_params.gibbs_as_default ) {
            agt_description.search_strategy = gibbs;
          }
          else {
            agt_description.search_strategy = montecarlo;
          }
        }
        /// Store agent description
        gh_params.mas_des.push_back ( agt_description );
        free (char_line);
      }
    }//while
    inputFile.close();
    if ( gh_params.verbose ) dump ();
  }
  else {
    cout << _dbg << "unable to open " << _in_file << " " << endl;
    free( fname );
    exit( 1 );
  }
  
  /// Default values
  //if ( _out_file == "" ) _out_file = "fold.out";
  if ( gh_params.follow_rmsd && (!_know_prot) ) {
    cout << _dbg << "Follow RMSD option not enable: set known protein first\n";
    gh_params.follow_rmsd = false;
  }
  /// Set default input parameters (tables)
  if ( (!energy_parameters_read) || (energy_parameters_read < 7) ) {
    _energy_charges = "config/coulomb.csv";
    _lj_params      = "config/lenard_jones.csv";
    _h_distances    = "config/h_distances.csv";
    _h_angles       = "config/h_angles.csv";
    _contact_params = "config/contact.csv";
    _tors_params    = "config/table_corr.pot";
    _angles_file    = "config/3combination";
  }
  /// Free resources
  inputFile.close ();
  free( fname );
}//read_file

bool
Input_data::know_prot () const {
  return _know_prot;
}//know_prot

void
Input_data::set_target_sequence ( string t_seq ) {
  _target_sequence = t_seq;
}//get_known_prot_file

string
Input_data::get_target_sequence () const {
  return _target_sequence;
}//get_known_prot_file

void
Input_data::init_data () {
  /// Output file
  gh_params.output_file = _out_file;
  /// Energy Weights (default values)
  gh_params.minimum_energy   = 0;
  gh_params.str_weights[ 0 ] = 0.8; ///1.2
  gh_params.str_weights[ 1 ] = 0.01;
  gh_params.str_weights[ 2 ] = 0.7;
  
  gh_params.crd_weights[ 0 ] = 8;
  gh_params.crd_weights[ 1 ] = 22;  //25;
  gh_params.crd_weights[ 2 ] = 7;   //3;
  
//  gh_params.crd_weights[ 0 ] = 8;
//  gh_params.crd_weights[ 1 ] = 25;  //25;
//  gh_params.crd_weights[ 2 ] = 3;   //3;
  
//  gh_params.crd_weights[ 0 ] = 4; //8
//  gh_params.crd_weights[ 1 ] = 25;  //25; 22
//  gh_params.crd_weights[ 2 ] = 3/50.0;   //3;
  
  /// Load Known Protein and Target sequence
  if ( _know_prot && ( _target_sequence.compare( "" ) == 0 ) ) {
    gh_params.known_protein  = new Protein();
    gh_params.target_protein = new Protein();
    gh_params.known_protein->load_protein( _known_prot_file );
    gh_params.target_protein->set_sequence ( gh_params.known_protein->get_sequence() );
  }
  else if ( _target_sequence.compare( "" ) != 0 ) {
    gh_params.target_protein = new Protein ();
    gh_params.target_protein->set_sequence( _target_sequence );
  }
  else {
    cout << _dbg << "Set FASTA for target\n";
    exit(2);
  }
  
  if ( gh_params.verbose ) gh_params.target_protein->print_sequence ();
  
  gh_params.n_res    = gh_params.target_protein->get_nres();
  gh_params.n_points = gh_params.n_res * 15;
  assert ( gh_params.n_res <= MAX_TARGET_SIZE );
  
  if ( gh_params.follow_rmsd ) {
    int bb_len = (int) gh_params.target_protein->get_bblen();
    gh_params.known_bb_coordinates = (real *) malloc( 3 * bb_len * sizeof(real) );
    for (uint i = 0; i < bb_len; i++)
      for (int j = 0; j < 3; j++)
        gh_params.known_bb_coordinates[i*3 + j] = gh_params.known_protein->get_tertiary()[ i ][ j ];
  }
  
  /// Last Agent -> Default values
  bool to_break = false;
  vector < int > last_agent;
  for ( int i = 0; i < gh_params.n_res; i++ ) {
    for ( uint ii = 0; ii < gh_params.mas_des.size(); ii++ ) {
      to_break = false;
      for ( int iii = 0; iii < gh_params.mas_des[ii].vars_bds.size(); iii++) {
        if ( (i >= gh_params.mas_des[ii].vars_bds[iii].first) &&
             (i <= gh_params.mas_des[ii].vars_bds[iii].second) ) {
          to_break = true;
          break;
        }
      }//iii
      if ( to_break ) break;
    }//ii
    
    if ( !to_break ) last_agent.push_back( i );
  }//i
  if ( last_agent.size() > 0 ) {
    MasAgentDes agt_description;
    agt_description.agt_type = coordinator;
    agt_description.sec_str_type = other;
    agt_description.priority = LOWER_PRIORITY;
    agt_description.quantum = MAX_QUANTUM;
    agt_description.scope.first  = -1;
    agt_description.scope.second = -1;
    for ( int i = 0; i < last_agent.size(); i++)
      agt_description.vars_list.push_back( last_agent[ i ] );
    /// Default search strategy: Montecarlo sampling
    if ( gh_params.gibbs_as_default ) {
      agt_description.search_strategy = gibbs;
    }
    else {
      agt_description.search_strategy = montecarlo;
    }
    
    gh_params.mas_des.push_back( agt_description );
    last_agent.clear();
  }
  /// Null pointers to indentify which array is used (set not NULL later if used)
  gd_params.beam_str     = NULL;
  gd_params.beam_str_upd = NULL;

  Utilities::print_debug ( _dbg, "Loading angles table" );
  load_angles ();
  load_angles_aux ();
  Utilities::print_debug ( _dbg, "Loading energy tables" );
  init_energy_tables ();
  Utilities::print_debug ( _dbg, "Alloc and Init states and energies info on Device" );
  alloc_states ();
  init_states ();
  alloc_energy ();
  init_energy ();
}//init_data

void
Input_data::alloc_constraints () {
  gh_params.max_scope_size = 0;
  gh_params.num_cons = g_constraints.size();
  
#ifdef INPUT_DATA_DBG
  for (int i = 0; i < g_constraints.size(); i++) g_constraints[i]->dump();
  cout << endl;
#endif
  
  /*
   * g_domain_constraint_relation[i][j][z] =>
   * i -> idx Domain
   * j -> event j
   * z -> constraint related to the event j involving V_i
   */
  gh_params.constraint_events.resize( g_logicvars.cp_variables.size() );
  for (int i = 0; i < g_logicvars.cp_variables.size(); i++)
    gh_params.constraint_events[ i ].resize( events_size );
  
  gh_params.constraint_descriptions_idx.resize( g_constraints.size() );
  for (int i = 0; i < g_constraints.size(); i++) {
    Constraint* c = g_constraints[i];
    gh_params.constraint_descriptions_idx[ c->get_id() ] = gh_params.constraint_descriptions.size();
    gh_params.constraint_descriptions.push_back( c->get_type() );         /// Type of constraint
    gh_params.constraint_descriptions.push_back( c->scope_size() );       /// N. of variables involved in it
    gh_params.constraint_descriptions.push_back( c->get_coeff().size() ); /// Coeff size
    gh_params.max_scope_size = max ( gh_params.max_scope_size, (int) c->scope_size() );
    for (int j = 0; j < c->scope_size(); j++) {
      int v_id = c->get_scope_var( j )->get_id();
      gh_params.constraint_descriptions.push_back( v_id );            // id of V_j
      for (int z = 0; z < c->get_num_of_events(); z++) {
        gh_params.constraint_events[ v_id ][ c->get_event( z ) ].push_back( c->get_id() );
      }
      gh_params.constraint_events[ v_id ][ all_events ].push_back( c->get_id() );
    }
    for (int j = 0; j < c->get_coeff().size(); j++) {
      gh_params.constraint_descriptions.push_back( c->get_coeff()[j] );
    }
  }//i
}//alloc_constraints

void
Input_data::load_angles() {
  ifstream in_file;
  in_file.open( _angles_file.c_str() );
  
  if ( !in_file.is_open() ){
    cout << _dbg << "error opening angles file: " << _angles_file << endl;
    exit(2);
  }
  
  gh_params.domain_angles.resize( gh_params.target_protein->get_nres() );
  for (int i = 0; i < gh_params.target_protein->get_nres(); i++)
    gh_params.domain_angles[ i ].resize( 3 );

  
  // Read file
  string line;
  string phi_psi, type_of_aa, token;
  string aa_1, aa_2, aa_3;
  real val;
  size_t found;
  vector< int > to_set;
  vector< aminoacid > aa_sequence = gh_params.target_protein->get_sequence_code();
  while ( in_file.good() ) {
    getline( in_file, line );
    if ( ( line.length() == 0 ) ||
         ( line[ 0 ] == '#' )   ||
         ( line[ 0 ] == '%' ) )
      continue;
    
    if ( line.length() == 11 ) {
      aa_1 = line.substr( 0, 3 );
      aa_2 = line.substr( 4, 3 );
      aa_3 = line.substr( 8, 3 );
      to_set.clear();
      for (uint i = 0; i < gh_params.target_protein->get_nres(); i++) {
        if ( i == 0 ) { // First AA
          if ( cv_aa_to_class ( aa_1 ) == aa_sequence[ i ]   &&
               cv_aa_to_class ( aa_2 ) == aa_sequence[ i ]   &&
               cv_aa_to_class ( aa_3 ) == aa_sequence[ i+1 ] ) {
            to_set.push_back ( i );
          }
        }
        else if ( i == gh_params.target_protein->get_nres() - 1 ) { // Last AA
          if ( cv_aa_to_class ( aa_1 ) == aa_sequence[ i-1 ] &&
               cv_aa_to_class ( aa_2 ) == aa_sequence[ i ]   &&
               cv_aa_to_class ( aa_3 ) == aa_sequence[ i ] ) {
            to_set.push_back ( i );
          }
        }
        else { // AAs
          if ( cv_aa_to_class ( aa_1 ) == aa_sequence[ i-1 ] &&
               cv_aa_to_class ( aa_2 ) == aa_sequence[ i ]   &&
               cv_aa_to_class ( aa_3 ) == aa_sequence[ i+1 ]) {
            to_set.push_back ( i );
          }
        }
      }//i
    }//line
    else { // Read angles
      type_of_aa = line.substr( 0, 1 );
      val = (real) Utilities::cv_string_to_str_type( type_of_aa );
      for ( uint i = 0; i < to_set.size(); i++ ) {
        gh_params.domain_angles[ to_set[ i ] ][ 2 ].push_back( val );
      }
      phi_psi = line.substr(2, line.size() - 2);
      found = phi_psi.find(" ");
      token = phi_psi.substr(0, found);
      val = atof(token.c_str());
      for (uint i = 0; i < to_set.size(); i++) {
        gh_params.domain_angles[ to_set[ i ] ][ 0 ].push_back( val );
      }
      token = phi_psi.substr(found+1, phi_psi.size() - (token.size() + 1));
      val = atof(token.c_str());
      for (uint i = 0; i < to_set.size(); i++) {
        gh_params.domain_angles[ to_set[ i ] ][ 1 ].push_back( val );
      }
    }//else
  }//while
  in_file.close();
}//load_angles

// Note: this is an auxiliary function
// Angles for coordinator agent
void
Input_data::load_angles_aux() {
  ifstream in_file;
  string angles_file_one = "config/1combination";
  in_file.open(angles_file_one.c_str());
  
  if (!in_file.is_open()){
    cout << _dbg << "error opening angles file: " << angles_file_one << endl;
    exit(2);
  }
  
  assert( gh_params.domain_angles.size() > 0 );
  
  int start_aa = 0;
  vector<int> not_to_consider;
  for ( uint i = 0; i < gh_params.mas_des.size(); i++ ) {
    if ( gh_params.mas_des[ i ].agt_type == structure ) {
      start_aa = gh_params.mas_des[ i ].vars_bds[ 0 ].first;
      for (int j = start_aa; j <= gh_params.mas_des[ i ].vars_bds[ 0 ].second; j++)
        not_to_consider.push_back( j );
    }
  }

  // Read the file
  string line;
  string phi_psi, type_of_aa, token;
  string aa_1;
  real val, val1, val2;
  size_t found;
  vector<int> to_set;
  vector< aminoacid > aa_sequence = gh_params.target_protein->get_sequence_code();
  bool put_angles = true;
  while (in_file.good()) {
    getline(in_file, line);
    if((line.length() == 0) ||
       (line[0] == '#')     ||
       (line[0] == '%'))
      continue;
    
    if (line.length() == 3) {
      aa_1 = line.substr(0, 3);
      to_set.clear();
      for ( uint i = 0; i < gh_params.target_protein->get_nres(); i++ ) {
        if ( cv_aa_to_class(aa_1) == aa_sequence[i] ) {
          put_angles = true;
          for (uint j = 0; j < not_to_consider.size(); j++) {
            if (not_to_consider[j] == i) {
              put_angles = false;
              break;
            }
          }
          if (put_angles) {
            to_set.push_back(i);
            put_angles = true;
          }
        }
      }//i
    }//line
    else { //read angles
      if (to_set.size() > 0) {
        //TYPE
        type_of_aa = line.substr(0, 1);
        val = (real) Utilities::cv_string_to_str_type(type_of_aa);
        //PHI
        phi_psi = line.substr( 2, line.size() - 2 );
        found = phi_psi.find(" ");
        token = phi_psi.substr( 0, found );
        val1 = atof(token.c_str());
        //PSI
        token = phi_psi.substr(found+1, phi_psi.size() - (token.size() + 1));
        val2 = atof(token.c_str());
      
        for (uint i = 0; i < to_set.size(); i++) {
          if ( gh_params.domain_angles[to_set[i]][0].size() <= 1000 ) {
            gh_params.domain_angles[to_set[i]][0].push_back(val1);
            gh_params.domain_angles[to_set[i]][1].push_back(val2);
            gh_params.domain_angles[to_set[i]][2].push_back(val);
          }
        }
      }//to_set
    }//else
  }//while
  
  in_file.close();
}//load_angles_aux

void 
Input_data::init_energy_tables() {
  // Energy Tables: sizes taken from tables' formats
  gh_params.h_distances    = (real*) malloc ( 25*3 * sizeof(real) );
  gh_params.h_angles       = (real*) malloc ( 73*9 * sizeof(real) );
  gh_params.contact_params = (real*) malloc ( 20*20 * sizeof(real) );
  gh_params.tors           = (real*) malloc ( 20*20*18*3 * sizeof(real) );
  gh_params.tors_corr      = (real*) malloc ( 18*18*5 * sizeof(real) );
  gh_params.beam_energies  = (real*) malloc ( MAX_SAMPLE_SET_SIZE * sizeof(real) );
  
  // Fill the matrices with csv values read from file
  std::vector< std::vector<real> > energy_params;
  // H_Distances
  read_energy_parameters ( _h_distances, energy_params );
  for (int i = 0; i < 25; i++)
    for (int j = 0; j < 3; j++)
      gh_params.h_distances[ 3*i + j ] = energy_params[ i ][ j ];
  energy_params.clear();
  // H_Angles
  read_energy_parameters ( _h_angles, energy_params );
  for (int i = 0; i < 73; i++)
    for (int j = 0; j < 9; j++)
      gh_params.h_angles[ 9*i + j ] =  energy_params[ i ][ j ];
  energy_params.clear();
  Utilities::print_debug ( _dbg, "Hydrogen parameters loaded" );
  // Contact Parameters
  read_energy_parameters ( _contact_params, energy_params );
  for (int i = 0; i < 20; i++)
    for (int j = 0; j < 20; j++)
      gh_params.contact_params[ 20*i+j ] = energy_params[ i ][ j ];
  energy_params.clear();
  Utilities::print_debug ( _dbg, "Contact parameters loaded" );
  
  // From ".h" file
  for (int i = 0; i < 20; i++)
    for (int j = 0; j < 20; j++)
      for (int z = 0; z < 18; z++)
        for (int t = 0; t < 3; t++)
          gh_params.tors [ i*20*18*3 + j*18*3 + z*3 + t ] = tors[ i ][ j ][ z ][ t ];
  for (int i = 0; i < 18; i++)
    for (int j = 0; j < 18; j++)
      for (int z = 0; z < 5; z++)
        gh_params.tors_corr[ i*18*5 + j*5 + z ] = tors_corr[ i ][ j ][ z ];
  Utilities::print_debug ( _dbg, "Torsional parameters loaded" );
}//init_energy_tables

void
Input_data::alloc_states () {
  HANDLE_ERROR( cudaMalloc( ( void** )&gd_params.domain_states, MAX_DIM * gh_params.n_res   * sizeof( uint ) ) );
  HANDLE_ERROR( cudaMalloc( ( void** )&gd_params.validity_solutions, gh_params.set_size * sizeof( real ) ) );
  HANDLE_ERROR( cudaMalloc( ( void** )&gd_params.domain_events, gh_params.n_res * sizeof( int ) ) );
  /// Pinned memory for random array
  HANDLE_ERROR( cudaHostAlloc( (void**)&gh_params.random_array ,
                              gh_params.n_res * MAX_GIBBS_SET_SIZE * sizeof(int),
                              cudaHostAllocDefault) );
  /// Random values
  int round_val = 1;
  while ( MAX_N_THREADS * round_val < gh_params.set_size ) round_val++;
  HANDLE_ERROR( cudaMalloc( ( void** )&gd_params.random_state, MAX_N_THREADS * round_val * sizeof( curandState )    ) );
  HANDLE_ERROR( cudaMalloc( ( void** )&gd_params.random_array, gh_params.n_res * MAX_GIBBS_SET_SIZE * sizeof( int ) ) );
}//alloc_states

void
Input_data::init_states () {
  uint all_valid = (uint) (pow ( (double)2.0, MAX_DIM ) - 1); /// 4294967295 /\ MAX_DIM = 32
  gh_params.domain_states = (uint*) malloc ( MAX_DIM * gh_params.n_res  * sizeof(uint) );
  /// Pinned memory for validity_solutions
  HANDLE_ERROR( cudaHostAlloc( (void**)&gh_params.validity_solutions ,
                               gh_params.set_size * sizeof(real),
                               cudaHostAllocDefault) );
  
  for (int i = 0; i < MAX_DIM * gh_params.n_res; i++)
    gh_params.domain_states[ i ] = all_valid;
  for (int i = 0; i < gh_params.set_size; i++) gh_params.validity_solutions[ i ] = 1;

  HANDLE_ERROR( cudaMemcpy( gd_params.domain_states, gh_params.domain_states,
                            MAX_DIM * gh_params.n_res * sizeof( uint ), cudaMemcpyHostToDevice ) );
  HANDLE_ERROR( cudaMemcpy( gd_params.validity_solutions, gh_params.validity_solutions,
                            gh_params.set_size * sizeof(real), cudaMemcpyHostToDevice ) );
  
  
}//init_states

void
Input_data::alloc_energy () {
  HANDLE_ERROR( cudaMalloc( ( void** )&gd_params.h_distances, 25*3 * sizeof(real) ) );
  HANDLE_ERROR( cudaMalloc( ( void** )&gd_params.h_angles, 73*9 * sizeof(real) ) );
  HANDLE_ERROR( cudaMalloc( ( void** )&gd_params.contact_params, 20*20 * sizeof(real) ) );
  HANDLE_ERROR( cudaMalloc( ( void** )&gd_params.tors, 20*20*18*3 * sizeof(real) ) );
  HANDLE_ERROR( cudaMalloc( ( void** )&gd_params.tors_corr, 18*18*5 * sizeof(real) ) );
  HANDLE_ERROR( cudaMalloc( ( void** )&gd_params.aa_seq,
                            gh_params.target_protein->get_nres() * sizeof(aminoacid) ) );
  HANDLE_ERROR( cudaMalloc( ( void** )&gd_params.secondary_s_info,
                            gh_params.target_protein->get_nres() * sizeof(ss_type) ) );
  if ( gh_params.follow_rmsd ) {
    int bb_len = (int) gh_params.target_protein->get_bblen();
    HANDLE_ERROR( cudaMalloc( ( void** )&gd_params.known_prot, 3 * bb_len * sizeof( real ) ) );
  }
}//alloc_energy

void
Input_data::init_energy () {
  ss_type* secondary_s_info = ( ss_type* ) malloc ( gh_params.target_protein->get_nres() * sizeof(ss_type) );
  for ( int i = 0; i < gh_params.mas_des.size(); i++ )
    for ( int j = 0; j < gh_params.mas_des[i].vars_list.size(); j++ )
      secondary_s_info[ gh_params.mas_des[i].vars_list[j] ] = gh_params.mas_des[i].sec_str_type;
  
  HANDLE_ERROR( cudaMemcpy( gd_params.secondary_s_info, secondary_s_info,
                           gh_params.target_protein->get_nres() * sizeof(ss_type),
                           cudaMemcpyHostToDevice ) );
  HANDLE_ERROR( cudaMemcpy( gd_params.h_distances, gh_params.h_distances,
                           25*3 * sizeof(real), cudaMemcpyHostToDevice ) );
  HANDLE_ERROR( cudaMemcpy( gd_params.h_angles, gh_params.h_angles,
                           73*9 * sizeof(real), cudaMemcpyHostToDevice ) );
  HANDLE_ERROR( cudaMemcpy( gd_params.contact_params, gh_params.contact_params,
                           20*20 * sizeof(real), cudaMemcpyHostToDevice ) );
  HANDLE_ERROR( cudaMemcpy( gd_params.tors, gh_params.tors,
                           20*20*18*3 * sizeof(real), cudaMemcpyHostToDevice ) );
  HANDLE_ERROR( cudaMemcpy( gd_params.tors_corr, gh_params.tors_corr,
                           18*18*5 * sizeof(real), cudaMemcpyHostToDevice ) );
  HANDLE_ERROR( cudaMemcpy( gd_params.aa_seq, &(gh_params.target_protein->get_sequence_code()[ 0 ]),
                           gh_params.target_protein->get_nres() * sizeof(aminoacid),
                           cudaMemcpyHostToDevice ) );
  if ( gh_params.follow_rmsd ) {
    int bb_len = (int) gh_params.target_protein->get_bblen();
    HANDLE_ERROR( cudaMemcpy( gd_params.known_prot, gh_params.known_bb_coordinates,
                              3 * bb_len * sizeof( real ), cudaMemcpyHostToDevice ) );
  }
  free ( secondary_s_info );
}//init_energy

void
Input_data::free_dt () {
  Utilities::print_debug ( _dbg, "Free memory on device" );
  HANDLE_ERROR( cudaFree( gd_params.h_distances ) );
  HANDLE_ERROR( cudaFree( gd_params.h_angles ) );
  HANDLE_ERROR( cudaFree( gd_params.contact_params ) );
  HANDLE_ERROR( cudaFree( gd_params.tors ) );
  HANDLE_ERROR( cudaFree( gd_params.tors_corr ) );
  HANDLE_ERROR( cudaFree( gd_params.aa_seq ) );
  HANDLE_ERROR( cudaFree( gd_params.domain_states ) );
  HANDLE_ERROR( cudaFree( gd_params.validity_solutions ) );
  HANDLE_ERROR( cudaFree( gd_params.domain_events ) );
  HANDLE_ERROR( cudaFree( gd_params.secondary_s_info ) );
  HANDLE_ERROR( cudaFree( gd_params.random_state ) );
  HANDLE_ERROR( cudaFree( gd_params.vars_to_shuffle ) );
  HANDLE_ERROR( cudaFree( gd_params.random_array ) );
  if ( !gd_params.vars_to_shuffle )
    HANDLE_ERROR( cudaFree( gd_params.vars_to_shuffle ) );
  if ( gh_params.follow_rmsd )
    HANDLE_ERROR( cudaFree( gd_params.known_prot ) );
}//free_dt


void
Input_data::read_energy_parameters ( string file_name, vector< vector<real> >& param_v ) {
  string line;
  string token;
  real value;
  size_t found;
  size_t t = 0;
  
  ifstream csv_file ( file_name.c_str() );
  if ( !csv_file.is_open() ) {
    cout << _dbg << "error opening energy parameters file " << file_name << endl;
    exit( 1 );
    return;
  }
  
  while ( csv_file.good() ) {
    vector< real > line_v;
    getline ( csv_file, line );
    if (line.compare(0,1, "%") == 0) continue;
    found = line.find_first_of( "," );
    while ( found != string::npos ) {
      token = line.substr( t, found - t );
      value = atof( token.c_str() );
      line_v.push_back( value );
      t = t + token.size() + 1;
      found = line.find_first_of( ",", found+1 );
    }
    token = line.substr( t, found - t );
    value = atof( token.c_str() );
    line_v.push_back( value );
    
    param_v.push_back( line_v );
    t = 0;
  }//while
  
  csv_file.close();
}//read_energy_parameters

void
Input_data::read_torsional_parameters ( string file_name, real tors_param[20][20][20][3] ) {
  string line;
  string token;
  real value;
  size_t found;
  size_t t = 0;
  bool already_written = false;
  
  ifstream tors_file (file_name.c_str());
  if (!tors_file.is_open()) {
    cout << _dbg << "error in opening energy parameters file " << file_name << endl;
    exit( 1 );
  }
  
  int i = 0, j = 0, aa = 0, z = 0, val = 0;
  while ( tors_file.good() ) {
    getline(tors_file, line);
    if (line.compare(0,1, "%") == 0)
      continue;
    found = line.find_first_of(" ");
    while (found != string::npos) {
      token = line.substr(t, found - t);
      value = atof(token.c_str());
      
      if (val < 2) {
        for (int aa_idx = 0; aa_idx < 20; aa_idx++)
          tors_param[i][j][aa_idx][val] = value;
        val++;
      }
      else {
        z = Input_data::convert_aa_pos(aa-2);
        tors_param[i][j][z][val] = value;
      }
      aa++;
      
      t = t + token.size() + 1;
      found = line.find_first_of(" ", found+1);
      already_written = true;
    }//while
    
    token = line.substr( t, found - t );
    value = atof( token.c_str() );
    
    if (already_written) {
      z = Input_data::convert_aa_pos( aa-2 );
      tors_param[i][j][z][val] = value;
      aa++; i++;
    }
    
    if ( i == 20 ) { j++; i = 0; }
    if ( aa == 22 ) { aa = 0; }
    
    val = 0;
    t = 0;
    already_written = false;
  }//while
  
  tors_file.close();
}//read_torsional_parameters

int
Input_data::convert_aa_pos( int a ) {
  if (a==0) return 0;
  if (a==1) return 4;
  if (a==2) return 3;
  if (a==3) return 6;
  if (a==4) return 13;
  if (a==5) return 7;
  if (a==6) return 8; 
  if (a==7) return 9;
  if (a==8) return 11;
  if (a==9) return 10;
  if (a==10) return 12;
  if (a==11) return 2;
  if (a==12) return 14;
  if (a==13) return 5;
  if (a==14) return 15;
  if (a==15) return 16;
  if (a==16) return 1;
  if (a==17) return 19;
  if (a==18) return 17;
  if (a==19) return 18;
  cout << "Error (convert_aa_pos): " << a << endl;
  return -1;
}//convert_aa_pos

void
Input_data::dump () {
  cout << _dbg << "Data imported:" << endl;
  cout << _dbg << "Energy Lenard-Jones   : " << _lj_params << endl;
  cout << _dbg << "Energy Hydrogen dis.  : " << _h_distances << endl;
  cout << _dbg << "Energy Hydrogen ang.  : " << _h_angles << endl;
  cout << _dbg << "Angles file           : " << _angles_file << endl;
  cout << _dbg << "Protein Known  ref    : " << _known_prot_file << endl;
  cout << _dbg << "Protein Target ref    : " << _target_prot_file << endl;
  cout << endl;
}//dump

void
Input_data::print_gpu_info () {
  cudaDeviceProp  prop;
  int count;
  HANDLE_ERROR( cudaGetDeviceCount( &count ) );
  for ( int i = 0; i < count; i++ ) {
    HANDLE_ERROR( cudaGetDeviceProperties( &prop, i ) );
    printf( "   --- General Information for device %d ---\n", i );
    printf( "Name: %s\n", prop.name );
    printf( "Compute capability: %d.%d\n", prop.major, prop.minor );
    printf( "Clock rate: %d\n", prop.clockRate );
    printf( "Device copy overlap: " );
    if (prop.deviceOverlap) printf( "Enabled\n" );
    else printf( "Disabled\n" );
    printf( "Kernel execition timeout : " );
    if (prop.kernelExecTimeoutEnabled) printf( "Enabled\n" );
    else printf( "Disabled\n" );
    printf( "   --- Memory Information for device %d ---\n", i );
    printf( "Total global mem:  %ld\n", prop.totalGlobalMem );
    printf( "Total constant Mem:  %ld\n", prop.totalConstMem );
    printf( "Max mem pitch:  %ld\n", prop.memPitch );
    printf( "Texture Alignment:  %ld\n", prop.textureAlignment );
    printf( "   --- MP Information for device %d ---\n", i );
    printf( "Multiprocessor count:  %d\n", prop.multiProcessorCount );
    printf( "Shared mem per mp:  %ld\n", prop.sharedMemPerBlock );
    printf( "Registers per mp:  %d\n", prop.regsPerBlock );
    printf( "Threads in warp:  %d\n", prop.warpSize );
    printf( "Max threads per block:  %d\n", prop.maxThreadsPerBlock );
    printf( "Max thread dimensions:  (%d, %d, %d)\n",
            prop.maxThreadsDim[0], prop.maxThreadsDim[1],
            prop.maxThreadsDim[2] );
    printf( "Max grid dimensions:  (%d, %d, %d)\n",
            prop.maxGridSize[0], prop.maxGridSize[1],
            prop.maxGridSize[2] );
    printf( "\n" );
  }
}//print_gpu_info

void
Input_data::print_help () {
  cout << "usage: ./cocos -i <infile> [-o <outfile>] [-g <int>] [-s <int>] [-a] [-e] [-r] [-v] [-d] [-q] [-h]\n" << endl;
  cout << "Options for Cocos:\n";
  cout << "\t" << "-h\n";
  cout << "\t\t" << "print this help message\n";
  cout << "\t" << "-i (string)\n";
  cout << "\t\t" << "set input\n";
  cout << "\t" << "-o (string) default: fold.out\n";
  cout << "\t\t" << "set output file name\n";
  cout << "\t" << "-c (integer) default: none\n";
  cout << "\t\t" << "set timeout (sec.) for Montecarlo sampling\n";
  cout << "\t" << "-g (integer) default: 10\n";
  cout << "\t\t" << "set number of samples for the Gibbs algorithm\n";
  cout << "\t" << "-s (integer) default: 1000\n";
  cout << "\t\t" << "set number of initial random point for the Gibbs algorithm\n";
  cout << "\t" << "-t (integer) default: 5\n";
  cout << "\t\t" << "set number of sampling steps of the Gibbs algorithm before swapping bins\n";
  cout << "\t" << "-a\n";
  cout << "\t\t" << "Automagically create an input file for cocos from FASTA sequence\n";
  cout << "\t" << "-e\n";
  cout << "\t\t" << "Enable CG constraint\n";
  cout << "\t" << "-r\n";
  cout << "\t\t" << "set RMSD as objective function\n";
  cout << "\t" << "-q\n";
  cout << "\t\t" << "set Gibbs sampling algorithm on all Coordinator agents (default: MonteCarlo)\n";
  cout << "\t" << "-v\n";
  cout << "\t\t" << "printf verbose info during computation\n";
  cout << "\t" << "-d\n";
  cout << "\t\t" << "printf device info\n";
}//print_help
