#include "input_data.h"

#include "logic_variables.h"
#include "constraint.h"
#include "tors_corr_bmf.h"
#include "tors_bmf.h"
#include "utilities.h"
#include "atom.h"
#include "atom_grid.h"

#include <getopt.h>

//#define INPUT_DATA_DBG

using namespace std;
using namespace Utilities;

/// Init static variable
Input_data* Input_data::_instance = nullptr;

Input_data::Input_data ( int argc, char* argv[] ) :
_dbg              ( "#log: Input_data - " ),
_know_prot        ( false ),
_in_file          ( "" ),
_out_file         ( "" ),
_target_sequence  ( "" ) {
  
  /// Set default values/parameters
  set_default_values ();
  
  /// Read input
  int c;
  int verbose_flag=0,  rmsd_flag=0,  allign_flag=0;
  int centroid_flag=0, gibbs_flag=0, translate_flag=0;
  while ( true ) {
    static struct option long_options[] =
    {
      /* These options set a flag. */
      {"verbose",       no_argument, &verbose_flag,           1}, /// Print verbose during search
      {"rmsd",          no_argument, &rmsd_flag,              1}, /// Use RMSD as obj function
      {"auto_allign",   no_argument, &allign_flag,            1}, /// Automatically find sec. structure
      {"cg_constraint", no_argument, &centroid_flag,          1}, /// Set CGs
      {"gibbs_default", no_argument, &gibbs_flag,             1}, /// Use Gibbs as default for coordinators agts
      {"translate",     no_argument, &translate_flag,         1}, /// Translate 2nd atom of final prediction on (0, 0, 0)
      {"random_moves",  no_argument, &gh_params.random_moves, 1}, /// Random translation/rotations peptide in docking analysis
      /* These options don't set a flag.
         We distinguish them by their indices. */
      {"help",          no_argument,       0,       'h'}, /// Print a help message
      {"input",         required_argument, 0,       'i'}, /// Set input file
      {"output",        required_argument, 0,       'o'}, /// Set output file
      {"angles",        required_argument, 0,       'a'}, /// Set the variables domains using [-180, +180] partitioned as specified
      {"set_size",      required_argument, 0,       's'}, /// Set size of sampling sets
      {"mc_timeout",    required_argument, 0,       'c'}, /// Set timeout for MonteCarlo sampling
      {"docking",       required_argument, 0,       'k'}, /// Set minimum number of contacts for docking
      {"gibbs",         required_argument, 0,       'g'}, /// Set number of Gibbs samples
      {"gb_iterations", required_argument, 0,       't'}, /// Set number of iterations before swapping bins in Gibbs sampling
      {0, 0, 0, 0}
    };
    
    /* getopt_long stores the option index here. */
    int option_index = 0;
    c = getopt_long (argc, argv, "hvri:o:a:s:c:k:g:t:",
                     long_options, &option_index);
    /* Detect the end of the options. */
    if ( c == -1 ) break;
    /* Switch on c to detect the input given by the user */
    switch ( c ) {
      case 0:
        /* If this option set a flag, do nothing else now. */
        if ( long_options[ option_index ].flag != 0 )
          break;
        printf ( "option %s", long_options[ option_index ].name );
        if (optarg)
          printf (" with arg %s", optarg);
        printf ("\n");
        break;
      case 'h':
        print_help();
        exit( 0 );
      case 'v':
        verbose_flag = 1;
        break;
      case 'r':
        gh_params.random_moves = 1;
        break;
      case 'i':
        _in_file  = optarg;
        break;
      case 'o':
        _out_file = optarg;
        break;
      case 'a':
        gh_params.set_angles = atoi ( optarg );
        break;
      case 's':
        gh_params.set_size = atoi ( optarg );
        break;
      case 'c':
        gh_params.timer = atoi ( optarg );
        break;
      case 'k':
        gh_params.sys_job         = docking;
        gh_params.min_n_contacts  = atoi ( optarg );
        break;
      case 'g':
        gh_params.n_gibbs_samples = atoi ( optarg );
        break;
      case 't':
        gh_params.n_gibbs_iters_before_swap = atoi ( optarg );
        break;
      default:
        exit( 0 );
    }//switch
  }//while
  
  /* Instead of reporting ‘--verbose’ as it is encountered,
     we report the final status resulting from them. */
  if ( verbose_flag   ) {
    gh_params.verbose = true;
    puts ("verbose flag is set");
  }
  if ( allign_flag    ) {
    create_input_file ();
    _in_file = "alignment.txt";
  }
  if ( rmsd_flag      ) { gh_params.follow_rmsd = true;       }
  if ( centroid_flag  ) { gh_params.centroid = true;          }
  if ( gibbs_flag     ) { gh_params.gibbs_as_default = true;  }
  if ( translate_flag ) { gh_params.translate_str_fnl = true; }
  
  /* Print any remaining command line arguments (not options). */
  if ( optind < argc ) {
    printf ("non-option ARGV-elements: ");
    while ( optind < argc )
      printf ("%s ", argv[ optind++ ]);
    putchar ('\n');
  }

  /* Parse and get info from the input file */
  if ( _in_file == "" ) {
    print_help();
    exit( 0 );
  }
  
  /* Read input file */
  read_file ();
  
  /* Init data structures */
  init_data ();
}//Input_data

void
Input_data::ask_for_seeds () {
  cout << "Input_data::Docking - Please insert:\n";
  cout << "x y z r h\n";
  cout << "where\n";
  cout << "x (real): x seed's coordinate\n";
  cout << "y (real): y seed's coordinate\n";
  cout << "z (real): z seed's coordinate\n";
  cout << "r (real): half diagonal of the cube centered in (x, y, z)\n";
  cout << "h (int) : height of the octree (i.e., number of partitions)\n";
  cout << "Press Enter to insert a new seed.\n";
  cout << "Write \"remove\" to remove the last inserted seed or \"done\" to exit.\n";
  
  /// Read input from user
  string line = "";
  vector < real > coords;
  /// Start reading input from user
  getline ( cin, line );
  while ( line.compare( "done" ) != 0 ) {
    if ( (line.compare( "remove" ) == 0) &&
         (gh_params.seed_coords.size() > 0) ) {
      gh_params.seed_coords.pop_back();
    }
    else {
      stringstream stream( line );
      real n;
      int parsed_val = 0;
      while( 1 ) {
        stream >> n;
        parsed_val++;
        if( (!stream) || (parsed_val >= 5) ) break;
        coords.push_back( n );
      }
      if ( parsed_val == 5 ) {
        coords.push_back( 4 );
      }
      gh_params.seed_coords.push_back( coords );
      coords.clear();
    }
    getline ( cin, line );
  }
  /// User error check
  if ( gh_params.seed_coords.size() == 0 ) {
    cout << "Please, insert at least one seed!\n";
    exit( 2 );
  }
}//ask_for_seeds

void
Input_data::set_default_values () {
  g_docking                   = nullptr;
  g_atom_grid                 = nullptr;
  gh_params.known_protein     = nullptr;
  gh_params.sys_job           = ab_initio;
  gh_params.gibbs_as_default  = false;
  gh_params.follow_rmsd       = false;
  gh_params.verbose           = false;
  gh_params.centroid          = false;
  gh_params.translate_str     = false;
  gh_params.translate_str_fnl = false;
  gh_params.atom_grid         = false;
  gh_params.set_angles        = -1;
  gh_params.n_gibbs_samples   = -1;
  gh_params.timer             = -1;
  gh_params.min_n_contacts    = -1;
  gh_params.random_moves      = 0;
  gh_params.n_coordinators    = 1;
  gh_params.num_models        = 1;
  gh_params.translation_point[ 1 ]    = 0;
  gh_params.translation_point[ 2 ]    = 0;
  gh_params.translation_point[ 3 ]    = 0;
  gh_params.n_gibbs_iters_before_swap = 0;
  gh_params.set_size                  = MAX_GIBBS_SET_SIZE;
}//set_default_values

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
  
  free ( gh_params.domain_states );
  free ( gh_params.validity_solutions );
  free ( gh_params.h_distances );
  free ( gh_params.h_angles );
  free ( gh_params.contact_params );
  free ( gh_params.tors );
  free ( gh_params.tors_corr );
  free ( gh_params.beam_energies );
  if ( _know_prot )
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
    string prg_name = "jnet";
    string prg_opts = "-s";
    
    char *prg_name_nc = new char[ prg_name.length() + 1 ];
    strcpy ( prg_name_nc, prg_name.c_str() );
    
    char *prg_opts_nc = new char[ prg_opts.length() + 1 ];
    strcpy ( prg_opts_nc, prg_opts.c_str() );
    
    static char *argv[] = { prg_name_nc, prg_opts_nc, cstr, NULL };
    execv( "./bin_jnet/jnet",argv );
    
    delete [] prg_name_nc;
    delete [] prg_opts_nc;
    
    exit(127); /// only if execv fails
  }
  else { /// pid!=0; parent process
    waitpid(pid,0,0); /// wait for child to exit
  }
  /// File alignment.txt created: open it and read the line
  ifstream inputFile;
  string line_fasta, buffer, line_allign, allign_input = "alignment.txt";
  char * fname = (char*) malloc ( (allign_input.size() + 1) * sizeof(char) );
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
  
  
  char * fname2 = (char*) malloc ( (_in_file.size() + 1) * sizeof(char) );
  strcpy( fname2, _in_file.c_str() );
  inputFile.open( fname2 );
  if( inputFile.is_open() ){
    while ( inputFile.good() ){
      getline ( inputFile, buffer );
      if (buffer.length() != 0){
        if (buffer.compare( 0, 1, ">" ) != 0 ){
          line_fasta = buffer;
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
  
  fprintf( fid, "AB_INITIO\n" );
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

/*
 * Read input file provided by the user
 * and set appropriate data structures
 * and file paths.
 */
void
Input_data::read_file () {
  ifstream inputFile ( _in_file, std::ifstream::in );
  if( inputFile.is_open() ){
    string line;
    string docking_str   = "DOCKING";
    string ab_initio_str = "AB_INITIO";
    getline ( inputFile, line );
    /*
     * The first line specifies if the tool is doing
     * Ab_initio prediction or docking.
     */
    size_t found_docking   = line.find( docking_str );
    size_t found_ab_initio = line.find( ab_initio_str );
    if ( found_ab_initio != string::npos ) {
      /// Proceed with parsing for ab_initio
      parse_for_ab_initio ( inputFile );
    }
    else if ( found_docking != string::npos ) {
      /// Proceed with parsing for docking
      gh_params.sys_job = docking;
      parse_for_docking ( inputFile );
    }
    else {
      cout
      << "Please, specify the type of analysis "
      << "to perform (either AB_INITIO or DOCKING):"
      << endl;
      /// Read input from user
      getline ( cin, line );
      while ( (line.compare( docking_str )   != 0) &&
              (line.compare( ab_initio_str ) != 0) ) {
        cout
        << "Please, specify the type of analysis "
        << "to perform (either AB_INITIO or DOCKING), read "
        << line
        << endl;
        getline ( cin, line );
      }
      if ( line.compare( ab_initio_str ) == 0 ) {
        /// Proceed with parsing for ab_initio
        parse_for_ab_initio ( inputFile );
      }
      else {
        /// Proceed with parsing for docking
        gh_params.sys_job = docking;
        parse_for_docking ( inputFile );
      }
    }
  }
  else {
    cout << _dbg << "Unable to open " << _in_file << endl;
    exit( 1 );
  }
  /// Close ifstream
  inputFile.close ();
}//read_file

void
Input_data::parse_for_ab_initio ( istream& inputFile ) {
  /// Set position to the beginning of the file
  inputFile.seekg ( 0, inputFile.beg );
  string line;
  int start = 12;
  while ( inputFile.good() ) {
    getline ( inputFile, line );
    if ( (line.compare( "DOCKING" )   == 0) ||
         (line.compare( "AB_INITIO" ) == 0) ) {
      continue;
    }
    /*
     * Target on FASTA format
     * e.g.,
     * >SEQUENCE_1
     * MTEITAAMVKELRES...
     */
    if ( line.compare( 0, 1, ">" ) == 0 ) {
      getline ( inputFile, line );
      set_target_sequence ( line );
    }
    ///  Read target / known_prot info (i.e., paths)
    if ( ((line.compare( 0, 10, "KNOWN_PROT" )  == 0)  ||
          (line.compare( 0, 11, "TARGET_PROT" ) == 0)) &&
         (!_know_prot) )  {
      start = line.find_first_of ( " " );
      start++;
      _known_prot_file = line.substr( start, line.size() - start );
      _known_prot_file += ".pdb";
      _know_prot = true;
      continue;
    }
    set_database ( line );
    /// Secondary Structure Descriptions, Agents, and Priorities
    set_agents ( line );
  }//while
  /// Set default values for databases
  if ( set_database ( "" ) < 7 ) {
    _energy_charges = "config/coulomb.csv";
    _lj_params      = "config/lenard_jones.csv";
    _h_distances    = "config/h_distances.csv";
    _h_angles       = "config/h_angles.csv";
    _contact_params = "config/contact.csv";
    _tors_params    = "config/table_corr.pot";
    _angles_file    = "config/3combination";
  }
  /// Consistency check
  if ( gh_params.follow_rmsd && (!_know_prot) ) {
    cout << _dbg << "Follow RMSD option not enable: set known protein.\n";
    gh_params.follow_rmsd = false;
  }
}//parse_for_ab_initio

void
Input_data::parse_for_docking ( istream& inputFile ) {
  /// Set position to the beginning of the file
  inputFile.seekg ( 0, inputFile.beg );
  string line;
  int start = 12;
  bool seed_found = false;
  while ( inputFile.good() ) {
    getline ( inputFile, line );
    if ( (line.compare( "DOCKING" )   == 0) ||
         (line.compare( "AB_INITIO" ) == 0) ) {
      continue;
    }
    ///  Read target / known_prot info (i.e., paths)
    if ( ((line.compare( 0, 10, "KNOWN_PROT" )  == 0)  ||
          (line.compare( 0, 11, "TARGET_PROT" ) == 0)  ||
          (line.compare( 0, 7,  "PEPTIDE" ) == 0) ) &&
        (!_know_prot) )  {
      start = line.find_first_of ( " " );
      start++;
      _known_prot_file = line.substr( start, line.size() - start );
      _known_prot_file += ".pdb";
      _know_prot = true;
      continue;
    }
    if ( line.compare( 0, 4, "SEED" )  == 0 ) {
      seed_found = true;
      start = line.find_first_of ( " " );
      start++;
      line = line.substr( start, line.size() - start );

      stringstream stream( line );
      real n;
      int parsed_val = 0;
      vector < real > coords;
      while( 1 ) {
        stream >> n;
        parsed_val++;
        if( (!stream) || (parsed_val > 5) ) break;
        coords.push_back( n );
      }
      if ( coords.size() == 4 ) {
        coords.push_back( 4 );
      }
      gh_params.seed_coords.push_back( coords );
      coords.clear();
      continue;
    }// seed
    set_database ( line );
    set_dock_constraints ( line );
  }//while
  /// Set default values for databases
  if ( set_database ( "" ) < 7 ) {
    _energy_charges = "config/coulomb.csv";
    _lj_params      = "config/lenard_jones.csv";
    _h_distances    = "config/h_distances.csv";
    _h_angles       = "config/h_angles.csv";
    _contact_params = "config/contact.csv";
    _tors_params    = "config/table_corr.pot";
    _angles_file    = "config/3combination";
  }
  /// Consistency check
  if ( gh_params.follow_rmsd && (!_know_prot) ) {
    cout << _dbg << "Follow RMSD option not enable: set known protein.\n";
    gh_params.follow_rmsd = false;
  }
  if ( g_docking == nullptr ) {
    cout << _dbg << " Set docking file for docking grid.\n";
    exit( 2 );
  }
  /// Read seeds for docking from user
  if ( (!seed_found) && (!gh_params.translate_str) ) {
    ask_for_seeds ();
  }
  /// Set default value
  if ( gh_params.min_n_contacts == -1 ) {
    gh_params.min_n_contacts = 4;
    if ( gh_params.force_contact.size() > gh_params.min_n_contacts ) {
      gh_params.min_n_contacts = gh_params.force_contact.size();
    }
  }
  
}//parse_for_docking

int
Input_data::set_database ( string line ) {
  static int energy_parameters_read = 0;
  
  int start = 12;   // Value taken from the format of the input file
  if ( energy_parameters_read == 7 ) { return energy_parameters_read; }
  start = line.find_first_of ( " " );
  start++;
  if ( line.compare( 0, 10, "COULOMBPAR" ) == 0 ) {
    _energy_charges = line.substr( start, line.size() - start );
    start = _energy_charges.find_first_not_of ( " " );
    _energy_charges = _energy_charges.substr(start, _energy_charges.size() - start);
    energy_parameters_read++;
  }
  else if ( line.compare( 0, 11, "LJPARAMETER" ) == 0 ) {
    _lj_params = line.substr( start, line.size() - start );
    start = _lj_params.find_first_not_of ( " " );
    _lj_params = _lj_params.substr(start, _lj_params.size() - start);
    energy_parameters_read++;
  }
  else if ( line.compare( 0, 11, "HDPARAMETER" ) == 0 ) {
    _h_distances = line.substr( start, line.size() - start );
    start = _h_distances.find_first_not_of ( " " );
    _h_distances = _h_distances.substr(start, _h_distances.size() - start);
    energy_parameters_read++;
  }
  else if ( line.compare( 0, 11, "HAPARAMETER" ) == 0 ) {
    _h_angles = line.substr( start, line.size() - start );
    start = _h_angles.find_first_not_of ( " " );
    _h_angles = _h_angles.substr(start, _h_angles.size() - start);
    energy_parameters_read++;
  }
  else if ( line.compare( 0, 7, "CONTACT" ) == 0 ) {
    _contact_params = line.substr( start, line.size() - start );
    start = _contact_params.find_first_not_of ( " " );
    _contact_params = _contact_params.substr(start, _contact_params.size() - start);
    energy_parameters_read++;
  }
  else if ( line.compare( 0, 7, "TORSPAR" ) == 0 ) {
    _tors_params = line.substr( start, line.size() - start );
    start = _tors_params.find_first_not_of ( " " );
    _tors_params = _tors_params.substr(start, _tors_params.size() - start);
    energy_parameters_read++;
  }
  else if (line.compare( 0, 6, "ANGLES" ) == 0 ) {
    _angles_file = line.substr(start, line.size() - start);
    start = _angles_file.find_first_not_of ( " " );
    _angles_file = _angles_file.substr(start, _angles_file.size() - start);
    energy_parameters_read++;
  }
  return energy_parameters_read;
}//set_database

void
Input_data::set_agents ( string line ) {
  char * pch;
  int value;
  static int lw_priority = LOWER_PRIORITY/2;
  if ( line.compare( 0, 2, "H " ) == 0 ||
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
}//set_agents

void
Input_data::set_dock_constraints ( string line ) {
  if (line.compare( 0, 9, "TRANSLATE" ) == 0 ) {
    gh_params.translate_str = true;
    size_t found = (line.substr( 9, line.size() )).find_first_not_of(" ");
    found += 9;
    if ( line[ found+1 ] == ' ') {
      /// N, H, C, O
      gh_params.translation_point[ 0 ] = Utilities::cv_string_to_atom_type( line.substr( found, 2 ) );
      found += 2;
    }
    else {
      /// CA
      gh_params.translation_point[ 0 ] = Utilities::cv_string_to_atom_type( line.substr( found, 3 ) );
      found += 3;
    }
    
    std::stringstream stream( line.substr( found, line.size() ) );
    
    double n;
    int parsed_val = 0;
    while( 1 ) {
      stream >> n;
      if( !stream ) break;
      if ( !parsed_val ) {
        gh_params.translation_point[ 0 ] += n*5;
      }
      else {
        gh_params.translation_point[ parsed_val ] = n;
      }
      parsed_val++;
    }
  }
  if (line.compare( 0, 9, "ATOM_GRID" ) == 0 ) {
    size_t found = (line.substr( 9, line.size() )).find_first_not_of(" ");
    found += 9;
    _atom_grid_file = line.substr(found, line.size() - found);
    gh_params.atom_grid = true;
    g_atom_grid = new AtomGrid ( 1, 0.25 );
    g_atom_grid->fill_grid ( _atom_grid_file );
  }
  if (line.compare( 0, 9, "DOCK_GRID" ) == 0 ) {
    size_t found     = (line.substr( 9, line.size() )).find_first_not_of(" ");
    found += 9;
    size_t found_aux = (line.substr( found, line.size() )).find_first_of(" ");
    string docking_file = line.substr(found, found_aux );
    int contact_distance = atoi((line.substr( found+found_aux+1, line.size() )).c_str());
    if ( !contact_distance ) { contact_distance = 3; }
    contact_distance *= -1;
    /*
     * Contact distance:
     * -3 Angstrom: if two atoms are less than 3 angstrom (but not clash)
     * then there is a contact (i.e., atom_grid returns false).
     */
    g_docking = new AtomGrid ( 1, contact_distance );
    g_docking->fill_grid ( docking_file );
  }
  if (line.compare( 0, 12, "DOCK_CONTACT" ) == 0 ) {
    size_t found     = (line.substr( 12, line.size() )).find_first_not_of(" ");
    found += 12;
    size_t found_aux = (line.substr( found, line.size() )).find_first_of(" ");
    string dock_con_vals = line.substr(found, line.size() );
    // Read values
    istringstream istr( dock_con_vals );

    real peptide_atom;
    real dist_min, dist_max;
    real dock_x, dock_y, dock_z;
    istr >> dist_min >> dist_max >> peptide_atom >> dock_x >> dock_y >> dock_z;
    vector< real > dock_contacts;
    dock_contacts.push_back ( dist_min );
    dock_contacts.push_back ( dist_max );
    dock_contacts.push_back ( peptide_atom );
    dock_contacts.push_back ( dock_x );
    dock_contacts.push_back ( dock_y );
    dock_contacts.push_back ( dock_z );
    
    gh_params.force_contact.push_back ( dock_contacts );
  }
  
}//set_dock_constraints

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
    cout << _dbg << "Set FASTA for target or provide a known protein pdb file.\n";
    exit(2);
  }

  if ( gh_params.verbose ) gh_params.target_protein->print_sequence ();
  
  gh_params.n_res    = gh_params.target_protein->get_nres();
  gh_params.n_points = gh_params.n_res * 15;
  assert ( gh_params.n_res <= MAX_TARGET_SIZE );

  int bb_len = (int) gh_params.target_protein->get_bblen();
  
  if (_know_prot) {
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

/* 
 * @note: this is an auxiliary function.
 * It sets Angles for coordinator agent
 * considering a different set of angles 
 * (i.e., different input format).
 */
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
  gd_params.domain_states = (uint*) malloc ( MAX_DIM * gh_params.n_res * sizeof(uint) );
  gd_params.domain_events = (int*)  malloc ( gh_params.n_res * sizeof(int) );
  gd_params.validity_solutions = (real*) malloc(gh_params.set_size * sizeof( real ));
}//alloc_states

void
Input_data::init_states () {
  uint all_valid = (uint) (pow ( (double)2.0, MAX_DIM ) - 1); /// 4294967295 /\ MAX_DIM = 32
  gh_params.domain_states = (uint*) malloc ( MAX_DIM * gh_params.n_res  * sizeof(uint) );
  gh_params.validity_solutions  = (real*) malloc ( gh_params.set_size * sizeof(real) );
  for (int i = 0; i < MAX_DIM * gh_params.n_res; i++)
    gh_params.domain_states[ i ] = all_valid;
  for (int i = 0; i < gh_params.set_size; i++) gh_params.validity_solutions[ i ] = 1;
  
  memcpy( gd_params.domain_states, gh_params.domain_states,
         MAX_DIM * gh_params.n_res * sizeof( uint ) );
  memcpy( gd_params.validity_solutions, gh_params.validity_solutions,
          gh_params.set_size * sizeof( real ) );
  
}//init_states

void
Input_data::alloc_energy () {
  gd_params.h_distances      = (real*) malloc ( 25*3 * sizeof(real) );
  gd_params.h_angles         = (real*) malloc ( 73*9 * sizeof(real) );
  gd_params.contact_params   = (real*) malloc ( 20*20 * sizeof(real) );
  gd_params.tors             = (real*) malloc ( 20*20*18*3 * sizeof(real) );
  gd_params.tors_corr        = (real*) malloc ( 18*18*5 * sizeof(real) );
  gd_params.aa_seq           = (aminoacid*) malloc ( gh_params.target_protein->get_nres() * sizeof(aminoacid) );
  gd_params.secondary_s_info = (ss_type*) malloc ( gh_params.target_protein->get_nres() * sizeof(ss_type) );
  if ( gh_params.follow_rmsd ) {
    int bb_len = (int) gh_params.target_protein->get_bblen();
    gd_params.known_prot = (real*) malloc ( 3 * bb_len * sizeof(real) );
  }
}//alloc_energy

void
Input_data::init_energy () {
  ss_type* secondary_s_info = ( ss_type* ) malloc ( gh_params.target_protein->get_nres() * sizeof(ss_type) );
  for ( int i = 0; i < gh_params.mas_des.size(); i++ )
    for ( int j = 0; j < gh_params.mas_des[i].vars_list.size(); j++ )
      secondary_s_info[ gh_params.mas_des[i].vars_list[j] ] = gh_params.mas_des[i].sec_str_type;
  
  memcpy( gd_params.secondary_s_info, secondary_s_info,
         gh_params.target_protein->get_nres() * sizeof(ss_type) );
  memcpy( gd_params.h_distances, gh_params.h_distances,
         25*3 * sizeof(real) );
  memcpy( gd_params.h_angles, gh_params.h_angles,
         73*9 * sizeof(real) );
  memcpy( gd_params.contact_params, gh_params.contact_params,
         20*20 * sizeof(real) );
  memcpy( gd_params.tors, gh_params.tors,
         20*20*18*3 * sizeof(real) );
  memcpy( gd_params.tors_corr, gh_params.tors_corr,
         18*18*5 * sizeof(real) );
  memcpy( gd_params.aa_seq, &(gh_params.target_protein->get_sequence_code()[ 0 ]),
         gh_params.target_protein->get_nres() * sizeof(aminoacid) );
  if ( gh_params.follow_rmsd ) {
    int bb_len = (int) gh_params.target_protein->get_bblen();
    memcpy( gd_params.known_prot, gh_params.known_bb_coordinates,
            3 * bb_len * sizeof(real) );
  }
  free ( secondary_s_info );
}//init_energy

void
Input_data::free_dt () {
  Utilities::print_debug ( _dbg, "Free memory on host" );
  free( gd_params.h_distances ) ;
  free( gd_params.h_angles );
  free( gd_params.contact_params );
  free( gd_params.tors );
  free( gd_params.tors_corr );
  free( gd_params.aa_seq );
  free( gd_params.domain_states );
  free( gd_params.domain_events );
  free( gd_params.secondary_s_info );
  if ( gh_params.follow_rmsd )
    free( gd_params.known_prot );
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
  cout << "\n" << " ---- ";
  if ( gh_params.sys_job == ab_initio ) {
    cout << "Ab_initio";
  }
  else {
    cout << "Docking";
  }
  cout << " ---- " << endl;
  cout << endl;
}//dump

void
Input_data::print_help () {
  string spaces = "        ";
  cout << "Usage: ./cocos -i <infile> [options]\n" << endl;
  cout << "            Options            |           Description      \n";
  cout << "============================== | ==============================\n";
  cout << " --rmsd                        | - Use RMSD as obj function.\n";
  cout << " --auto_allign                 | - Automatic allignment of\n";
  cout << "                               |   secondary structures.\n";
  cout << " --cg_constraint               | - Set CG constraint.\n";
  cout << " --gibbs_default               | - Use Gibbs as default for\n";
  cout << "                               |   coordinators agents.\n";
  cout << " --translate                   | - Translate 2nd atom of\n";
  cout << "                               |   prediction on (0, 0, 0).\n";
  cout << " -r|--random_moves             | - Perform random translations\n";
  cout << "                               |   and rotations of the\n";
  cout << "                               |   peptide when in docking\n";
  cout << "                               |   analysis.\n";
  cout << " -v|--verbose                  | - Printf verbose info\n";
  cout << "                               |   during computation.\n";
  cout << " -h|--help                     | - Print this help message.\n";
  cout << " -i|--input      (string)      | - Read and set input.\n";
  cout << " -o|--output     (string)      | - Set output file.\n";
  cout << " -a|--angles     (integer)     | - Set variables' domains\n";
  cout << "                               |   partitioning [-180, +180]\n";
  cout << "                               |   as specified (deg).\n";
  cout << " -s|--set_size   (integer)     | - Set size of sampling sets.\n";
  cout << " -c|--mc_timeout (integer)     | - Set timeout for \n";
  cout << "                               |   MonteCarlo sampling.\n";
  cout << " -k|--docking    (integer)     | - Set minimum number of\n";
  cout << "                               |   contacts for docking.\n";
  cout << "                               |   Default: 4.\n";
  cout << " -g|--gibbs      (integer)     | - Set number of Gibbs\n";
  cout << "                               |   samples.\n";
  cout << " -t|--gb_iterations            | - Set number of iterations\n";
  cout << "                 (integer)     |   before swapping bins in\n";
  cout << "                               |   Gibbs sampling.\n";
  cout << "=============================  | =============================\n";
  cout << "You may want to try:\n";
  cout << "\t" << "./cocos -i proteins/1ZDD.in.cocos -v\n";
  cout << "Other examples, input data, and structures are present in the folder \"protein\".\n";
  cout << "For any questions, feel free to write at: campe8@nmsu.edu.\n";
}//print_help


