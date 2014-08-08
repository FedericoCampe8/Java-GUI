#ifndef COCOS_INPUT_DATA__
#define COCOS_INPUT_DATA__

#include "globals.h"
#include "typedefs.h"
#include "protein.h"

class Input_data {
private:
  std::string _dbg;
  bool        _know_prot;
  std::string _in_file;
  std::string _out_file;
  std::string _known_prot_file;
  std::string _target_prot_file;
  std::string _target_sequence;
  std::string _angles_file;
  std::string _energy_charges;    // Coulomb potentials
  std::string _lj_params;         // Lenard-Jones parameters
  std::string _h_distances;       // Hydrogen bond energy parameters
  std::string _h_angles;          // Hydrogen bond energy parameters
  std::string _contact_params;    // Contact energy parameters
  std::string _tors_params;       // Torsional energy parameters
  
  /// Domains
  void read_file ();
  void load_angles ();
  void load_angles_aux ();
  /// Energy
  int  convert_aa_pos( int aa );
  void init_energy_tables ();
  void read_energy_parameters ( std::string file_name, std::vector< std::vector<real> >& );
  void read_torsional_parameters ( std::string file_name, real tors_param[20][20][20][3] );
  /// Utilities
  void create_input_file ();
  void alloc_states ();
  void alloc_energy ();
  void init_states ();
  void init_energy ();
  void free_dt ();
  /// Other
  void print_help ();
  void print_gpu_info ();
public:
  /// Constructor 
  Input_data ( int argc, char* argv[] );
  ~Input_data ();
  
  bool know_prot () const;
  void set_target_sequence ( std::string t_seq );
  std::string get_target_sequence () const;
  void init_data ();
  void alloc_constraints ();
  void clear_data ();
  
  void dump ();
}; //input_data

#endif
