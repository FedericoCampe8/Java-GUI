/*********************************************************************
 * INPUT DATA 
 * 
 * Saves information about the inputs 
 *********************************************************************/
#ifndef INPUT_DATA__
#define INPUT_DATA__

#include "typedefs.h"
#include "protein.h"
#include "fragment.h"

#include <string>
#include <vector>

using namespace std;

class Input_data {
 private:
  /* Data files (files' name) */
  string fragmentdb;	 // fragment database  
  string fragmentdb1;	 // fragment database length 1
  string fragmentdb2;	 // fragment database length 2
  string fragmentdb3;	 // fragment database length 3
  string fragmentdb4;	 // fragment database length 4 
  string energydb;	 // energy table
  string classdb;	 // amino acid class definitions
  string constraint_file; // protein constraints
  string frag_spec_file; // file containing special fragments
  string torsdbpath;     // base path for the torsional database 
  string corrdb;         // tors correlation in the base path
  string target_prot_file;
  string known_prot_file;
  string known_prot_chain;
  int multipleFragmentdb;  
  
  /* Mpi parameters */
  int pid;	    // Current process id
  int nagents;	    // Number of processes in the pool

 public:
  Protein target;
  Protein known_prot;

  /* Constructor, destructor */
  Input_data (int argc, char* argv[]);

  /* Methods */
  std::string get_known_prot_chain() const {return known_prot_chain; }
  string get_fragmentdb() const {return fragmentdb;}
  void set_fragmentdb(const string frgDb){
    fragmentdb = frgDb;
    multipleFragmentdb = 0;
  }
  void set_fragmentdb_l(const string frgDb, int n);
  string get_fragmentdb_l(int i) const;
  int get_multipleFragmentdb () const {return multipleFragmentdb;}  
  bool is_multipleFragmentdb () const {return (multipleFragmentdb > 0);} 
  string get_energydb() const {return energydb;}
  void set_energydb(const string enrDb) {energydb = enrDb;}
  string get_classdb() const {return classdb;}
  void set_classdb(const string clsDb){classdb = clsDb;}
  string get_constraintfile() const {return constraint_file;}
  void set_constraintfile(const string cstFl) {constraint_file = cstFl;}
  string get_torsdbpath() const {return torsdbpath;}
  void set_torsdbpath(const string trsDbPath) {torsdbpath = trsDbPath;}
  string get_corrdb() const {return corrdb;}
  void set_corrdb(const string crDb){corrdb = crDb;}
  string get_target_prot_file(){return target_prot_file;}
  string get_known_prot_file(){return known_prot_file;}
  
  /* Read the file with info about the others files */
  void read_files(const string filename);

  void load_constraint_file (vector<Fragment>& fragment_set);
  void dump_log();

}; //input_data

#endif
