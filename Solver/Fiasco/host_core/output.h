/* 
 * Object: Output
 * Process the outputs of the program.
 * Bufferize protein solutions and/or output them on a file 
 */
#ifndef FIASCO_OUTPUT__
#define FIASCO_OUTPUT__

#include "typedefs.h"

#include <string>
#include <vector>

class Atom;

/* Atom - all members are public */
class Output {
 private:
  std::vector<std::string> buffer;
  std::string file_out;
  size_t model_ct;
  std::vector<real> energy;
  
 public:
  Output(int argc, char* argv[]);
  void store_results( real energy = 1000 );
  void store_results( real rmsd, real energy = 1000 );
  void store_best_results( real energy = 1000 );
  void store_best_results( real rmsd, real energy = 1000  );
  void store_results( uint atom_s, uint atom_e, real energy = 1000, real rmsd = -1, bool store_best = false );
  // for debug
  void store_points (std::vector<real> points); 
  void dump();
  void set_out_file (std::string s) {file_out = s;}
  std::string get_out_file () {return file_out;}
  
};

#endif
