#include "rmsd_energy.h"
#include "utilities.h"
#include "mathematics.h"
#include "rmsd_fast.h"

using namespace std;
using namespace Utilities;
using namespace Math;

RmsdEnergy *
RmsdEnergy::set_parameters ( real * known_prot ) {
  _known_prot = known_prot;
 
  return this;
}//set_parameters

void
RmsdEnergy::calculate_energy ( real* setOfStructures, real* setOfEnergies,
                               real* validStructures, int n_res,
                               int bb_start, int bb_end,
                               int scope_start, int scope_end,
                               int n_bytes, int n_blocks, int n_threads ) {
  int num_of_res = scope_start - scope_end + 1;
  Rmsd_fast::get_rmsd( setOfStructures, setOfEnergies,
                       validStructures, _known_prot,
                       num_of_res, scope_start, scope_end, n_blocks );
}//calculate_energy


