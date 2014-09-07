/*********************************************************************
 * Atom Grid
 *********************************************************************/
#ifndef FIASCO_ATOM_GRID__
#define FIASCO_ATOM_GRID__

#include "typedefs.h"

#include <vector>

class Atom;
class VariablePoint;

struct AtomGridCell {
  AtomGridCell ();  
  ~AtomGridCell () {};
  AtomGridCell (const AtomGridCell& other);
  AtomGridCell& operator= (const AtomGridCell& other);

  std::vector<Atom> atom_list; // as a map
  size_t size;  // actual no of atom stored
};


class AtomGrid {
 private:
  static constexpr int grid_side = 3; // the grid side (in Angstrom)
  static constexpr int grid_edge = 128;
  int grid_max_dist;
  static constexpr real prot_Ca_Ca = 3.2; //static constexpr
  static constexpr real prot_Cg_Cg = 1.0; //static constexpr

  // converts cell coordinates to linear index
  size_t convert_cell_to_key(int x, int y, int z);
  size_t convert_pos_to_key(point p);  
  
 public:
  std::vector<AtomGridCell> space;
  
  AtomGrid (int maxdist=1);
  ~AtomGrid() {};
  AtomGrid (const AtomGrid& other);
  AtomGrid& operator= (const AtomGrid& other);
  
  void reset ();
  void remove(int idx);
  void allocate_more_atoms(int idx);
  void add (point p, atom_type t, int atom_idx);
  bool query (const point& vp, atom_type type, int ref_aa);
  bool query(const VariablePoint *vp);
  bool query(const Atom& a);
};
#endif
