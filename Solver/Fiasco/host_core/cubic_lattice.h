/*********************************************************************
 * object: CubicLattice
 * This object represents a 3 dimensional cubic lattice. 
 * TODO: Based on the number of voxels, and the maximum extension of
 * the chain -- from AA 0 to the marker, build a grid (reals) containing
 * numof_voxels blocks.
 * Hence find a mapping from this grid to the bitset containing truth 
 * values for the voxels.
 *
 * TODO:
 *  more accurate description of the hash function used here
 *********************************************************************/
#ifndef FIASCO_CUBICLATTICE_CONSTRAINT_
#define FIASCO_CUBICLATTICE_CONSTRAINT_

#include "typedefs.h"
#include "bitset.h"
#include <vector>

class CubicLattice : public Bitset {
 private:
  int voxel_side;
  real lattice_side;		// 2*radius of sphere inscribed in this cube 
  size_t numof_voxels;		// total number of voxels in the lattice
  size_t numof_voxels_per_side;  // maybe not needed
  const real ratio;
  const real point_to_voxel_ratio;
  point center_of_mass;
  
  std::vector<size_t> points_in_lattice; // AUX D.S. to handle efficient 
				         // reset operations

 public:
  CubicLattice ();
  CubicLattice (real _lattice_side, size_t _numof_voxels, 
		point center_of_mass, bool val=false);
  
  size_t set (uint x, uint y, uint z, bool val = true);
  size_t set (const point& p, bool val = true);
  bool test (uint x, uint y, uint z) const;
  bool test (const point& p) const;
  void reset ();

  real get_lattice_side () const {return lattice_side; }
  size_t get_numof_voxels () const {return numof_voxels; }
  size_t get_numof_voxels_per_side () const {return numof_voxels_per_side; }
  //  void dump() {std::cout << "aa\n";};
};

#endif
