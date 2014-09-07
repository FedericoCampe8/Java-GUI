#include "cubic_lattice.h"

#include <iostream>
#include <cmath>
#include <cassert>
#include <cstring>
#include <limits>

//#define DEBUG_CL

CubicLattice::CubicLattice (real _lattice_side, size_t _numof_voxels, 
			    point _center_of_mass, bool _val)
  : Bitset (_numof_voxels+1, _val),
    lattice_side (_lattice_side), // in Anmstrongs
    numof_voxels (_numof_voxels), // volume
    numof_voxels_per_side (ceil(pow(_numof_voxels, (1/3.0)))-1), 
    ratio ((real)((numof_voxels_per_side+1)/(real)_lattice_side)),
    point_to_voxel_ratio (ratio * (_lattice_side/2)) {
  
#ifdef DEBUG_CL
  std::cout << "Lattice side: " << lattice_side  
	    << " Numof voxels: " << numof_voxels
	    << " (per side: " << numof_voxels_per_side
	    << ") \n ratio: " << ratio
	    << " point_to_voxel_ratio: " << point_to_voxel_ratio 
	    << std::endl;
  std::cout << "center: " << _center_of_mass[0] << ", "
	    << _center_of_mass[1] << ", "
	    << _center_of_mass[2] << std::endl;
#endif

  memcpy (center_of_mass, _center_of_mass, sizeof (point));
  
}//-

size_t
CubicLattice::set (const point& p, bool val) {
  uint x = std::abs(p[0] - center_of_mass[0] + lattice_side/2) * ratio;
  uint y = std::abs(p[1] - center_of_mass[1] - lattice_side/2) * ratio;
  uint z = std::abs(p[2] - center_of_mass[2] - lattice_side/2) * ratio;
#ifdef DEBUG_CL
  std::cout << "Setting p: " << p[0] << " " << p[1] << " " << p[2] << std::endl;
  std::cout<< "x= " << p[0] << " + lattice/2 = " << lattice_side/2 << "=" << x << std::endl;
  std::cout<< "y= " << p[1] << " - lattice/2 = " << lattice_side/2 << "=" << y << std::endl;
  std::cout<< "x= " << p[2] << " - lattice/2 = " << lattice_side/2 << "=" << z << std::endl;
  std::cout << "after translation p: " << x << " " << y << " " << z << std::endl;
#endif
  return set (x, y, z);
}//-


size_t
CubicLattice::set (uint x, uint y, uint z, bool val) {
  size_t voxel_to_set = x + y * numof_voxels_per_side 
    + z * (numof_voxels_per_side*numof_voxels_per_side);

#ifdef DEBUG_CL
  std::cout << "nvoxels=" << numof_voxels_per_side << " x=" <<x <<" + y=" << y << " + z=" << z << std::endl;
  std::cout << "CubiLattice::set() voxel no.  " << voxel_to_set << std::endl;
#endif
  // assert (voxel_to_set <= numof_voxels);
  if (voxel_to_set > numof_voxels) 
    return std::numeric_limits<size_t>::max();
  Bitset::set (voxel_to_set, val);
  points_in_lattice.push_back(voxel_to_set);
  return voxel_to_set;
}//-


bool 
CubicLattice::test (const point& p) const {
  uint x = std::abs(p[0] - center_of_mass[0] + lattice_side/2) * ratio;
  uint y = std::abs(p[1] - center_of_mass[1] - lattice_side/2) * ratio;
  uint z = std::abs(p[2] - center_of_mass[2] - lattice_side/2) * ratio;
#ifdef DEBUG_CL
  std::cout << "Testing p: " << p[0] << " " << p[1] << " " << p[2] << std::endl;
  std::cout << "after translation p: " << x << " " << y << " " << z << std::endl;
#endif
  return test (x, y, z);
}//-


bool 
CubicLattice::test (uint x, uint y, uint z) const {
  size_t voxel_to_set = x + y * numof_voxels_per_side 
    + z * (numof_voxels_per_side*numof_voxels_per_side);
#ifdef DEBUG_CL
  std::cout << "CubiLattice::test() voxel no.  " << voxel_to_set << std::endl;
#endif
  
  //  assert (voxel_to_set <= numof_voxels);
  if (voxel_to_set > numof_voxels)
    return true;
  return Bitset::test (voxel_to_set);
}//-


void
CubicLattice::reset () {
  std::vector<size_t>::iterator it =
    points_in_lattice.begin();
  for (; it != points_in_lattice.end(); ++it) {
    Bitset::set (*it, false);
  }
  points_in_lattice.clear();
}//-
