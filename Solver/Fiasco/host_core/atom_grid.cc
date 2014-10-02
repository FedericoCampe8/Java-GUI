#include "atom_grid.h"
#include "utilities.h"
#include "globals.h"
#include "mathematics.h"
#include "trailstack.h"
#include "variable_point.h"

#include <cassert>
#include <cmath>

using namespace std;
using namespace Utilities;
using namespace Math;

AtomGridCell::AtomGridCell() : size(0) {
}//-

AtomGridCell::AtomGridCell (const AtomGridCell& other) {
  atom_list = other.atom_list;
  size = other.size;
}//-

AtomGridCell&
AtomGridCell::operator= (const AtomGridCell& other) {
  if (this != &other) { 
    atom_list = other.atom_list;
    size = other.size;
  }
  return *this;
}//-


AtomGrid::AtomGrid(int maxdist) :
  grid_max_dist (maxdist) {
  space.resize(grid_edge*grid_edge*grid_edge);
}//-

AtomGrid::AtomGrid (const AtomGrid& other) {
  grid_max_dist = other.grid_max_dist;
  space = other.space;
}//-

AtomGrid& 
AtomGrid::operator= (const AtomGrid& other) {
  if (this != &other) {
    grid_max_dist = other.grid_max_dist;
    space = other.space;
  }
  return *this;
}//-

void 
AtomGrid::reset() {
  for (uint i=0; i<space.size(); i++) {
    space[i].atom_list.clear();
    space[i].size = 0;
  }
}//-

void
AtomGrid::remove(int idx) {
   space.at(idx).size--;
}//

void 
AtomGrid::add (point p, atom_type _type, int _ref_aa) {
  size_t idx = convert_pos_to_key (p);
  if (space.at(idx).size >= space.at(idx).atom_list.size())
    allocate_more_atoms (idx);
  int k = space.at(idx).size;
  space[idx].atom_list[k].set_position(p);
  space[idx].atom_list[k].ref_aa = _ref_aa;
  space[idx].atom_list[k].set_type(_type);
  space[idx].size++;
  g_trailstack.trail_gridpoint (idx);  
}//-

void
AtomGrid::allocate_more_atoms (int idx){
  Atom a;
  space.at(idx).atom_list.push_back (a);
}//-

bool
AtomGrid::query (const Atom& a) {
  return query (a.position, a.type, a.ref_aa);
}//

bool
AtomGrid::query(const VariablePoint *vp) {
  int id = vp->idx ();
  atom_type type = Utilities::get_atom_type(id);
  int aa_idx = Utilities::get_aaidx_from_bbidx(id, type);
  return query(vp->lower_bound, type, aa_idx);
}//-

//@todo: Investigate on Hash function
bool
AtomGrid::query (const point& vp, atom_type type, int ref_aa) {
  int x=(int)floor(vp[0]/grid_side);
  int y=(int)floor(vp[1]/grid_side);
  int z=(int)floor(vp[2]/grid_side);
  int d = grid_max_dist;
  real epsilon = 30;
  size_t a = 0;
  size_t idx, natoms;
  real distx, disty, distz, dist, limit;
  atom_radii radius = Utilities::get_atom_radii (type);

  // Look in the neighborhood 
  for(int i=x-d; i<=x+d; i++)
    for(int j=y-d; j<=y+d; j++)
      for(int k=z-d; k<=z+d; k++) {
        idx = convert_cell_to_key (i,j,k);
        
        /* 
         * For each atom in the cell: 
         * test distance the minimum intra-distance
         * between itself and its neighborhood
         */
        natoms = space.at(idx).size;
        for(a=0; a < natoms; a++) {

	  int other_ref_aa = 
	    space[idx].atom_list[a].ref_aa;

          if(abs(other_ref_aa - ref_aa) > 2) {
            distx = space[idx].atom_list[a][0] - vp[0];
            disty = space[idx].atom_list[a][1] - vp[1];
            distz = space[idx].atom_list[a][2] - vp[2];
            dist = sqrt(distx*distx + disty*disty + distz*distz);

        if(dist <= 0) continue;
              
	    atom_type other_type = 
	      space[idx].atom_list[a].type;
            atom_radii other_radius = 
	      space[idx].atom_list[a].radius;
	    
            limit = (type == CG || other_type == CG) ? 
                    ((radius + other_radius) - epsilon)/2 :
                    (radius + other_radius) - epsilon;
	   
            if(dist*(100) < limit)
              return false;
          }
        }
      }
  return true;
}//-

/***************************************************
 *                                                 *
 *              Auxiliary functions                *
 *                                                 *
 ***************************************************/
size_t
AtomGrid::convert_cell_to_key(int x, int y, int z) {
  return 
  (x+grid_edge/2)+
  (y+grid_edge/2)*grid_edge+
  (z+grid_edge/2)*grid_edge*grid_edge;
}//-

size_t
AtomGrid::convert_pos_to_key(point p) {
  int x,y,z;
  x=(int)floor(p[0]/grid_side);
  y=(int)floor(p[1]/grid_side);
  z=(int)floor(p[2]/grid_side);
  return convert_cell_to_key(x,y,z);
}//-
