/*********************************************************************
 * Fiasco Utilities
 * This Namespace contains common utility functions used by FIASCO
 *********************************************************************/
#ifndef FIASCO_UTILITIES__
#define FIASCO_UTILITIES__

#include "typedefs.h"
#include "hilbert.h"

#include <iostream>
#include <vector>
#include <string>

#define NITROGEN_ATOM 0
#define CALPHA_ATOM   1
#define CBETA_ATOM    2
#define OXYGEN_ATOM   3
#define CARBON_ATOM   4
#define SULPHUR_ATOM  5
#define CENTROID_ATOM 6
#define HYDROGEN_ATOM 7
#define OTHER_ATOM    8

class Fragment;
class Protein;
class Atom;

namespace Utilities{
  /* conversion tools */
  std::string cv_aa1_to_aa3(char a);
  std::string cv_aa3_to_aa1(std::string a);
  std::string cv_class_to_aa3(aminoacid a);
  char cv_class_to_aa1(aminoacid a);
  aminoacid cv_aa_to_class(char a);
  aminoacid cv_aa_to_class(std::string a);
  int cv_class_to_n ( aminoacid a );

  bitmask_t convert_point_to_hilbert_value (const point& p);
  void convert_hilbert_value_to_point (const bitmask_t& hilbert_value, point& p);

  // energy 
  int get_bin(int angle);
  real get_tors_angle(point p1, point p2, point p3, point p4);

  // I/O aux functions
  void populate_fragment_assembly_db 
    (std::vector<Fragment>& fragment_set, int fragment_len, std::string filename);
  void populate_fragment_multiple_assembly_db 
    (std::vector< std::vector<Fragment> >& fragment_set, int fragment_len, std::string filename);
  void output_pdb_format 
    (std::string outf, const size_t id, const Protein& P, real rmsd);
  void output_pdb_format (std::string outf, const std::vector<Atom>& vec);
  void init_file(std::string outf);
  size_t count_active 
    (std::vector<std::vector <bool> > vec, uint lev_s, uint lev_e);
  void usage();
  
  // Clear
  void clear (std::vector<int>&);
  void clear (std::vector< std::vector<int> >&);
  void clear (std::vector<real>&);
  void clear (std::vector< std::vector<real> >&);
  void clear (std::vector<bool>&);
  void clear (std::vector< std::vector<bool> >&);
  
  // Display
  void dump(const point& a, const std::string prefix="");
  void dump(const point& a, const point& b, const std::string prefix="");
  void dump(const std::vector<bool> b);
  void dump(const std::vector<int> n);
  void dump(const std::vector<int> v, int s, int e);
  void dump(const std::vector<bool> v, int s, int e);
  //void dump(const std::vector<aminoacid> v, int s, int e);
  void dump(const R_MAT r);

  atom_type  get_atom_type(uint bbidx);
  atom_type  get_atom_type(std::string str);
  atom_radii get_atom_radii(uint bbidx);
  atom_radii get_atom_radii (atom_type t);
  
  int get_bbidx_from_aaidx (uint aaidx, atom_type type);
  int get_aaidx_from_bbidx (uint bbidx, atom_type type);

}//-

#endif
