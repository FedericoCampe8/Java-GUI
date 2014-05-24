/*********************************************************************
 * Authors: Ferdinando Fioretto (ffiorett@cs.nmsu.edu)               *
 *          Federico Campeotto                                       *
 *          Alessandro Dal Palu', Enrico Pontelli, Agostino Dovier   *
 * (C) Copyright 2010-2011                                           *
 *                                                                   *
 * This file is part of FIASCO.                                      *
 *                                                                   *
 * FIASCO is free software; you can redistribute it and/or           *
 * modify it under the terms of the GNU General Public License       *
 * as published by the Free Software Foundation;                     *
 *                                                                   *
 * FIASCO is distributed WITHOUT ANY WARRANTY; without even the      *
 * implied  warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR  *
 * PURPOSE. See the GNU General Public License for more details.     *
 *                                                                   *
 * You should have received a copy of the GNU General Public License *
 * along with this program; if not, see http://www.gnu.org/licenses. *
 *                                                                   *
 *********************************************************************/
#ifndef FIASCO_PROTEIN__
#define FIASCO_PROTEIN__

#include "typedefs.h"


#include <string>
#include <vector>
#include <iostream>

class Atom;

class Sheet {
 private:
  uint aa_s;
  uint aa_e;
  // other stuff here
 public:
  Sheet (int a1, int a2) : aa_s (a1), aa_e (a2) {};
};

class Helix {
 private:
  uint aa_s; 
  uint aa_e;
  // other stuff here
  std::string helix_name;
 public:
  Helix (uint a1, uint a2, std::string name="") 
    : aa_s (a1), aa_e (a2), helix_name(name) {};
  
  std::string get_name () {return helix_name; }
  std::pair<uint, uint> get_aa_bounds() {return std::make_pair(aa_s, aa_e); }
};

class SSbond {
 private:
  uint aa_s;
  uint aa_e;
 public:
  SSbond (uint s1, uint s2) 
    : aa_s (s1), aa_e (s2) {};
};

class Loop {
 private:
  uint atom_s;
  uint atom_e;
  std::string loop_name;
 public:
  Loop (uint a1, uint a2, std::string name="") 
    : atom_s (a1), atom_e (a2), loop_name (name) {};
  std::string get_name () {return loop_name; }
  std::pair<uint, uint> get_bb_bounds () const {
    return std::make_pair (atom_s, atom_e); 
  }//-

};

/* Manage the protein target representation */
class Protein {
 private:
  std::string id;
  std::string name;
  uint nres;			// protein length
  uint offset;			// offset related to the first AA number read


  std::vector<Helix> helices;
  std::vector<Sheet> sheets;
  std::vector<SSbond> ssbonds;
  std::vector<Loop> loops;
  real volume;

 public:
  std::string sequence;
  std::vector<aminoacid> seq_code; // AA code for accensing energy table
  //- TODO this shoudl be stored as an aminoacid
  std::vector<Atom> backbone;
  std::vector<std::vector<Atom> > side_chains;
  //-

  Protein() {}
  Protein(std::string filename, std::string chain = "A", std::string pdb_code = "none");
  Protein (const Protein& other); 
  Protein& operator= (const Protein& other); 
  ~Protein() {};
  Atom& operator[]  (const int index) {return backbone[index];}

  std::vector<Atom> get_backbone(){ return backbone; }
  void set_id (std::string pid) {id = pid;}
  std::string get_id() const {return id;}
  void set_name (std::string pname) {name = pname;}
  std::string get_name() const {return name;}
  void set_nres(int n) {nres = n;}  
  uint get_nres() const {return nres;}
  uint get_bblen() const {return backbone.size();}
  void set_offset (int off) {offset = off;}
  uint get_offset() const {return offset;}
  void load_protein (std::string filename, 
		     std::string chain = "A",
		     std::string pdb_code = "none");
  void dump ();
 

  // Loops
  size_t numof_loops() {return loops.size(); }
  void set_loop (uint s, uint e, std::string loop_name);
  Loop* get_loop (uint idx);
  Loop* get_loop (std::string loop_name);
  
  // Helices
  size_t numof_helices() {return helices.size(); }
  void set_helix (uint aa_s, uint aa_e, std::string helix_name="");
  Helix* get_helix (uint idx);
  Helix* get_helix (std::string helix_name);

};

#endif
