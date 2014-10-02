#include "utilities.h"
#include "protein.h"
#include "atom.h"

#include <cassert>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <stdlib.h>

using namespace std;
using namespace Utilities;

Protein::Protein(string filename, string chain, string pdb_code)  {
  load_protein (filename, chain, pdb_code);
}
//-

// Given a string name return the associated atom type
atom_type get_atom_type_from_pdb (string name) {
  if (name.find(" N  ") != string::npos ||
      name.find(" N A") != string::npos) 
    return N;
  else if (name.find(" CA  ") != string::npos ||
           name.find(" CA A") != string::npos ||
           name.find(" CA B") != string::npos) 
    return CA;
  else if (name.find(" C  ") != string::npos ||
           name.find(" C  A") != string::npos) 
    return CB;
  else if (name.find(" O  ") != string::npos ||
           name.find(" O  A") != string::npos) 
    return O;  
  else if (name.find(" H  ") != string::npos ||
           name.find(" H  A") != string::npos) 
    return H;
  else if (name.find(" S  ") != string::npos ||
           name.find(" S  A") != string::npos) 
    return S;
  else return X;
}

Protein::Protein (const Protein& other) {
  id = other.id;
  name = other.name;
  nres = other.nres;
  offset = other.offset;
  // TODO: this has been done for the DB generation
  // helices = other.helices;
  // sheets = other.sheets;
  // ssbonds = other.ssbonds;
  // loops  = other.loops;
  volume = other.volume;
  sequence = other.sequence;
  seq_code = other.seq_code;
  backbone = other.backbone;
  side_chains = other.side_chains;
}//-


Protein&
Protein::operator= (const Protein& other) {
  if (this != &other) {
    id = other.id;
    name = other.name;
    nres = other.nres;
    offset = other.offset;
    // TODO: this has been done for the DB generation
    // helices = other.helices;
    // sheets = other.sheets;
    // ssbonds = other.ssbonds;
    // loops  = other.loops;
    volume = other.volume;
    sequence = other.sequence;
    seq_code = other.seq_code;
    backbone = other.backbone;
    side_chains = other.side_chains;
  }
  return *this;
}//-


void 
Protein::load_protein (string filename, string chain, string pdb_code) {
  // tokens
  const string dbref = "DBREF ";
  const string sheet = "SHEET ";
  const string helix = "HELIX ";
  const string ssbond = "SSBOND";
  const string anisou = "ANISOU";
  const string atom  = "ATOM  ";
  const string seq   = "SEQRES";
  const string end   = "ENDMDL";
  string aa;
  string line, token, buf;
  int a1, a2;
  aminoacid c;
  real x, y, z;
  atom_type type = X;
  flag ok = 0;

    ifstream target ( filename.c_str() );
    if ( !target.is_open() ) {
        string new_file;
        int ctr = 0;
        for ( auto c : filename ) {
            if ( c == '.') {
                if ( ctr > 0 &&
                    ctr != (filename.size() - 1) &&
                    filename[ctr-1] != '.' &&
                    filename[ctr+1] != '.' ) {
                    break;
                }
            }
            ctr++;
            new_file += c;
        }
        new_file += ".pdb";
        
        target.open( new_file );
        if ( !target.is_open() ) {
            cout << "Error: Unable to open the file " << new_file << " in loading the target protein" << endl;
            return;
        }
    }
    
  // string name;
  name = filename.substr(0, filename.size() - 4);
  set_name (pdb_code);
  nres=0;
  std::vector<Atom> side_chain;

  while (target.good()) {
    getline(target, line);
    token = line.substr(0, 6);
    
    if (token == dbref) {
      id = line.substr(7,4);
    }
    else if (token == sheet) {
      buf = line.substr(23,3);
      a1 = atoi (buf.c_str());
      buf = line.substr(34,3);
      a2 = atoi (buf.c_str());
      Sheet h(a1, a2);
      sheets.push_back(h);
    }
    else if (token == helix) {
      buf = line.substr(22,3);
      a1 = atoi (buf.c_str());
      buf = line.substr(34,3);
      a2 = atoi (buf.c_str());
      Helix h(a1, a2);
      helices.push_back(h);
    }
    else if (token == ssbond) {
      buf = line.substr(18,3);
      a1 = atoi (buf.c_str());
      buf = line.substr(32,3);
      a2 = atoi (buf.c_str());
      SSbond h(a1, a2);
      ssbonds.push_back(h);
    }
    // else if (token == seq) { 
    //   for (uint ii=0; ii< ceil((float)line.substr(19).size()/4); ii++) {
    // 	if (ii<13 && 19+(ii*4)+3 <= line.size()) {
    // 	  aa = line.substr(19+(ii*4), 3);
    // 	  if (aa.compare("   ") != 0 && aa.size()==3) {
    // 	    if (cv_aa3_to_aa1(aa).compare("err")) {
    // 	      sequence.append(cv_aa3_to_aa1(aa));
    // 	      c = cv_aa_to_class(aa);
    // 	      if (c != err) seq_code.push_back(c);
    // 	    }
    // 	  }
    // 	}
    //   }
    // }
    else if (token == anisou) {
      // skip
    }
    else if (token == end){
      break;
    }  
    else if (token == atom) {
      ok = 0;
      
      // atom name      // stupid check -- HACK!
      buf = line.substr(12,5);
        
      if (get_atom_type_from_pdb (buf) == type) 
	continue;
        
      type = get_atom_type_from_pdb (buf);
     
      // check alt_location
      buf = line.substr(16,1);
      if(buf == " " || buf == "A") ok = 1;
      
      // Protein chain 
      buf = line.substr(21,1); // chain
      ok *= (buf == chain) ? 1 : 0;

      // offset protein 
      if (nres == 0) 
          offset = atoi (line.substr(22,5).c_str());

      buf = line.substr(26,1); // res_ins
        
      // coordinates
      x = atof(line.substr(30,8).c_str());
      y = atof(line.substr(38,8).c_str());
      z = atof(line.substr(46,8).c_str());

      // occupancy
      // buf = line.substr(54,6);
        
      if (ok) {
	// Backbone
	if (type == N || type == CA || type == CB || type == O){
	  Atom a (x, y, z, type, nres);
	  backbone.push_back(a);

	  if (type == CA) {
	    // push back side chain of previous level
	    if (!side_chains.empty()) {
	      side_chains.push_back(side_chain);
	      side_chain.clear();
	    }
	    //-

	    buf = line.substr(17,3);	    // amino acid name

	    if (cv_aa3_to_aa1(buf).compare("err")) {
	      sequence.append(cv_aa3_to_aa1(buf));
	      c = cv_aa_to_class(buf);
	      if (c != err) 
		seq_code.push_back(c);
	    }
	    nres++;
	  }
	}
	// Side chain
	else {
	  if (nres == 0) continue;
	  Atom a (x, y, z, type);
	  side_chain.push_back (a);
	}
	//-
      }
    }
  }
  //-end-while

  // push back last side chain
  if (!side_chain.empty())
    side_chains.push_back(side_chain);
  //-

  target.close();
}//-

void
Protein::set_loop (uint s, uint e, std::string loop_name) {
  Loop l(s, e, loop_name); 
  loops.push_back(l);
}
//-

Loop* 
Protein::get_loop (uint idx) {
  return (idx < loops.size()) ? 
    &loops[idx] : NULL; 
}
//-

Loop* 
Protein::get_loop (std::string loop_name) {
  for (uint i = 0; i < loops.size(); i++) {
    if (loops[i].get_name() == loop_name)
      return &loops[i];
  }
  return NULL;
}
//-

void 
Protein::set_helix (uint aa_s, uint aa_e, std::string helix_name) {
  Helix h (aa_s, aa_e, helix_name); 
  helices.push_back(h); 
}
//-

Helix* 
Protein::get_helix (uint idx) {
  return (idx < helices.size()) ? 
    &helices[idx] : NULL ; 
}
//-

Helix* 
Protein::get_helix (std::string helix_name) {
  for (uint i = 0; i < helices.size(); i++) {
    if (helices[i].get_name() == helix_name)
      return &helices[i];
  }
  return NULL;
}
//-

void 
Protein::dump() {
  cout << "Protein: " << name << " (id: " << id << ") :: ";
  cout << "Seq: " << sequence << endl;
  //Utilities::dump(aaclass_seq);   
  cout << "N.res: " << nres << " backbone len: " << get_bblen() << endl;
  for (uint i=0; i<backbone.size(); i++) 
    backbone[i].dump();
}//-
