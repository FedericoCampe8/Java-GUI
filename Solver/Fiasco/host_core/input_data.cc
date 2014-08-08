#include "input_data.h"
#include "utilities.h"
#include "globals.h"
#include "atom.h"


#include "variable_point.h"
#include "logic_variables.h"
#include "output.h"

#include <iostream>
#include <cassert>
#include <vector>
#include <fstream>
#include <string>
#include <sstream>
#include <stdlib.h> // for atoi

#include <iomanip>

using namespace std;
using namespace Utilities;


Input_data::Input_data (int argc, char* argv[]) 
  : known_prot_chain ("A") { 

  string inputfile;

  for (int narg=0; narg < argc; narg++) {
    if (!strcmp ("--input",argv[narg]) || 
	!strcmp("-i", argv[narg])) {
      inputfile = argv[narg + 1];
      break;
    }
  }

  multipleFragmentdb = 0; 
  // Load inputs
  read_files (inputfile);
  load_constraint_file (g_assembly_db);
}//-

void 
Input_data::set_fragmentdb_l(const string frgDb, int n){
  switch(n){
  case(1):
    fragmentdb1 = frgDb;
    if(1 > multipleFragmentdb) 
      multipleFragmentdb = 1;
    break;
  case(2):
    fragmentdb2 = frgDb;
    if(2 > multipleFragmentdb) 
      multipleFragmentdb = 2;
    break;
  case(3):
    fragmentdb3 = frgDb;
    if(3 > multipleFragmentdb) 
      multipleFragmentdb = 3;
    break;
  case(4):
    fragmentdb4 = frgDb;
    if(4 > multipleFragmentdb) 
      multipleFragmentdb = 4;
    break;
  default:
    fragmentdb = frgDb;
    if(multipleFragmentdb < 1)
      multipleFragmentdb = 0;
  }
}//-

string
Input_data::get_fragmentdb_l(int i) const {
  switch(i){
  case(1):
    return fragmentdb1;
    break;
  case(2):
    return fragmentdb2;
    break;
  case(3):
    return fragmentdb3;
    break;
  case(4):
    return fragmentdb4;
    break;
  default:
    return fragmentdb;
  }
}//-


// TODO:
// read from _CON e _PDB file and save info in the approriate fields
void
Input_data::read_files (std::string filename) {
  ifstream inputFile;
  std::string line;
  char * fname;
  int start = 12; // parameter from the format of the input file
  fname = new char[filename.size() + 1];
  strcpy(fname, filename.c_str());

  /* Open the file (ios::in) */
  inputFile.open(fname);
  if(inputFile.is_open()){
    while (inputFile.good()){
      getline (inputFile,line);
      if(line.compare(0, 10, "KNOWN_PROT") == 0) {
	known_prot_file = line.substr(start, line.size() - start);
	known_prot_file += ".pdb";
      }
      if(line.compare(0, 11, "TARGET_PROT") == 0) {
	target_prot_file = line.substr(start, line.size() - start);
	target_prot_file += ".pdb";
      }
      if(line.compare(0,5, "CHAIN") == 0) {
	known_prot_chain = line.substr(start, start+1);
      }
      if(line.compare(0, 10, "FRAGMENTDB") == 0)
	set_fragmentdb(line.substr(start, line.size() - start));
      if(line.compare(0, 11, "FRAGMENTDB1") == 0)
	set_fragmentdb_l(line.substr(start+1, line.size() - start), 1);
      if(line.compare(0, 11, "FRAGMENTDB2") == 0)
	set_fragmentdb_l(line.substr(start+1, line.size() - start), 2);
      if(line.compare(0, 11, "FRAGMENTDB3") == 0)
	set_fragmentdb_l(line.substr(start+1, line.size() - start), 3);
      if(line.compare(0, 11, "FRAGMENTDB4") == 0)
	set_fragmentdb_l(line.substr(start+1, line.size() - start), 4);
      if(line.compare(0, 8, "ENERGYDB") == 0)
	set_energydb(line.substr(start, line.size() - start));
      if(line.compare(0, 7, "CLASSDB") == 0)
	set_classdb(line.substr(start, line.size() - start));
      if(line.compare(0, 10, "TORSDBPATH") == 0)
	set_torsdbpath(line.substr(start, line.size() - start));
      if(line.compare(0, 6, "CORRDB") == 0)
	set_corrdb(line.substr(start, line.size() - start));
      if(line.compare(0, 11, "CONSTRAINTS") == 0)
	constraint_file = line.substr(start, line.size() - start);
      if(line.compare(0, 11, "FRAG_SEC_FL") == 0)
	frag_spec_file = line.substr(start, line.size() - start);
    }
    inputFile.close();
  }else {
    cout << "Unable to open file" << filename << endl;
  }
  delete[] fname;
}//Input_data


typedef struct special_fragment_{
  string pid;    // original protein
  int frag_no;   // how each subfragment is coded in the input file 
  // (-1 if alone -> renumbered afterwards)
  uint offset_pdb; // reference to pdb file
  uint bb_start, bb_end; // placement on target
  uint aa_start, aa_end; // aa start and end at the target
  
  vector<int> bundle; // bundle constraint
  int link;     // ptr to fragment set for this fragment
} special_fragment;    


// note:
// at this stage this code only manage one boundle group!
// NOTE:
//  in order to be consistent with the bundle constraint, this function
//  shall be called before the popolating fragment from the db
//  we need the special fragments to be at the top domain positions.
//  Also these special elements shall not be sorted (or use a stable sort)
//  to preserve the order
void
Input_data::load_constraint_file (vector<Fragment>& fragment_set) {
  ifstream in_confile, in_pdbfile;
  
  in_confile.open(constraint_file.c_str());
  in_pdbfile.open(frag_spec_file.c_str());
  
  if (!in_confile.is_open()){
    cout << "Error opening file: " << constraint_file << endl;
    return;
  }
  if (!in_pdbfile.is_open()){
    cout << "Error opening file: " << frag_spec_file << endl;
    return;
  }

  string line;
  string pdb_buff, mdl = "MODEL", endmdl = "ENDMDL", atom = "ATOM";
  string con_bundle = "bundle", con_dist_geq = "distGEQ", con_box = "box", 
    con_ellis = "ellisoid";
  string prot_id = "Protein ID: ";
  string frag_no = "Fragment n. : ";
  string off_tar = "Offset on target:";
  string b_constr_str = "begin constraints";
  string e_constr_str = "end constraints";
  
  vector <special_fragment> frag_list;

  // read target protein sequence 
  getline(in_confile, line);
  g_target.sequence = line;
  g_target.set_nres(line.length());
  
  g_target.sequence.clear();
  g_target.set_nres(0);
    
  while (in_confile.good()) {
    getline(in_confile, line);

    if(line.length() == 0)
      continue;

    if (line.compare(0, prot_id.length(), prot_id)==0) {
      special_fragment f;
      // Protein ID.
      f.pid =
	line.substr (prot_id.length(), line.size() - prot_id.length());

      // Fragment NO.
      getline (in_confile, line);
      string nofs = line.substr(14, line.length());
      f.frag_no = atoi (nofs.c_str());

      // Fragment offset in the host protein 
      getline(in_confile, line); 
      f.offset_pdb = atoi(line.c_str());
      
      getline(in_confile, line); // offset on target: (skip)

      // Fragment offset on target
      getline(in_confile, line);
      uint offset_tar_s = atoi(line.c_str())-1;
      f.aa_start = offset_tar_s;
      getline(in_confile, line);
      uint offset_tar_e = atoi(line.c_str())-1;
      f.aa_end = offset_tar_e;

      uint bbs = offset_tar_s > 0 ? 
	get_bbidx_from_aaidx(offset_tar_s-1, CB) 
	: 0;
      uint bbe = offset_tar_e < g_target.get_nres()-1 ? 
	get_bbidx_from_aaidx(offset_tar_e+1, N) 
	: get_bbidx_from_aaidx(offset_tar_e, O);

      f.bb_start = bbs;
      f.bb_end = bbe;
      
      getline(in_confile, line);

      //Constraints: 
      if (line.compare(b_constr_str) == 0) {
	while (line.compare(e_constr_str) != 0) {
	  getline (in_confile, line);
	  // Boundle
	  if (line.compare (0, con_bundle.length(), con_bundle) == 0) {
	    int nfrag, fid;
	    in_confile >> nfrag;
	    for (int i=0; i<nfrag; i++) {
	      in_confile >> fid;
	      f.bundle.push_back(fid);
	    }
	  }
	}
      }
      frag_list.push_back (f);
    }
  }// end constraint file
  
  // Import special fragment from pdb for every fragment imported
  line = "";
  while (in_pdbfile.good()) {
    //    getline(in_pdbfile, line);
    in_pdbfile >> pdb_buff;

    // search start model
    if (pdb_buff == mdl) {
      
      //    if (line.compare(0, mdl.size(), mdl) == 0) {
      vector<aminoacid> aa_seq;
      vector<Atom> backbone;
      special_fragment* f_spec = NULL;
      real x,y,z;
      string type_atom, type_aa;

      in_pdbfile >> pdb_buff;
      int mod_no = atoi(pdb_buff.c_str());
      
      // Retreieve correct special fragment
      for (uint m=0; m<frag_list.size(); m++) {
	if (frag_list[m].frag_no == mod_no) {
	  f_spec = &frag_list[m];
	  break;
	}
      }
      assert (f_spec != NULL);

      // Read atoms details
      while (pdb_buff.compare(0, endmdl.size(), endmdl) != 0) {
	in_pdbfile >> pdb_buff;
	if(pdb_buff.compare(atom) == 0) {
	  in_pdbfile >> pdb_buff;	   // N of atom
	  in_pdbfile >> type_atom;         // Type of atom
	  in_pdbfile >> type_aa;	   // Type of AA
	  
	  in_pdbfile >> pdb_buff;	   // chain 
	  in_pdbfile >> pdb_buff;	   // AA number
	  in_pdbfile >> x >> y >> z; // coordinates
	  
	  Atom a;
	  a.set_position(x,y,z);
	  if (type_atom.compare("CA") == 0) {
	    aa_seq.push_back(cv_aa_to_class(type_aa));
	    a.set_type(CA);
	    backbone.push_back(a);
	  }
	  if (type_atom.compare("N") == 0){
	    a.set_type(N);
	    backbone.push_back(a);
	  }
	  if (type_atom.compare("C") == 0){
	    a.set_type(CB);
	    backbone.push_back(a);
	  }
	  if (type_atom.compare("O") == 0){
	    a.set_type(O);
	    backbone.push_back(a);
	  }
	}
      }// model bounds

      // Set end offset on target
      f_spec->bb_end = f_spec->bb_start + backbone.size()-1;
      // Create the fragment
      Fragment frag 
	(f_spec->frag_no, special, f_spec->pid, f_spec->offset_pdb, 
	 f_spec->aa_start, f_spec->aa_end, f_spec->bb_start, f_spec->bb_end, 
	 1, 1.0, aa_seq, backbone); 

      // check for boundle constraint
      if (!f_spec->bundle.empty()) {
	// check if this is the first fragment of the bundle
	if (f_spec->frag_no == f_spec->bundle[0]) {
	  frag.compute_normal_base(0);
	  frag.change_coordinate_system();
	  
	  // link other fragments of the bundle with this one
	  for (uint i=1; i<f_spec->bundle.size(); i++) {
	    for (uint m=0; m<frag_list.size(); m++) {
	      if (frag_list[m].frag_no == f_spec->bundle[i]) {
		frag_list[m].link = fragment_set.size();
		break;
	      }
	    }
	  }
	}
	else {
	  // transform according to the first fragment of the bundle 
	  frag.copy_rot_mat (fragment_set[f_spec->link]);
	  frag.copy_sh_vec (fragment_set[f_spec->link]);
	  frag.change_coordinate_system();
	}
      }
      else {
	// single fragments needs to be normalized as default
	frag.compute_normal_base(0);
	frag.change_coordinate_system();
      }
      fragment_set.push_back (frag);
      backbone.clear();
    } // model processing ends here
  } // pdb file processing ends here
  
  // Close the files
  in_confile.close();
  in_pdbfile.close();

}//-

void
Input_data::dump_log() {
  cout << "#log: Data imported ---------------" << endl;
  cout << "#log: Fragment Assembly DB  : " << fragmentdb << endl;
  cout << "#log: Energy DB             : " << energydb << endl;
  cout << "#log: Aminoacid class defs  : " << classdb << endl;
  cout << "#log: Protein Known  ref    : " << known_prot_file << endl;
  cout << "#log: Protein Target ref    : " << target_prot_file << endl;
  cout << "#log: Constraint file       : " << constraint_file << endl;
  cout << "#log: Special fragments file: " << frag_spec_file << endl;
  cout << "      -----------------------------" << endl;

}
