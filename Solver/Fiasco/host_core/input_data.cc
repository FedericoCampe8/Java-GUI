#include "input_data.h"
#include "utilities.h"
#include "globals.h"
#include "atom.h"
#include "mathematics.h"

#include "variable_point.h"
#include "logic_variables.h"
#include "output.h"

#include "tors_corr_bmf.h"
#include "tors_bmf.h"

#include <iostream>
#include <cassert>
#include <vector>
#include <fstream>
#include <string>
#include <sstream>
#include <stdlib.h> // for atoi
#include <algorithm>    // std::swap

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

Input_data::~Input_data () {
  clear();
}//~Input_data

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
      if ( line.compare(0, 3, "SEQ") == 0 ) {
        size_t position = line.find_first_of ( " " );
        seq_type = line.substr ( position + 1, line.size() );
      }
    }
    inputFile.close();
  }else {
    cout << "Unable to open file" << filename << endl;
  }
  delete[] fname;
}//Input_data


typedef struct special_fragment_{
  string pid;    // original protein
  int idx_bb;    // idx in the DB based on leftomst assignment
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
    
  while (in_confile.good()) 
  {
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
      frag_list.push_back (f);
    }
  }// end constraint file

  // Inefficient Sort Special Fragments based on their Offsets on target
  for(int i=0; i<frag_list.size(); i++) {
    int si = frag_list[ i ].bb_start;
    for(int j=i+1; j<frag_list.size(); j++) {
      int sj = frag_list[ j ].bb_start;
      if( sj < si ) std::swap(frag_list[ i ], frag_list[ j ]);
    }
    // Re-order fragment numbers in order of their bb_start number
    frag_list[ i ].idx_bb = i;
    frag_list[ i ].link = -1;
  }

  fragment_set.resize( frag_list.size() );

  vector<int> boundle_fragments;
  int idx_of_first_frag_in_boundle = -1;

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
	if (frag_list[ m ].frag_no == mod_no) {
	  f_spec = &frag_list[ m ];
	  break;
	}
      }
      assert (f_spec != NULL);

      // Read atoms details
      while (pdb_buff.compare(0, endmdl.size(), endmdl) != 0) 
      {
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
      

      // Set end offset on target
      f_spec->bb_end = f_spec->bb_start + backbone.size()-1;
      
      // check for boundle constraint
      if( g_params.fix_fragments )
      {
	// check if this is the first fragment of the bundle
	if (f_spec->frag_no == frag_list[ 0 ].frag_no ) {

	  if( frag.backbone[ 0 ].is_type( N ) ) {
	    Math::set_identity(frag.rot_m);
	    Math::set_identity(frag.shift_v);
	  } else {
	    frag.compute_normal_base(0);
	    frag.change_coordinate_system();
	  }

	  // link other fragments of the bundle with this one (will be inserted
	  // later at the end of the fragment_set 
	  int curr_frag_idx = f_spec->idx_bb;//fragment_set.size();
	  idx_of_first_frag_in_boundle = curr_frag_idx;
	}
	else {
	  int curr_frag_idx = f_spec->idx_bb; //fragment_set.size();	  
	  boundle_fragments.push_back( curr_frag_idx );
	}
      }
      else {
	// single fragments needs to be normalized as default
	if( frag.backbone[ 0 ].is_type( N ) ) {
	  Math::set_identity(frag.rot_m);
	  Math::set_identity(frag.shift_v);
	} else {
	  frag.compute_normal_base(0);
	  frag.change_coordinate_system();
	}
      }
      fragment_set[ f_spec->idx_bb ] = frag;
            
      // fragment_set.push_back (frag);
      backbone.clear();
    } // model processing ends here
  } // pdb file processing ends here
  

  if( idx_of_first_frag_in_boundle >= 0 ) {
    int n_res_bundle = 0;
    for( int i : boundle_fragments )
    {
      // transform according to the first fragment of the bundle
      int j = idx_of_first_frag_in_boundle;

      fragment_set[ i ].copy_rot_mat( fragment_set[ j ] );
      fragment_set[ i ].copy_sh_vec( fragment_set[ j ] );
      fragment_set[ i ].change_coordinate_system();

      n_res_bundle += fragment_set[ i ].nres();
    }
    int j = idx_of_first_frag_in_boundle;
    fragment_set[ j ].n_res_of_fragment_associated_in_bundle_constraint = n_res_bundle;
  }
  
  // Close the files
  in_confile.close();
  in_pdbfile.close();
}//-


void
Input_data::alloc_energy () {
  g_params.h_distances       = new real [25*3];
  g_params.h_angles          = new real [73*9];
  g_params.contact_params    = new real [20*20];
  g_params.tors              = new real [20*20*18*3];
  g_params.tors_corr         = new real [18*18*5];
  g_params.aa_seq            = new aminoacid [ g_target.get_nres()];
  g_params.secondary_s_info  = new ss_type [ g_target.get_nres()];
}//alloc_energy

void
Input_data::init_energy () {
  
  assert ( g_target.get_nres() > 0 );
  assert ( g_target.seq_code.size() > 0 );
  
  int idx = 0;
  for ( auto c : seq_type ) {
    if ( c == 'H' ) {
      g_params.secondary_s_info [ idx ] = helix;
    }
    else if ( c == 'S' ) {
      g_params.secondary_s_info [ idx ] = sheet;
    }
    else {
      g_params.secondary_s_info [ idx ] = other;
    }
    idx++;
  }
  if ( idx < g_target.get_nres() )
    for ( ; idx < g_target.get_nres(); idx++ )
      g_params.secondary_s_info [ idx ] = other;
  
  for ( int i = 0; i < g_target.get_nres(); i++ ) {
    g_params.aa_seq[ i ] = g_target.seq_code[ i ];
  }

  // Fill the matrices with csv values read from file
  std::vector< std::vector<real> > energy_params;
  // H_Distances
  read_energy_parameters ( "config/h_distances.csv", energy_params );
  for (int i = 0; i < 25; i++)
    for (int j = 0; j < 3; j++)
      g_params.h_distances[ 3*i + j ] = energy_params[ i ][ j ];
  energy_params.clear();
  // H_Angles
  read_energy_parameters ( "config/h_angles.csv", energy_params );
  for (int i = 0; i < 73; i++)
    for (int j = 0; j < 9; j++)
      g_params.h_angles[ 9*i + j ] =  energy_params[ i ][ j ];
  energy_params.clear();
  cout << "Hydrogen parameters loaded" << endl;
  // Contact Parameters
  read_energy_parameters ( "config/contact.csv", energy_params );
  for (int i = 0; i < 20; i++)
    for (int j = 0; j < 20; j++)
      g_params.contact_params[ 20*i+j ] = energy_params[ i ][ j ];
  energy_params.clear();
  cout << "Contact parameters loaded" << endl;
  
  
  // From ".h" file
  for (int i = 0; i < 20; i++)
    for (int j = 0; j < 20; j++)
      for (int z = 0; z < 18; z++)
        for (int t = 0; t < 3; t++)
          g_params.tors [ i*20*18*3 + j*18*3 + z*3 + t ] = tors[ i ][ j ][ z ][ t ];
  for (int i = 0; i < 18; i++)
    for (int j = 0; j < 18; j++)
      for (int z = 0; z < 5; z++)
        g_params.tors_corr[ i*18*5 + j*5 + z ] = tors_corr[ i ][ j ][ z ];
  cout << "Torsional parameters loaded" << endl;
}//init_energy


void
Input_data::read_energy_parameters ( string file_name, vector< vector<real> >& param_v ) {
  string line;
  string token;
  real value;
  size_t found;
  size_t t = 0;
  
  ifstream csv_file ( file_name.c_str() );
  
  if ( !csv_file.is_open() ) {
    cout <<  "error opening energy parameters file " << file_name << endl;
    exit( 1 );
    return;
  }
  
  
  while ( csv_file.good() ) {
    vector< real > line_v;
    getline ( csv_file, line );
    if (line.compare(0,1, "%") == 0) continue;
    found = line.find_first_of( "," );
    while ( found != string::npos ) {
      token = line.substr( t, found - t );
      value = atof( token.c_str() );
      line_v.push_back( value );
      t = t + token.size() + 1;
      found = line.find_first_of( ",", found+1 );
    }
    token = line.substr( t, found - t );
    value = atof( token.c_str() );
    line_v.push_back( value );
    
    param_v.push_back( line_v );
    t = 0;
  }//while
  
  csv_file.close();
}//read_energy_parameters


void
Input_data::clear() {
  delete [] g_params.h_distances;
  delete [] g_params.h_angles;
  delete [] g_params.contact_params;
  delete [] g_params.tors;
  delete [] g_params.tors_corr;
  delete [] g_params.aa_seq;
  delete [] g_params.secondary_s_info;
}

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
