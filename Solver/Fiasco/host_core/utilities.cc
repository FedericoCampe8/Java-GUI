#include "utilities.h"
#include "globals.h"
#include "mathematics.h"
#include "protein.h"
#include "atom.h"
#include "logic_variables.h"
#include "variable_point.h"
#include "variable_fragment.h"
#include "fragment.h"
#include "hilbert.h"

#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>

using namespace std;
using namespace Utilities;
using namespace Math;

string 
Utilities::cv_aa1_to_aa3(char a){
  if (a=='a' || a=='A') return "ALA";
  if (a=='r' || a=='R') return "ARG";
  if (a=='n' || a=='N') return "ASN";
  if (a=='d' || a=='D') return "ASP";
  if (a=='c' || a=='C') return "CYS"; 
  if (a=='q' || a=='Q') return "GLN";
  if (a=='e' || a=='E') return "GLU";
  if (a=='g' || a=='G') return "GLY";
  if (a=='h' || a=='H') return "HIS";
  if (a=='i' || a=='I') return "ILE";
  if (a=='l' || a=='L') return "LEU";
  if (a=='k' || a=='K') return "LYS";
  if (a=='m' || a=='M') return "MET";
  if (a=='f' || a=='F') return "PHE";
  if (a=='p' || a=='P') return "PRO";
  if (a=='s' || a=='S') return "SER";
  if (a=='t' || a=='T') return "THR";
  if (a=='w' || a=='W') return "TRP";
  if (a=='y' || a=='Y') return "TYR";
  if (a=='v' || a=='V') return "VAL";
  return "!";
}//-

string
Utilities::cv_aa3_to_aa1(string a){
  if (!a.compare("ALA")) return "a";
  if (!a.compare("ARG")) return "r";
  if (!a.compare("ASN")) return "n";
  if (!a.compare("ASP")) return "d";
  if (!a.compare("CYS")) return "c";
  if (!a.compare("GLN")) return "q";
  if (!a.compare("GLU")) return "e";
  if (!a.compare("GLY")) return "g";
  if (!a.compare("HIS")) return "h";
  if (!a.compare("ILE")) return "i";
  if (!a.compare("LEU")) return "l";
  if (!a.compare("LYS")) return "k";
  if (!a.compare("MET")) return "m";
  if (!a.compare("PHE")) return "f";
  if (!a.compare("PRO")) return "p";
  if (!a.compare("SER")) return "s";
  if (!a.compare("THR")) return "t";
  if (!a.compare("TRP")) return "w";
  if (!a.compare("TYR")) return "y";
  if (!a.compare("VAL")) return "v";
  return "err";
}//-

string
Utilities::cv_class_to_aa3(aminoacid a){
  if (a==ala) return "ala";
  if (a==arg) return "arg";
  if (a==asn) return "asn";
  if (a==asp) return "asp";
  if (a==cys) return "cys";
  if (a==gln) return "gln";
  if (a==glu) return "glu";
  if (a==gly) return "gly";
  if (a==his) return "his";
  if (a==ile) return "ile";
  if (a==leu) return "leu";
  if (a==lys) return "lys";
  if (a==met) return "met";
  if (a==phe) return "phe";
  if (a==pro) return "pro";
  if (a==ser) return "ser";
  if (a==thr) return "thr";
  if (a==trp) return "trp";
  if (a==tyr) return "tyr";
  if (a==val) return "val";
  return "";
}//-

char
Utilities::cv_class_to_aa1(aminoacid a){
  if (a==ala) return 'A';
  if (a==arg) return 'R';
  if (a==asn) return 'N';
  if (a==asp) return 'D';
  if (a==cys) return 'C';
  if (a==gln) return 'Q';
  if (a==glu) return 'E';
  if (a==gly) return 'G';
  if (a==his) return 'H';
  if (a==ile) return 'I';
  if (a==leu) return 'L';
  if (a==lys) return 'K';
  if (a==met) return 'M';
  if (a==phe) return 'F';
  if (a==pro) return 'P';
  if (a==ser) return 'S';
  if (a==thr) return 'T';
  if (a==trp) return 'W';
  if (a==tyr) return 'Y';
  if (a==val) return 'V';
  return 0;
}//-

aminoacid
Utilities::cv_aa_to_class(char a){
  if(a=='a' || a=='A') return ala;
  if(a=='r' || a=='R') return arg;
  if(a=='n' || a=='N') return asn;
  if(a=='d' || a=='D') return asp;
  if(a=='c' || a=='C') return cys;
  if(a=='q' || a=='Q') return gln;
  if(a=='e' || a=='E') return glu;
  if(a=='g' || a=='G') return gly;
  if(a=='h' || a=='H') return his;
  if(a=='i' || a=='I') return ile;
  if(a=='l' || a=='L') return leu;
  if(a=='k' || a=='K') return lys;
  if(a=='m' || a=='M') return met;
  if(a=='f' || a=='F') return phe;
  if(a=='p' || a=='P') return pro;
  if(a=='s' || a=='S') return ser;
  if(a=='t' || a=='T') return thr;
  if(a=='w' || a=='W') return trp;
  if(a=='y' || a=='Y') return tyr;
  if(a=='v' || a=='V') return val;
  return err;
}//-

aminoacid
Utilities::cv_aa_to_class(string a){
  if (!a.compare("ALA")) return ala;
  if (!a.compare("ARG")) return arg;
  if (!a.compare("ASN")) return asn;
  if (!a.compare("ASP")) return asp;
  if (!a.compare("CYS")) return cys;
  if (!a.compare("GLN")) return gln;
  if (!a.compare("GLU")) return glu;
  if (!a.compare("GLY")) return gly;
  if (!a.compare("HIS")) return his;
  if (!a.compare("ILE")) return ile;
  if (!a.compare("LEU")) return leu;
  if (!a.compare("LYS")) return lys;
  if (!a.compare("MET")) return met;
  if (!a.compare("PHE")) return phe;
  if (!a.compare("PRO")) return pro;
  if (!a.compare("SER")) return ser;
  if (!a.compare("THR")) return thr;
  if (!a.compare("TRP")) return trp;
  if (!a.compare("TYR")) return tyr;
  if (!a.compare("VAL")) return val;
  return err;
}//-

size_t 
Utilities::count_active 
(std::vector<std::vector <bool> > vec, uint lev_s, uint lev_e) {
  size_t count = 0;
  for (uint i = lev_s; i <= lev_e; i ++) {
    for (uint ii = 0; ii < vec[i].size(); ii ++)
      count += (vec[i][ii]) ? 1 : 0;
  }
  return count; 
}//-

void
Utilities::populate_fragment_assembly_db (vector<Fragment>& fragment_set, int fragment_len, string filename) {
  string dbg = "Utilities::populate_fragment_assembly_db () - ";
  ifstream in_assemblydb;
  string delim;
  string line;
  string res_name;
  char res_id;
  real prob;
  real x,y,z;
  int fid;
  string prot_name;
  fragment_type type;
  in_assemblydb.open(filename.c_str());

  if (!in_assemblydb.is_open()){
    cout << "Error: Could not import the Fragment Assembly DB file: " << filename << endl;
    exit(0);
  }
    
  in_assemblydb.seekg(ios_base::beg);
  // populate classes -- skip AA definitions!
  // DEPRECATED
  // while (in_assemblydb.good() && line_count < 20) {
  //   in_assemblydb >> res_name >> res_id;
  //   // save class --> not needed
  //   line_count++;
  // }
  // populate fragment DB
  while(in_assemblydb.good()) {
    res_id = 127;
    //bool read = false;
    // vector<aminoacid> aa_seq;
    vector<aminoacid> aa_seq;
    vector<Atom> backbone;
    
    // define fragment type
    for (int i = 0; i<fragment_len; i++) {
        in_assemblydb >> res_id;	
        aa_seq.push_back(Utilities::cv_aa_to_class(res_id) );
    }
      
    // eof reached
    if (res_id == 127){
        break;
    }  
    
    if (res_id == special) type = special;
    else if (res_id == helix) type = helix;
    else if (res_id == sheet) type = sheet;
    else type = standard;

    // Read coordinates 4*n + 3   (C' O N Ca)* C' O N
    for (int i=0; i<fragment_len; i++) {
      in_assemblydb >> x >> y >> z; // C' 
      Atom a1(x, y, z, CB);
      backbone.push_back(a1);
      in_assemblydb >> x >> y >> z; // O
      Atom a2(x, y, z, O);
      backbone.push_back(a2);
      in_assemblydb >> x >> y >> z; // N
      Atom a3(x, y, z, N);
      backbone.push_back(a3);
      in_assemblydb >> x >> y >> z; // Ca
      Atom a4(x, y, z, CA);
      backbone.push_back(a4);
    }
    // last three atoms
    in_assemblydb >> x >> y >> z; // C'
    Atom a5(x, y, z, CB);
    backbone.push_back(a5);
    in_assemblydb >> x >> y >> z; // O
    Atom a6(x, y, z, O);
    backbone.push_back(a6);
    in_assemblydb >> x >> y >> z; // N
    Atom a7(x, y, z, N);
    backbone.push_back(a7);
    // Probability;
    in_assemblydb >> prob;
    // Fragment ID
    in_assemblydb >> fid;
    // DEPRECATED
    // // Protein Name
    // in_assemblydb >> prot_name;
    // // Frequency
    // in_assemblydb >> freq;
    // // srcAaPos;
    // in_assemblydb >> srcAaPos;
    
    /* Create the fragment */
    Fragment f (fid, type, "", 0, 0, fragment_len-1, 0, 7*fragment_len-1,
		-1, prob, aa_seq, backbone);
    f.compute_normal_base ();
    f.change_coordinate_system ();
    
    if (f.check_steric_clashes())
      fragment_set.push_back (f);
  }
  sort(fragment_set.begin(), fragment_set.end(), Fragment());
  /* Close the Fragment DB */  
  in_assemblydb.close();
}//-


void
Utilities::dump (const point& a, const string prefix) {
  cout << prefix << "<" << a[0] << ", " << a[1] 
       << ", " << a[2] << ">" << endl;
}//-

void
Utilities::dump(const point& a, const point& b, const string prefix) {
  if (a[0]==b[0] && a[1]==b[1] && a[2]==b[2])
    dump(a, prefix);
  else {
    cout<<prefix<<"[<"<<a[0]<<", "<<a[1]<<", "<<a[2]<<">, ";
    cout<<"<"<<b[0]<<", "<<b[1]<<", "<<b[2]<<">]";
  }
}//-

void
Utilities::dump(const R_MAT r) {
  cout << "|" << r[0][0] << " " << r[0][1] << " " << r[0][2] << "|" << endl;
  cout << "|" << r[1][0] << " " << r[1][1] << " " << r[1][2] << "|" << endl;
  cout << "|" << r[2][0] << " " << r[2][1] << " " << r[2][2] << "|" << endl;
}//-

void
Utilities::dump(const std::vector<bool> v) {
  cout <<   "      + 1  2  3  4  5  6  7  8  9 " << endl;
  cout <<   "        --------------------------" << endl;
  for (uint i=0; i<v.size(); i++) {
    if (i<10) cout << "D(" << i << ")  : ";
    if (i>=10 && i<100) cout << "D(" << i << ") : ";
    if (i>=100 && i<1000) cout << "D(" << i << "): ";
    for (int j=0; j<10; j++) {
      if (v.at(i)) cout << "T  ";
      else         cout << "F  ";
      if (i < v.size()-1) ++i;
      else break;
        if(j+1 == 10) i = i-1;  
    } 
    cout << endl;
  }
}//-

void
Utilities::dump(const std::vector<int> n) {
  cout <<   "      + 1  2  3  4  5  6  7  8  9 " << endl;
  cout <<   "        --------------------------" << endl;
  for (uint i=0; i<n.size(); i++) {
    if (i<10) cout << " " << i << "    : ";
    if (i>=10 && i<100) cout << "D(" << i << ") : ";
    if (i>=100 && i<1000) cout << "D(" << i << "): ";
    for (int j=0; j<10; j++) {
      cout << n.at(i);
      i++;
    }
    cout << endl;
  }
}//-

void 
Utilities::dump(const std::vector<int> v, int s, int e) {
  for (int i=s; i<e; i++) {
    cout << v[i];
    if (v[i]>=0 && v[i]<10) cout <<"   ";
    else if (v[i]>=10 && v[i]<100) cout <<"  ";
    else if (v[i]>=100 && v[i]<1000) cout <<" ";
  }
  cout << endl;
}//-

void 
Utilities::dump(const std::vector<bool> v, int s, int e) {
  for (int i=s; i<e; i++)
    cout << v[i] <<" ";
  cout << endl;
}//-

/*
void 
Utilities::dump ( const std::vector<aminoacid> v, int s, int e ) {
  for (int i=s; i<e; i++) {
    cout << v[i];
    if ( v[i] >= 0 && v[i] < 10 ) cout <<"   ";
    else if ( v[i] >= 10 ) cout <<"  ";
  }
  cout << endl;
}//-
*/

void 
Utilities::init_file(string outf) {
  FILE *fid = fopen (outf.c_str(), "a");
  fprintf (fid, "MDL NO.    ");
  fprintf (fid,"\t TotEnergy");
  fprintf (fid,"\t contr_ori");
  fprintf (fid,"\t contr_con");
  fprintf (fid,"\t contr_tor");
  fprintf (fid,"\t RMSD:    \n");
  fclose (fid);
}//-


void 
Utilities::output_pdb_format (string outf, const size_t id, 
			      const Protein& P, real rmsd=0){
  FILE *fid;
  char fx[4],fy[4],fz[4];
  int i = 0;
  int k = 0;
  real x,y,z;
  int naa = P.get_nres();
  int bblen = P.get_bblen();

  /* Open an output file */
  fid = fopen (outf.c_str(), "a");
  if (fid < 0){
    printf("Cannot open %s to write!\n", outf.c_str());
    return;
  }
    
  int atom=1;
  /* Write the solution to the output file */
  fprintf(fid, "MODEL    %zu\n", id);
  /* Write info about the energetic components */
  if (rmsd > 0)
    fprintf(fid, "REMARK    Rmsd: %.6f\n", rmsd);
  
  fprintf(fid, "REMARK    Energy: %.6f\n", g_logicvars->energy);

  k=-1;
  for (i = 0 ; i < bblen; i++) {
    if (i%4 == NITROGEN_ATOM) k++;
    strcpy (fx, " ");
    strcpy (fy, " ");
    strcpy (fz, " ");      
    /* Get Calpha locations */
    x = g_logicvars->var_point_list.at(i)[0];
    y = g_logicvars->var_point_list.at(i)[1];
    z = g_logicvars->var_point_list.at(i)[2];
    /* Set correct spacing */
    /* Specify the output format */
    if (x < 0 && x > -10)  strcpy (fx, "  ");
    if (y < 0 && y > -10)  strcpy (fy, "  ");
    if (z < 0 && z > -10)  strcpy (fz, "  ");
    if (x >= 0 && x < 10)  strcpy (fx, "   ");
    if (y >= 0 && y < 10)  strcpy (fy, "   ");
    if (z >= 0 && z < 10)  strcpy (fz, "   ");
    if (x > 10 && x < 100) strcpy (fx, "  ");
    if (y > 10 && y < 100) strcpy (fy, "  ");
    if (z > 10 && z < 100) strcpy (fz, "  ");
 
    fprintf(fid,"ATOM    %3d  ",atom++);
    if (i%4 == NITROGEN_ATOM)
      fprintf (fid, "N   ");
    if (i%4 == CALPHA_ATOM)
      fprintf (fid, "CA  ");
    if (i%4 == CBETA_ATOM)
      fprintf (fid, "C   ");
    if (i%4 == OXYGEN_ATOM)
      fprintf (fid, "O   ");
    
    fprintf (fid, "%s A %3d    %s%3.3f%s%3.3f%s%3.3f  1.00  1.00\n",
	     cv_aa1_to_aa3(P.sequence[k]).c_str(), k+1, fx, x, fy, y, fz, z);
  }
  
  if(g_logicvars->var_cg_list.size() > 0){  
    for (i = 1 ; i < naa-1; i++){
      strcpy (fx, " ");
      strcpy (fy, " ");
      strcpy (fz, " ");      
      /* Get Centroids locations */ 
      x = g_logicvars->var_cg_list.at(i-1)[0];
      y = g_logicvars->var_cg_list.at(i-1)[1];
      z = g_logicvars->var_cg_list.at(i-1)[2];
    /* Set correct spacing */
    /* Specify the output format */
      if (x < 0 && x > -10)  strcpy (fx, "  ");
      if (y < 0 && y > -10)  strcpy (fy, "  ");
      if (z < 0 && z > -10)  strcpy (fz, "  ");
      if (x >= 0 && x < 10)  strcpy (fx, "   ");
      if (y >= 0 && y < 10)  strcpy (fy, "   ");
      if (z >= 0 && z < 10)  strcpy (fz, "   ");
      if (x > 10 && x < 100) strcpy (fx, "  ");
      if (y > 10 && y < 100) strcpy (fy, "  ");
      if (z > 10 && z < 100) strcpy (fz, "  ");
      fprintf(fid,"ATOM    %3d  CG  %s A %3d    %s%3.3f%s%3.3f%s%3.3f  1.00  1.00\n",
	      atom++, cv_aa1_to_aa3(P.sequence[i]).c_str(), i, fx, x, fy, y, fz, z);

    }
  }

  fprintf(fid,"ENDMDL\n");
    
  fclose(fid);   
}//-

void 
Utilities::output_pdb_format (string outf, const vector<Atom>& vec) {
  FILE *fid;
  char fx[4],fy[4],fz[4];
  int k = 0;
  real x,y,z;
  /* Open an output file */
  fid = fopen (outf.c_str(), "a");
  if (fid < 0){
    printf("Cannot open %s to write!\n", outf.c_str());
    return;
  }    
  int atom=1;
  fprintf(fid, "MODEL    001\n");
  /* Write the solution to the output file */
  k=0;
  for (uint i = 0 ; i < vec.size(); i++) {
    if (vec[i].is_type(N)) k++;
    strcpy (fx, " ");
    strcpy (fy, " ");
    strcpy (fz, " ");      
    /* Get Calpha locations */
    x = vec.at(i)[0];
    y = vec.at(i)[1];
    z = vec.at(i)[2];
    /* Set correct spacing */
    /* Specify the output format */
    if (x < 0 && x > -10)  strcpy (fx, "  ");
    if (y < 0 && y > -10)  strcpy (fy, "  ");
    if (z < 0 && z > -10)  strcpy (fz, "  ");
    if (x >= 0 && x < 10)  strcpy (fx, "   ");
    if (y >= 0 && y < 10)  strcpy (fy, "   ");
    if (z >= 0 && z < 10)  strcpy (fz, "   ");
    if (x > 10 && x < 100) strcpy (fx, "  ");
    if (y > 10 && y < 100) strcpy (fy, "  ");
    if (z > 10 && z < 100) strcpy (fz, "  ");
 
//    if(atom == 1 && (vec.at(i).type == CB || vec.at(i).type == O))
//        continue;
      
    fprintf(fid,"ATOM    %3d  ",atom++);
    if (vec.at(i).type == N)
      fprintf (fid, "N   ");
    if (vec.at(i).type == CA)
      fprintf (fid, "CA  ");
    if (vec.at(i).type == CB)
      fprintf (fid, "C   ");
    if (vec.at(i).type == O)
      fprintf (fid, "O   ");
    
    fprintf (fid, "XXX A %3d    %s%3.3f%s%3.3f%s%3.3f  1.00  1.00\n",
	     k+1, fx, x, fy, y, fz, z);
  }
  fprintf(fid,"ENDMDL\n");
    
  fclose(fid);   
}//-


// Return the atom backbone index given the type and the amino acid
int 
Utilities::get_bbidx_from_aaidx (uint aaidx, atom_type type){
  switch (type) {
  case N:
    return aaidx*4;
  case CA:
    return aaidx*4+1;
  case CB:
    return aaidx*4+2;
  case O:
    return aaidx*4+3;
  case CG:
    return aaidx; // assuming aaidx>1 and <n
  default:
    return -1;
  } 
  return -1;
}//-

// Return the aminoacid index given the type and the bb atom
int
Utilities::get_aaidx_from_bbidx (uint bbidx, atom_type type){
  switch (type) {
  case N:
    return bbidx/4;
  case CA:
    return (bbidx-1)/4;
  case CB:
    return (bbidx-2)/4;
  case O:
    return (bbidx-3)/4;
  case CG:   
    return bbidx - g_target.get_bblen();
  default:
    return -1;
  } 
  return -1;
}//-

atom_type
Utilities::get_atom_type(uint bbidx) {
    if (bbidx >= g_target.get_bblen()) {
        return CG;
    }
  switch (bbidx%4) {
  case NITROGEN_ATOM:
    return N;
  case CALPHA_ATOM:
    return CA;
  case CBETA_ATOM:
    return CB;
  case OXYGEN_ATOM:
    return O;
  case CENTROID_ATOM:
    return CG;
  }
  return X;
}//-


atom_type
Utilities::get_atom_type(std::string str) {
  if (str == "N")
    return N;
  if (str == "CA")
    return CA;
  if (str == "CB")
    return CB;
  if (str == "O")
    return O;;
  if (str == "CG")
    return CG;
  
  return X;
}//-


atom_radii 
Utilities::get_atom_radii (uint bbidx) {
 if (bbidx >= g_target.get_bblen()) {
        return rCG;
  }
  switch (bbidx%4) {
  case NITROGEN_ATOM:
    return rN;
  case CALPHA_ATOM:
    return rC;
  case CBETA_ATOM:
    return rC;
  case OXYGEN_ATOM:
    return rO;
  case CENTROID_ATOM:
    return rCG;
  }
  return rC;
}

atom_radii
Utilities::get_atom_radii (atom_type t) {
  switch (t) {
  case N : return rN;
  case CA : return rC;
  case CB : return rC;
  case O : return rO;
  case H : return rH;
  case S : return rS;
  case CG : return rCG;
  default: return rC;
  }
  return rC;
}//-


void 
Utilities::usage() {
  std::cout << "usage: ./fiasco --input <inputfile> --outfile <outfile> "
	    << "OPTIONAL: \n"
	    << "--domain-size <limit>\n"
	    << "--ensembles <max_numof_ensembles>\n"
	    << "--timeout <t_search_limit> <t_total_limit> \n"
	    << "--uniform aa_1 .. aa_n : voxel-side= K [center= X Y Z ] \n"
	    << "--ellipsoid  aa_1 .. aa_n : f1= X Y Z f2= X Y Z sum-rad= K \n"
	    << "--jmf <min num cluster> <max num clusters> <betaR cluster angle> <distance>  <iterations to ignore>\n"
	    << "--end-anchor [<radius> <steps>] <err_prismatic> [<numof bins>] <err_revloute> " 
	    << std::endl << std::endl 
	    << "   where:\n"
	    << "   --domain-size \n"
	    << "     <limit> is the maximum number of fragments to be used for each variable \n"
	    << "   --ensembles \n"
	    << "     <max_numof_enembles> is the maximum number of conformations to be generated \n"
	    << "   --timeout \n"
	    << "     <t_search_limit> limit on the search procedure time \n"
	    << "     <t_total_limit>  limit on the total execution time \n"
     	    << "   --uniform: \n"
	    << "     aa_1 .. aa_n, is a list of amino acids (starting from 0) \n"
	    << "         for which CAs will be involved in the uniform constraint\n"
	    << "         (every a_i will be placed using one grid). \n"
	    << "     K, is the side of a voxel in Angstroms. \n"
	    << "     X Y Z [optional] are the coordinates for the grid (lattice cube) center of mass. \n"
	    << "         If not specified the grid will be centeed in the origin. \n"
	    << std::endl
	    << "  --ellipsoid:  \n"
	    << "    aa_1 .. aa_n, is a list of amino acids (starting from 0) for which CAs \n "
	    << "        will be involved in the constraint \n"
	    << "    f1 and f2 X Y Z, are the coordinates for the two focus \n"
	    << "    sum-rad, is the radius sum \n"
	    << "  --jmf \n"
	    << "    Sorry, this option has been disabled in the current version.\n"
	    << "  --end-anchors \n"
	    << "    Sorry, this option has been disabled in the current version.\n";


  exit(0);
}


// FIASCO EXTENSIONS FOR HILBERT CURVE  
bitmask_t 
Utilities::convert_point_to_hilbert_value (const point& p) {
  // translate points of (256, 256, 256)
  // minimum value of a points is (-256, -256, -256)
  point coordinates = {p[0], p[1], p[2]};
  vec3 tvec = {256, 256, 256};
  Math::translate (coordinates, tvec);
  // convert real ~ to bitmask_t
  bitmask_t bit_coordinates[3];
  bit_coordinates[0] = (bitmask_t) (coordinates[0]*1000);
  bit_coordinates[1] = (bitmask_t) (coordinates[1]*1000);
  bit_coordinates[2] = (bitmask_t) (coordinates[2]*1000); 
//  return hilbert_c2i (3, 21, bit_coordinates);
  return 1;
}//-

void 
Utilities::convert_hilbert_value_to_point (const bitmask_t& hilbert_value, point& p) {
  bitmask_t coordinates[3];
//  hilbert_i2c (3, 21, hilbert_value, coordinates);
  coordinates[0] = coordinates[1] = coordinates[2] = 1;
  p[0] = (real)coordinates[0]/1000;
  p[1] = (real)coordinates[1]/1000;
  p[2] = (real)coordinates[2]/1000;
  vec3 tvec = {-256, -256, -256};
  Math::translate (p, tvec);
}//-


