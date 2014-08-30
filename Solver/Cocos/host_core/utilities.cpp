#include "utilities.h"
#include "mathematics.h"
#include "atom.h"
#include "constraint.h"
#include "logic_variables.h"

using namespace std;
using namespace Math;
using namespace Utilities;

/***************************************
 *  Input and set constraints options  *
 ***************************************/
void
Utilities::set_search_labeling_strategies () {
  vector<int> vars ( 1, 0 );
  vector<int> k_angle_shuffle_coeff ( 1, gh_params.set_size );
  vector<int> shuffle;
  for ( int i = 0; i < gh_params.mas_des.size(); i++ ) {
    if ( gh_params.mas_des[ i ].agt_type == structure ) {
      g_constraints.push_back ( new Constraint ( c_sang, gh_params.mas_des[ i ].vars_list ) );
    }
    else if ( gh_params.mas_des[ i ].agt_type == coordinator ) {
      /// -------- MONTECARLO --------
      if ( gh_params.mas_des[ i ].search_strategy == montecarlo ) {
        
        for ( int j = 0; j < gh_params.mas_des[ i ].vars_list.size(); j++ ) {
          /// Skip tails for ab-initio prediction
          if ( gh_params.sys_job == ab_initio ) {
            if ( (gh_params.mas_des[ i ].vars_list[ j ] == 0) ||
               (gh_params.mas_des[ i ].vars_list[ j ] == (gh_params.n_res-1)) ) {
                continue;
            }
          }
          
          shuffle.push_back ( gh_params.mas_des[ i ].vars_list[ j ] );
        }
        for ( int j = 0; j < gh_params.mas_des[ i ].vars_list.size(); j++ ) {
          vars[ 0 ] = gh_params.mas_des[ i ].vars_list[ j ];
          g_constraints.push_back ( new Constraint ( c_mang, vars, shuffle ) );
        }
      }
      else if ( gh_params.mas_des[ i ].search_strategy == gibbs ) {
        /// -------- GIBBS --------
        for ( int j = 0; j < gh_params.mas_des[ i ].vars_list.size(); j++ ) {
          vars[ 0 ] = gh_params.mas_des[ i ].vars_list[ j ];
          g_constraints.push_back ( new Constraint ( c_k_rang, vars ) );
        }//j
      }
      else {
        cout <<
        "Search strategy for Coordinator agent not yet supported\n";
        exit(2);
      }
    }//coordinator
  }
  cout << "Sang/Rang/Mang constraint set... \n";
}//set_search_labeling_strategies

void
Utilities::set_all_distant_constraint () {
  vector<int> all_dist_coeff ( gh_params.n_res + 2, 0 );
  for ( int i = 0; i < gh_params.mas_des.size(); i++ ) {
    /// Set proprer coeffs based on the type of agent and search method
    for ( int j = 0; j < gh_params.mas_des[ i ].vars_list.size(); j++ ) {
      if ( gh_params.mas_des[ i ].agt_type == structure ) {
        int v_dsize = g_logicvars.cp_variables[ gh_params.mas_des[ i ].vars_list[ j ] ]->get_domain_size ();
        all_dist_coeff[ g_logicvars.cp_variables[ gh_params.mas_des[ i ].vars_list[ j ] ]->get_id() ] = v_dsize;
      }
      else if ( gh_params.mas_des[ i ].agt_type == coordinator ) {
        all_dist_coeff[ g_logicvars.cp_variables[ gh_params.mas_des[ i ].vars_list[ j ] ]->get_id() ] = MAX_GIBBS_SET_SIZE;
      }
    }
    /// Set all distant constraint
    all_dist_coeff[ gh_params.n_res + 1 ] = structure;
    if ( gh_params.mas_des[ i ].agt_type == coordinator ) {
      all_dist_coeff[ gh_params.n_res ] = 1;
      all_dist_coeff[ gh_params.n_res + 1 ] = coordinator;
    }
    all_dist_coeff[ gh_params.n_res ] = 0;
    g_constraints.push_back ( new Constraint ( c_all_dist, gh_params.mas_des[ i ].vars_list, all_dist_coeff, 1 ) );
    all_dist_coeff.assign( gh_params.n_res + 2, 0 );
  }
  cout << "All_distant constraint set... \n";
}//set_all_distant_constraint

void
Utilities::set_distance_constraint () {
  /// Set manually!!!
  cout << "DISTANCE Constraint -> ";
  cout << "Set manually - Just for test!\n";
  cout << "Instructions:\n";
  cout << "1) Open file \"host_core/utilities.cpp\"\n";
  cout << "2) Fine method \"set_distance_constraint ()\"\n";
  cout << "3) Set distances between amino acids checking the coordinator agent first and last scope as given in input\n";
  cout << "4) Comment the exit(2) below!\n";
  cout << "5) Make clean\n";
  cout << "6) Make\n";
  cout << "7) Run\n";
  /// This following exit(2) to comment!!!
  exit(2);
  for ( int i = 0; i < gh_params.mas_des.size(); i++ ) {
    if ( (gh_params.mas_des[ i ].agt_type == coordinator) &&
        ((gh_params.mas_des[ i ].scope.first  >= 53) &&
         (gh_params.mas_des[ i ].scope.second <= 69) ) ) {
          vector<int> dist_coeff;
          dist_coeff.push_back( 53 );    /// First AA
          dist_coeff.push_back( 68 );   /// Second AA
          dist_coeff.push_back( 550 );  /// Distance * 100
          
          g_constraints.push_back ( new Constraint ( c_dist, gh_params.mas_des[ i ].vars_list, dist_coeff, 1 ) );
        }
    if ( (gh_params.mas_des[ i ].agt_type == coordinator) &&
        ((gh_params.mas_des[ i ].scope.first  >= 0) &&
         (gh_params.mas_des[ i ].scope.second <= gh_params.n_res) ) ) {
          vector<int> dist_coeff;
          dist_coeff.push_back( 3 );    /// First AA
          dist_coeff.push_back( 33 );   /// Second AA
          dist_coeff.push_back( 550 );  /// Distance * 100
          g_constraints.push_back ( new Constraint ( c_dist, gh_params.mas_des[ i ].vars_list, dist_coeff, 1 ) );
          
          dist_coeff.clear();
          dist_coeff.push_back( 23 );   /// First AA
          dist_coeff.push_back( 39 );   /// Second AA
          dist_coeff.push_back( 1300 );  /// Distance * 100
          g_constraints.push_back ( new Constraint ( c_dist, gh_params.mas_des[ i ].vars_list, dist_coeff, 1 ) );
          
          dist_coeff.clear();
          dist_coeff.push_back( 33 );   /// First AA
          dist_coeff.push_back( 55 );   /// Second AA
          dist_coeff.push_back( 550 );  /// Distance * 100
          g_constraints.push_back ( new Constraint ( c_dist, gh_params.mas_des[ i ].vars_list, dist_coeff, 1 ) );
        }
  }
  cout << "Distant constraint set... \n";
}//set_distance_constraint

void
Utilities::set_centroid_constraint () {
  vector<int> cg_coeff ( gh_params.n_res + 2, 0 );
  for ( int i = 0; i < gh_params.mas_des.size(); i++ ) {
    /// Set proprer coeffs based on the type of agent and search method
    for ( int j = 0; j < gh_params.mas_des[ i ].vars_list.size(); j++ ) {
      if ( gh_params.mas_des[ i ].agt_type == structure ) {
        int v_dsize = g_logicvars.cp_variables[ gh_params.mas_des[ i ].vars_list[ j ] ]->get_domain_size ();
        cg_coeff[ g_logicvars.cp_variables[ gh_params.mas_des[ i ].vars_list[ j ] ]->get_id() ] = v_dsize;
      }
      else if ( gh_params.mas_des[ i ].agt_type == coordinator ) {
        cg_coeff[ g_logicvars.cp_variables[ gh_params.mas_des[ i ].vars_list[ j ] ]->get_id() ] = MAX_GIBBS_SET_SIZE;
      }
    }
    /// Set all distant constraint
    cg_coeff[ gh_params.n_res + 1 ] = structure;
    if ( gh_params.mas_des[ i ].agt_type == coordinator ) {
      cg_coeff[ gh_params.n_res ] = 1;
      cg_coeff[ gh_params.n_res + 1 ] = coordinator;
    }
    cg_coeff[ gh_params.n_res ] = 0;
    g_constraints.push_back ( new Constraint ( c_cg, gh_params.mas_des[ i ].vars_list, cg_coeff, 2 ) );
    cg_coeff.assign( gh_params.n_res + 2, 0 );
  }
  cout << "CG constraint set... \n";
}//set_centroid_constraint

void
Utilities::set_atom_grid_constraint () {
  /// Set all distant constraint
  vector<int> coeff ( gh_params.n_res + 2, 0 );
  for ( int i = 0; i < gh_params.mas_des.size(); i++ ) {
    coeff[ gh_params.n_res + 1 ] = structure;
    if ( gh_params.mas_des[ i ].agt_type == coordinator ) {
      coeff[ gh_params.n_res ]     = 1;
      coeff[ gh_params.n_res + 1 ] = coordinator;
    }
    g_constraints.push_back ( new Constraint ( c_atom_grid, gh_params.mas_des[ i ].vars_list, coeff, 1 ) );
    coeff.assign( gh_params.n_res + 2, 0 );
  }//i
  cout << "Atom_Grid constraint set... \n";
}//set_atom_grid_constraint

/***************************************
 *           Conversion tools          *
 ***************************************/
string
Utilities::cv_string_to_str_type( ss_type type ) {
  if ( type == helix )    return "HELIX";
  if ( type == g_helix )  return "G-HELIX";
  if ( type == pi_helix ) return "PI_HELIX";
  if ( type == turn )     return "TURN";
  if ( type == sheet )    return "SHEET";
  if ( type == coil )     return "COIL";
  return "OTHER";
}//cv_string_to_str_type

ss_type
Utilities::cv_string_to_str_type( string type ) {
  if ( type.compare( "H" ) == 0 ) return helix;
  if ( type.compare( "G" ) == 0 ) return g_helix;
  if ( type.compare( "I" ) == 0 ) return pi_helix;
  if ( type.compare( "T" ) == 0 ) return turn;
  if ( type.compare( "E" ) == 0 ) return sheet;
  if ( type.compare( "C" ) == 0 ) return coil;
  return other;
}//cv_string_to_str_type

string 
Utilities::cv_aa1_to_aa3( char a ){
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
}//cv_aa1_to_aa3

string
Utilities::cv_aa3_to_aa1 ( string a ){
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
}//cv_aa3_to_aa1

string
Utilities::cv_class_to_aa3 ( aminoacid a ){
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
}//cv_class_to_aa3

aminoacid
Utilities::cv_aa_to_class ( char a ){
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
    cout << "Error in cv_aa_to_class\n";
    getchar();
    return err;
}//cv_aa_to_class

aminoacid
Utilities::cv_aa_to_class ( string a ) {
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
}//cv_aa_to_class

int
Utilities::cv_class_to_n(aminoacid a){
  if (a==ala) return 0;
  if (a==arg) return 1;
  if (a==asn) return 2;
  if (a==asp) return 3;
  if (a==cys) return 4;
  if (a==gln) return 5;
  if (a==glu) return 6;
  if (a==gly) return 7;
  if (a==his) return 8;
  if (a==ile) return 9;
  if (a==leu) return 10;
  if (a==lys) return 11;
  if (a==met) return 12;
  if (a==phe) return 13;
  if (a==pro) return 14;
  if (a==ser) return 15;
  if (a==thr) return 16;
  if (a==trp) return 17;
  if (a==tyr) return 18;
  if (a==val) return 19;
  cout << "#error (cv_class_to_n -> " << a << ")" << endl;
  getchar();
  return -1;
}//cv_class_to_n

atom_type
Utilities::get_atom_type (string name) {
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
  else if (name.find(" H1 ")  != string::npos ||
           name.find(" H1 A") != string::npos ||
           name.find(" HA ")  != string::npos )
    return H;
  else if (name.find(" S  ") != string::npos ||
           name.find(" S  A") != string::npos)
    return S;
  else return X;
}//get_atom_type

atom_type
Utilities::cv_string_to_atom_type( string name ) {
  if ( name.find("N") != string::npos )
    return N;
  else if ( name.find("CA") != string::npos )
    return CA;
  else if ( name.find("C") != string::npos )
    return CB;
  else if ( name.find("O") != string::npos )
    return O;
  else if ( name.find("H") != string::npos )
    return H;
  else return X;
}//cv_cv_string_to_str_type

/***************************************
 *      Offsets and Atom postions      *
 ***************************************/
// @Note: N - Ca - C - O - H
atom_type
Utilities::get_atom_type( uint bbidx ) {
  switch ( bbidx % 5 ) {
    case 0:
      return N;
    case 1:
      return CA;
    case 2:
      return CB;
    case 3:
      return O;
    case 4:
      return H;
  }
  return X;
}//get_atom_type

atom_radii 
Utilities::get_atom_radii ( uint bbidx ) {
  switch ( bbidx % 5 ) {
    case 0:
      return rN;
    case 1:
      return rC;
    case 2:
      return rC;
    case 3:
      return rO;
    case 4:
      return rH;
  }//switch
  return rC;
}//get_atom_radii

int
Utilities::get_aaidx_from_bbidx ( uint bbidx, atom_type type ) {
  switch ( type ) {
    case N:
      return bbidx/5;
    case CA:
      return (bbidx-1)/5;
    case CB:
      return (bbidx-2)/5;
    case O:
      return (bbidx-3)/5;
    case H:
      return (bbidx-4)/5;
    default:
      cout << "#error (get_aaidx_from_bbidx)" << endl;
      getchar();
      return -1;
  }
}//get_aaidx_from_bbidx

int 
Utilities::get_bbidx_from_aaidx ( uint aaidx, atom_type type ) {
  switch ( type ) {
    case N:
      return aaidx*5;
    case CA:
      return aaidx*5+1;
    case CB:
      return aaidx*5+2;
    case O:
      return aaidx*5+3;
    case H:
      return aaidx*5+4;
    default:
      cout << "#error (get_bbidx_from_aaidx)" << endl;
      getchar();
      return -1;
  } 
}//get_bbidx_from_aaidx

void
Utilities::calculate_aa_points( bool dir, real bb[] ) {
  real x = 0, y = 0, z = 0;
  
  // X offset values
  real l_can_x = 1.283;   // x: N  <- Ca
  real l_nc_x  = 1.109;   // x: C  <- N
  real l_cac_x = 1.301;   // x: Ca -> C
  real l_cn_x  = 1.109;   // x: C  -> N
  // Y offset values
  real l_can_y = 0.696;   // y: N  <- Ca
  real l_nc_y  = 0.734;   // y: C  <- N
  real l_cac_y = 0.766;   // y: Ca -> C
  real l_cn_y  = 0.734;   // y: C  -> N
  real l_nh_y  = 1.000;   // y: N <-> H
  real l_co_y  = 1.240;   // y: C <-> O
  
  if ( dir ) {
    l_can_y *= -1.0;
    l_nc_y  *= -1.0;
    l_cac_y *= -1.0;
    l_cn_y  = l_nc_y;
    l_nh_y  *= -1.0;
    l_co_y  *= -1.0;
  }
  ///LEFT-C
  bb[ 0 ]  = x - l_can_x - l_nc_x;
  bb[ 1 ]  = y + l_can_y - l_nc_y;
  bb[ 2 ]  = z;
  ///LEFT-O
  bb[ 3 ]  = x - l_can_x - l_nc_x;
  bb[ 4 ]  = y + l_can_y - l_nc_y - l_co_y;
  bb[ 5 ]  = z;
  ///LEFT-H
  bb[ 6 ]  = x - l_can_x;
  bb[ 7 ]  = y + l_can_y + l_nh_y;
  bb[ 8 ]  = z;
  ///LEFT-N
  bb[ 9 ]  = x - l_can_x;
  bb[ 10 ] = y + l_can_y;
  bb[ 11 ] = z;
  ///CENTRAL-CA
  bb[ 12 ] = x;
  bb[ 13 ] = y;
  bb[ 14 ] = z;
  ///RIGHT-C
  bb[ 15 ] = x + l_cac_x;
  bb[ 16 ] = y + l_cac_y;
  bb[ 17 ] = z;
  ///RIGHT-O
  bb[ 18 ] = x + l_cac_x;
  bb[ 19 ] = y + l_cac_y + l_co_y;
  bb[ 20 ] = z;
  ///RIGHT-H
  bb[ 21 ] = x + l_cac_x + l_cn_x;
  bb[ 22 ] = y + l_cac_y - l_cn_y - l_nh_y;
  bb[ 23 ] = z;
  ///RIGHT-N
  bb[ 24 ] = x + l_cac_x + l_cn_x;
  bb[ 25 ] = y + l_cac_y - l_cn_y;
  bb[ 26 ] = z;
}//calculate_aa_points

/***************************************
 *        Overalp and Rotations        *
 ***************************************/

void
Utilities::overlap_structures ( point& pa, point& pb, point& pc,
                                point * str_out,
                                int len ) {
  R_MAT rot_m;
  vec3  shift_v;
  compute_normal_base( str_out, rot_m, shift_v );
  change_coordinate_system( str_out, rot_m, shift_v, len );
  overlap( pa, pb, pc, str_out, len );
}//overlap_structures

/*
 * @note:
 * str_out must be of size equal to str_right
 */
void
Utilities::overlap_structures ( point& pa, point& pb, point& pc,
                                point * str_in, point * str_out,
                                int len, int offset ) {
  for ( int i = 0; i < len; i++ ) {
    str_out[ i ][ 0 ] = str_in[ i + offset ][ 0 ];
    str_out[ i ][ 1 ] = str_in[ i + offset ][ 1 ];
    str_out[ i ][ 2 ] = str_in[ i + offset ][ 2 ];
  }
  
  R_MAT rot_m;
  vec3  shift_v;
  compute_normal_base( str_out, rot_m, shift_v );
  change_coordinate_system( str_out, rot_m, shift_v, len );
  overlap( pa, pb, pc, str_out, len );
}//overlap_structures

/*
 * @note:
 * str_left must be of size equal to str_right and to the target protein
 */
void
Utilities::overlap_structures ( point * str_left, point * str_right, int aa_idx ) {
  // First we save the part of the structure to overlap
  int num_aa = gh_params.target_protein->get_nres() - aa_idx - 1;
  int num_bb = 5 * num_aa + 7;
  int bb_star = get_bbidx_from_aaidx ( aa_idx - 1, CB );
  point pa, pb, pc;
  point str_to_overlap [ num_bb ];
  for (int i = 0; i < num_bb; i++) {
    str_to_overlap[ i ][ 0 ] = str_right[ bb_star + i ][ 0 ];
    str_to_overlap[ i ][ 1 ] = str_right[ bb_star + i ][ 1 ];
    str_to_overlap[ i ][ 2 ] = str_right[ bb_star + i ][ 2 ];
  }
  for (int i = 0; i < 3; i++) {
    pa[ i ] = str_left[ bb_star     ][ i ];
    pb[ i ] = str_left[ bb_star + 1 ][ i ];
    pc[ i ] = str_left[ bb_star + 3 ][ i ];
  }
  
  // Then we overlap the old structure on the new points
  R_MAT rot_m;
  vec3  shift_v;
  compute_normal_base( str_to_overlap, rot_m, shift_v );
  change_coordinate_system( str_to_overlap, rot_m, shift_v, num_bb );
  overlap( pa, pb, pc, str_to_overlap, num_bb );
  
  //Now we copy of the overlapped structure 
  for ( int i = 0; i < num_bb; i++ ) {
    str_left[ bb_star + i ][ 0 ] = str_to_overlap[ i ][ 0 ];
    str_left[ bb_star + i ][ 1 ] = str_to_overlap[ i ][ 1 ];
    str_left[ bb_star + i ][ 2 ] = str_to_overlap[ i ][ 2 ];
  }
}//overlap_structures

void
Utilities::compute_normal_base( point * backbone, real rot_m[3][3], real shift_v[3] ) {
  vec3 x, y, z, v;
  // Build the plane for rotation
  for (int i = 0; i < 3; i++) {
    v[ i ] = backbone[ 1 ][ i ] - backbone[ 0 ][ i ];
    z[ i ] = backbone[ 3 ][ i ] - backbone[ 1 ][ i ];
  }
  
  // Build the Orthogonal Base
  Math::vcross ( z, v, y ); 	// y orthogonal to z, v
  Math::vcross ( y, z, x );   // x orthogonal to z and y
  
  // Normalize: Obtain the Orthonormal Base
  Math::vnorm ( x );
  Math::vnorm ( y );
  Math::vnorm ( z );
  
  // Build the Rotation matrix (orthonormal)
  shift_v[0] = shift_v[1] = shift_v[2] = 0;
  for (int i = 0; i < 3; i++) {
    rot_m[ i ][ 0 ] = x[ i ];
    rot_m[ i ][ 1 ] = y[ i ];
    rot_m[ i ][ 2 ] = z[ i ];
    shift_v[i] -= backbone[ 0 ][ i ];
  }
}//compute_normal_base

void
Utilities::change_coordinate_system ( point * backbone, real rot_m[3][3], real shift_v[3], int bb_len ) {
  // Translate the vector in 0,0,0 : tt = (f - s0)
  real tt[ 3*bb_len ];
  for (uint i = 0; i < bb_len; i++){
    tt[ 3*i + 0 ] = backbone[i][0] + shift_v[0];
    tt[ 3*i + 1 ] = backbone[i][1] + shift_v[1];
    tt[ 3*i + 2 ] = backbone[i][2] + shift_v[2];
  }
  
  // Rotate to the transport of R the fragment
  for (uint i = 0; i < bb_len; i++){
    for (uint j = 0; j < 3; j++) {
      backbone[i][j] =
      rot_m[0][j] * tt[3*i]     +
      rot_m[1][j] * tt[3*i + 1] +
      rot_m[2][j] * tt[3*i + 2];
      if (std::abs((double)backbone[i][j]) < 1.0e-5) backbone[i][j] = 0.0;
    }
  }
}//change_coordinate_system

/* Overlap right on N by default */
void
Utilities::overlap ( point& p1, point& p2, point& p3,
                     point * backbone, int bb_len, int offset ) {
  // Build rotation plane
  vec3 x, y, z, v;
  Math::vsub ( p2, p1, v );  // v = p2 - p1
  Math::vsub ( p3, p2, z );  // z = p3 - p2
  
  // Build the Orthogonal Base
  Math::vcross ( z, v, y ); // y orthogonal to z, v
  Math::vcross ( y, z, x ); // x orthogonal to z and y
  
  // Normalize: Obtain the Orthonormal Base
  Math::vnorm ( x );
  Math::vnorm ( y );
  Math::vnorm ( z );
  
  // Build the Rotation matrix (orthonormal)
  R_MAT rot_m;
  for ( uint i = 0; i < 3; i++ ) {
    rot_m[ i ][ 0 ] = x[ i ];
    rot_m[ i ][ 1 ] = y[ i ];
    rot_m[ i ][ 2 ] = z[ i ];
  }
  
  // f_t = RotMat(p1, p2, p3) x f
  for ( uint i = 0; i < bb_len; i++ ) {
    real px =
    rot_m[0][0] * backbone[i][0] +
    rot_m[0][1] * backbone[i][1] +
    rot_m[0][2] * backbone[i][2];
    real py =
    rot_m[1][0] * backbone[i][0] +
    rot_m[1][1] * backbone[i][1] +
    rot_m[1][2] * backbone[i][2];
    real pz =
    rot_m[2][0] * backbone[i][0] +
    rot_m[2][1] * backbone[i][1] +
    rot_m[2][2] * backbone[i][2];
    
    if (std::abs((double)px) < 1.0e-4) px = 0;
    if (std::abs((double)py) < 1.0e-4) py = 0;
    if (std::abs((double)pz) < 1.0e-4) pz = 0;
    
    backbone[ i ][ 0 ] = px;
    backbone[ i ][ 1 ] = py;
    backbone[ i ][ 2 ] = pz;
  }
  
  /*
   * Translate the fragment so that it overlaps f on the plan < p1, p2, p3 >,
   * superimposition on a3 and the third point of f.
   */
  vec3  shift_v;
  shift_v[0] = -backbone[ offset ][0] + p3[0];
  shift_v[1] = -backbone[ offset ][1] + p3[1];
  shift_v[2] = -backbone[ offset ][2] + p3[2];
  
  for ( uint i = 0; i < bb_len; i++ ) {
    backbone[ i ][ 0 ] += shift_v[0];
    backbone[ i ][ 1 ] += shift_v[1];
    backbone[ i ][ 2 ] += shift_v[2];
  }
}//overlap

void
Utilities::translate_structure ( real* structure, int reference, real x, real y, real z, int length ) {
  x -= structure[ reference*3 + 0 ];
  y -= structure[ reference*3 + 1 ];
  z -= structure[ reference*3 + 2 ];
  for (int i = 0; i < length; i++) {
    structure[ i*3 + 0 ] += x;
    structure[ i*3 + 1 ] += y;
    structure[ i*3 + 2 ] += z;
  }//i
}//translate_structure


/***************************************
 *          I/O aux functions          *
 ***************************************/
void 
Utilities::output_pdb_format ( string outf, const vector< Atom >& vec, real energy ) {
    FILE *fid;
    char fx[4], fy[4], fz[4];
    int k = -1;
    real x,y,z;
    /* Open an output file */
    fid = fopen ( outf.c_str(), "a" );
    if (fid < 0){
        printf("Cannot open %s to write!\n", outf.c_str());
        return;
    }    
    int atom=1;
    if ( energy == 0 ) energy = gh_params.minimum_energy;
  
    fprintf(fid, "MODEL    001\n");
    fprintf(fid, "REMARK \t ENERGY %f\n", energy );
    //fprintf(fid, "MODEL    001\n");
    
    /* Write the solution into the output file */;
    for (uint i = 0; i < vec.size(); i++) {
        if ( vec[i].is_type(N) ) k++;
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
        
        fprintf(fid,"ATOM    %3d  ",atom++);
        if (vec.at(i).type == N)
            fprintf (fid, "N   ");
        if (vec.at(i).type == H)
            fprintf (fid, "H   ");
        if (vec.at(i).type == CA)
            fprintf (fid, "CA  ");
        if (vec.at(i).type == CB)
            fprintf (fid, "C   ");
        if (vec.at(i).type == O)
            fprintf (fid, "O   ");
        
        fprintf (fid, "XXX A %3d    %s%3.3f%s%3.3f%s%3.3f  1.00  1.00\n",
                 k, fx, x, fy, y, fz, z);
    }
    fprintf(fid,"ENDMDL\n");
  fclose(fid);   
}//output_pdb_format

string
Utilities::output_pdb_format( point* structure, int len, real rmsd, real energy ){
  stringstream s;
  real x, y, z;
  int aa_idx = -1;
  int atom_s = 0, atom_e = len;
  int atom_counter = 0;
  if ( energy == 0 ) energy = gh_params.minimum_energy;
  
  //s << "REMARK \t ENERGY: " << energy << endl;
  //if ( rmsd > 0 ) s << "REMARK \t RMSD: " << rmsd << endl;
  s << "MODEL " << gh_params.num_models++ << endl;
  s << "REMARK \t ENERGY: " << energy << endl;
  for (int i = atom_s; i < atom_e; i++) {
    atom_counter++;
    
    x = structure[i][0];
    y = structure[i][1];
    z = structure[i][2];
    
    s << "ATOM   "<< setw(4) << i+1 << "  ";
    
    if (i%5 == 0){
      s << "N   ";
      aa_idx = get_aaidx_from_bbidx(i, N);
    }
    if (i%5 == 1){
      s << "CA  ";
      aa_idx = get_aaidx_from_bbidx(i, CA);
    }
    if (i%5 == 2){
      s << "C   ";
      aa_idx = get_aaidx_from_bbidx(i, CB);
    }
    if (i%5 == 3) {
      s << "O   ";
      aa_idx = get_aaidx_from_bbidx(i, O);
    }
    if (i%5 == 4) {
      s << "H   ";
      aa_idx = get_aaidx_from_bbidx(i, H);
    }
    
    s << cv_aa1_to_aa3( gh_params.target_protein->get_sequence()[ aa_idx ] )
    << " A "
    << setw(3) << get_aaidx_from_bbidx(i, atom_type(i%5))
    << "    "
    << fixed;

    if (fabs(x) >= 100) {
      if (fabs(y) >= 100) {
        if (fabs(z) >= 100) {
          s
	    << get_format_spaces(x) << setprecision(2) << x
	    << get_format_spaces(y) << setprecision(2) << y
	    << get_format_spaces(z) << setprecision(2) << z
	    << "  1.00  1.00\n";
        }
        else {
          s
	    << get_format_spaces(x) << setprecision(2) << x
	    << get_format_spaces(y) << setprecision(2) << y
	    << get_format_spaces(z) << setprecision(3) << z
	    << "  1.00  1.00\n";
        }
      }
      else {
        if (fabs(z) >= 100) {
          s
	    << get_format_spaces(x) << setprecision(2) << x
	    << get_format_spaces(y) << setprecision(3) << y
	    << get_format_spaces(z) << setprecision(2) << z
	    << "  1.00  1.00\n";
        }
        else {
          s
	    << get_format_spaces(x) << setprecision(2) << x
	    << get_format_spaces(y) << setprecision(3) << y
	    << get_format_spaces(z) << setprecision(3) << z
	    << "  1.00  1.00\n";
        }
      }
    }
    else {
      if (fabs(y) >= 100) {
        if (fabs(z) >= 100) {
          s
	    << get_format_spaces(x) << setprecision(3) << x
	    << get_format_spaces(y) << setprecision(2) << y
	    << get_format_spaces(z) << setprecision(2) << z
	    << "  1.00  1.00\n";
        }
        else {
          s
	    << get_format_spaces(x) << setprecision(3) << x
	    << get_format_spaces(y) << setprecision(2) << y
	    << get_format_spaces(z) << setprecision(3) << z
	    << "  1.00  1.00\n";
        }
      }
      else {
        if (fabs(z) >= 100) {
          s
	    << get_format_spaces(x) << setprecision(3) << x
	    << get_format_spaces(y) << setprecision(3) << y
	    << get_format_spaces(z) << setprecision(2) << z
	    << "  1.00  1.00\n";
        }
        else {
          s
	    << get_format_spaces(x) << setprecision(3) << x
	    << get_format_spaces(y) << setprecision(3) << y
	    << get_format_spaces(z) << setprecision(3) << z
	    << "  1.00  1.00\n";
        }
      }
    }
  }
  
  atom_counter++;
  
  int  CG_radius;
  real my_CG[ 3 ];
  for ( int i = 0; i < ((len/5) - 2); i++ ) {
    calculate_cg_atom( cv_aa_to_class ( gh_params.target_protein->get_sequence()[ i+1 ] ),
                       structure[ i*5 + 1    ],
                       structure[ (i+1)*5 +1 ],
                       structure[ (i+2)*5 +1 ],
                       my_CG, &CG_radius );
    x = my_CG[ 0 ];
    y = my_CG[ 1 ];
    z = my_CG[ 2 ];
    
    s<<"ATOM   "
    <<setw(4)<<atom_counter++
    <<"  CG  "
    <<cv_aa1_to_aa3( gh_params.target_protein->get_sequence()[ i+1 ] )
    <<" A "<<setw(3)<<i+2<<"    "
    <<fixed
    <<get_format_spaces(x)<<setprecision(3)<<x
    <<get_format_spaces(y)<<setprecision(3)<<y
    <<get_format_spaces(z)<<setprecision(3)<<z
    <<"  1.00  1.00\n";
  }//i
  
  s << "ENDMDL\n";
  return s.str();
}//print_results

string
Utilities::output_pdb_format( real* structure, real rmsd, real energy ){
  stringstream s;
  real x, y, z;
  int len = (gh_params.n_res) * 5;
  int aa_idx = -1;
  int atom_s = 0, atom_e = len;
  if ( energy == 0 ) energy = gh_params.minimum_energy;
  
  s << "MODEL 0\n";
  s << "REMARK \t ENERGY: " << energy << endl;
  //if ( rmsd > 0 ) s << "REMARK \t RMSD: " << rmsd << endl;
  for (int i = atom_s; i < atom_e; i++) {
    x = structure[ 3*i + 0 ];
    y = structure[ 3*i + 1 ];
    z = structure[ 3*i + 2 ];
    
    s << "ATOM   "<< setw(4) << i+1 << "  ";
    
    if (i%5 == 0){
      s << "N   ";
      aa_idx = get_aaidx_from_bbidx(i, N);
    }
    if (i%5 == 1){
      s << "CA  ";
      aa_idx = get_aaidx_from_bbidx(i, CA);
    }
    if (i%5 == 2){
      s << "C   ";
      aa_idx = get_aaidx_from_bbidx(i, CB);
    }
    if (i%5 == 3) {
      s << "O   ";
      aa_idx = get_aaidx_from_bbidx(i, O);
    }
    if (i%5 == 4) {
      s << "H   ";
      aa_idx = get_aaidx_from_bbidx(i, H);
    }
    
    s << cv_aa1_to_aa3( gh_params.target_protein->get_sequence()[ aa_idx ] )
    << " A "
    << setw(3) << get_aaidx_from_bbidx(i, atom_type(i%5))
    << "    "
    << fixed;
    if (fabs(x) >= 100) {
      if (fabs(y) >= 100) {
        if (fabs(z) >= 100) {
          s
	    << get_format_spaces(x) << setprecision(2) << x
	    << get_format_spaces(y) << setprecision(2) << y
	    << get_format_spaces(z) << setprecision(2) << z
	    << "  1.00  1.00\n";
        }
        else {
          s
	    << get_format_spaces(x) << setprecision(2) << x
	    << get_format_spaces(y) << setprecision(2) << y
	    << get_format_spaces(z) << setprecision(3) << z
	    << "  1.00  1.00\n";
        }
      }
      else {
        if (fabs(z) >= 100) {
          s
	    << get_format_spaces(x) << setprecision(2) << x
	    << get_format_spaces(y) << setprecision(3) << y
	    << get_format_spaces(z) << setprecision(2) << z
	    << "  1.00  1.00\n";
        }
        else {
          s
	    << get_format_spaces(x) << setprecision(2) << x
	    << get_format_spaces(y) << setprecision(3) << y
	    << get_format_spaces(z) << setprecision(3) << z
	    << "  1.00  1.00\n";
        }
      }
    }
    else {
      if (fabs(y) >= 100) {
        if (fabs(z) >= 100) {
          s
	    << get_format_spaces(x) << setprecision(3) << x
	    << get_format_spaces(y) << setprecision(2) << y
	    << get_format_spaces(z) << setprecision(2) << z
	    << "  1.00  1.00\n";
        }
        else {
          s
	    << get_format_spaces(x) << setprecision(3) << x
	    << get_format_spaces(y) << setprecision(2) << y
	    << get_format_spaces(z) << setprecision(3) << z
	    << "  1.00  1.00\n";
        }
      }
      else {
        if (fabs(z) >= 100) {
          s
	    << get_format_spaces(x) << setprecision(3) << x
	    << get_format_spaces(y) << setprecision(3) << y
	    << get_format_spaces(z) << setprecision(2) << z
	    << "  1.00  1.00\n";
        }
        else {
          s
	    << get_format_spaces(x) << setprecision(3) << x
	    << get_format_spaces(y) << setprecision(3) << y
	    << get_format_spaces(z) << setprecision(3) << z
	    << "  1.00  1.00\n";
        }
      }
    }
  }
  s << "ENDMDL\n";
  return s.str();
}//print_results

int 
Utilities::get_format_digits(real x) {
  if (x < 0 && x > -10)     return 4;
  if (x >= 0 && x < 10)     return 4;
  if (x > 10 && x < 100)    return 5;   
  if (x > 100 || x < -100)  return 6;
  return 5;
}//get_format_digits

string 
Utilities::get_format_spaces(real x) {
  if (x < 0 && x > -10)  return "  ";
  if (x >= 0 && x < 10)  return "   ";
  if (x > 10 && x < 100) return "  ";   
  return " ";
}//get_format_spaces


/***************************************
 *               Display               *
 ***************************************/
void
Utilities::print_debug ( std::string s ) {
  if ( gh_params.verbose )
    cout << s << endl;
}//print_debug

void
Utilities::print_debug ( std::string s1, std::string s2 ) {
  if ( gh_params.verbose )
    cout << s1 << s2 << endl;
}//print_debug

/*
template <class T>
void
Utilities::print_debug ( std::string s, T param ) {
  if ( gh_params.verbose ) {
    ostringstream s_val;
    s_val << param;
    cout << s << s_val.str() << endl;
  }
}//print_debug

template <class T>
void
Utilities::print_debug ( std::string s1, std::string s2, T param ) {
  if ( gh_params.verbose ) {
    ostringstream s_val;
    s_val << param;
    cout << s1 << s2 << s_val.str() << endl;
  }
}//print_debug
*/

void
Utilities::dump(point a) {
  cout << "<" << a[0] << ", " << a[1] << ", " << a[2] << ">" << endl;
}//dump

void
Utilities::dump(point a, point b) {
  cout << "[("<< a[0] << ", " << a[1] << ", " << a[2] << ") -- ";
  cout << "[("<< b[0] << ", " << b[1] << ", " << b[2] << ")]";
}//dump

void
Utilities::dump(std::vector< std::vector <real> > str) {
  point a;
  for (uint i = 0; i < str.size(); i++) {
    for (int j = 0; j < 3; j++) a[j] = str[i][j];
    cout << i << " ";
    dump(a);
  }
}//dump

void
Utilities::rotate_point_about_line( real* in_point, real theta_rad,
                                   real* p1, real* p2,
                                   real* out_point ) {
  real u[3];
  real q2[3];
  real scale;
  
  /// Step 1
  out_point[ 0 ] = in_point[ 0 ] - p1[ 0 ];
  out_point[ 1 ] = in_point[ 1 ] - p1[ 1 ];
  out_point[ 2 ] = in_point[ 2 ] - p1[ 2 ];
  u[ 0 ]         = p2[ 0 ] - p1[ 0 ];
  u[ 1 ]         = p2[ 1 ] - p1[ 1 ];
  u[ 2 ]         = p2[ 2 ] - p1[ 2 ];
  
  scale = sqrt ( ( u[0]*u[0] ) + ( u[1]*u[1] ) + ( u[2]*u[2] ) );
  if ( scale > CLOSE_TO_ZERO_VAL ) {
    u[0] /= scale;
    u[1] /= scale;
    u[2] /= scale;
  }
  else
    u[0] = u[1] = u[2] = 0.0;
  
  /// Step 2
  scale = sqrt ( ( u[1]*u[1] ) + ( u[2]*u[2] ) );
  if ( scale > CLOSE_TO_ZERO_VAL ) {
    q2[0] = out_point[0];
    q2[1] = out_point[1] * u[2] / scale - out_point[2] * u[1] / scale;
    q2[2] = out_point[1] * u[1] / scale + out_point[2] * u[2] / scale;
  }
  else {
    q2[0] = out_point[0];
    q2[1] = out_point[1];
    q2[2] = out_point[2];
    scale = 0.0;
  }
  
  /// Step 3
  out_point[0] = q2[0] * scale - q2[2] * u[0];
  out_point[1] = q2[1];
  out_point[2] = q2[0] * u[0] + q2[2] * scale;
  
  /// Step 4
  q2[0] = out_point[0] * cos( theta_rad ) - out_point[1] * sin( theta_rad );
  q2[1] = out_point[0] * sin( theta_rad ) + out_point[1] * cos( theta_rad );
  q2[2] = out_point[2];
  
  /// Inverse of step 3
  out_point[0] =   q2[0] * scale + q2[2] * u[0];
  out_point[1] =   q2[1];
  out_point[2] = - q2[0] * u[0] + q2[2] * scale;
  
  /// Inverse of step 2
  if ( scale > CLOSE_TO_ZERO_VAL ) {
    q2[0] =   out_point[0];
    q2[1] =   out_point[1] * u[2] / scale + out_point[2] * u[1] / scale;
    q2[2] = - out_point[1] * u[1] / scale + out_point[2] * u[2] / scale;
  }
  else {
    q2[0] = out_point[0];
    q2[1] = out_point[1];
    q2[2] = out_point[2];
  }
  
  /// Inverse of step 1
  out_point[ 0 ] = q2[ 0 ] + p1[ 0 ];
  out_point[ 1 ] = q2[ 1 ] + p1[ 1 ];
  out_point[ 2 ] = q2[ 2 ] + p1[ 2 ];
}//rotate_point_about_line

void
Utilities::move_phi ( real * aa_points, real degree,
                      int v_id, int ca_pos, int first_res, int threadIdx ) {
  
  if ( threadIdx >= first_res ) {
    int ca_res = ( threadIdx * 5 + 1 ) * 3;
    if ( threadIdx < v_id ) {
      ///ROTATE LEFT-CA
      rotate_point_about_line( &aa_points[ ca_res ], degree,
                               &aa_points[ ca_pos ], &aa_points[ ca_pos-3 ],
                               &aa_points[ ca_res ] );
      ///ROTATE LEFT-N
      rotate_point_about_line( &aa_points[ ca_res-3 ], degree,
                               &aa_points[ ca_pos ], &aa_points[ ca_pos-3 ],
                               &aa_points[ ca_res-3 ] );
    }
    
    ///ROTATE LEFT-H
    rotate_point_about_line( &aa_points[ ca_res+9 ], degree,
                             &aa_points[ ca_pos ], &aa_points[ ca_pos-3 ],
                             &aa_points[ ca_res+9 ] );
    
    if ( threadIdx > 0 ) {
      ///ROTATE LEFT-O
      rotate_point_about_line( &aa_points[ ca_res-9 ], degree,
                               &aa_points[ ca_pos ], &aa_points[ ca_pos-3 ],
                               &aa_points[ ca_res-9 ] );
      ///ROTATE LEFT-C
      rotate_point_about_line( &aa_points[ ca_res-12 ], degree,
                               &aa_points[ ca_pos ], &aa_points[ ca_pos-3 ],
                               &aa_points[ ca_res-12 ] );
    }
  }
}//move_phi


void
Utilities::move_psi ( real * aa_points, real degree,
                     int v_id, int ca_pos, int last_res, int threadIdx ) {
  if ( threadIdx < last_res ) {
    int ca_res = ( threadIdx * 5 + 1 ) * 3;
    if ( threadIdx > v_id ) {
      ///ROTATE RIGHT-CA
      rotate_point_about_line( &aa_points[ ca_res ], degree,
                              &aa_points[ ca_pos ], &aa_points[ ca_pos+3 ],
                              &aa_points[ ca_res ] );
      ///ROTATE RIGHT-C
      rotate_point_about_line( &aa_points[ ca_res+3 ], degree,
                              &aa_points[ ca_pos ], &aa_points[ ca_pos+3 ],
                              &aa_points[ ca_res+3 ] );
      
    }
    
    ///ROTATE RIGHT-O
    rotate_point_about_line( &aa_points[ ca_res+6 ], degree,
                            &aa_points[ ca_pos ], &aa_points[ ca_pos+3 ],
                            &aa_points[ ca_res+6 ] );
    
    if ( threadIdx > v_id ) {
      ///ROTATE RIGHT-H
      rotate_point_about_line( &aa_points[ ca_res+9 ], degree,
                              &aa_points[ ca_pos ], &aa_points[ ca_pos+3 ],
                              &aa_points[ ca_res+9 ] );
    }
    
    if ( threadIdx < last_res-1 ) {
      ///ROTATE RIGHT-N
      rotate_point_about_line( &aa_points[ ca_res+12 ], degree,
                              &aa_points[ ca_pos ], &aa_points[ ca_pos+3 ],
                              &aa_points[ ca_res+12 ] );
    }
  }
}//move_psi

  void
Utilities::copy_structure_from_to ( real* s1, real* s2, int n_threads) {
  /// Copy back the rotated structure
  memcpy ( s1, s2, n_threads * 15*sizeof(real) );
}//copy_structure_from_to

void
Utilities::calculate_cg_atom ( aminoacid a,
                              real* ca1, real* ca2, real* ca3,
                              real* cg, int* radius ) {
  /// Placement of the centroid using dist, chi2, e tors
  /// v1 is the normalized vector w.r.t. ca1, ca2
  real v1[3];
  vsub ( ca2, ca1, v1 );
  vnorm ( v1 );
  
  /// v2 is the normalized vector w.r.t. ca2, ca3
  real v2[3];
  vsub ( ca3, ca2, v2 );
  vnorm ( v2 );
  
  /// Compute v1 (subtracting the component along v2)
  /// in order to obtain v1 and v2 orthogonal each other
  real x = vdot ( v1, v2 );
  v1[ 0 ] = v1[ 0 ] - x * v2[ 0 ];
  v1[ 1 ] = v1[ 1 ] - x * v2[ 1 ];
  v1[ 2 ] = v1[ 2 ] - x * v2[ 2 ];
  vnorm ( v1 );
  
  /// Compute v3 orthogonal to v1 and v2
  real v3[3];
  vcross ( v1, v2, v3 );
  
  /// Using Cramer method
  real factor;
  real b[3];
  real R[3][3];
  real D, Dx, Dy, Dz;
  real tors   = centroid_torsional_angle ( a ) * PI_VAL/180;
  b[0] = cos( (centroid_chi2 ( a )) * PI_VAL/180 );
  factor = sqrt( 1 - ( b[0] * b[0] ) );
  b[1] = sin( tors ) * factor ;
  b[2] = cos( tors ) * factor ;
  
  R[0][0] = v2[0];
  R[0][1] = v2[1];
  R[0][2] = v2[2];
  
  R[1][0] = v3[0];
  R[1][1] = v3[1];
  R[1][2] = v3[2];
  
  R[2][0] = -v1[0];
  R[2][1] = -v1[1];
  R[2][2] = -v1[2];
  
  D =
  R[0][0] * R[1][1] * R[2][2] +
  R[0][1] * R[1][2] * R[2][0] +
  R[0][2] * R[1][0] * R[2][1] -
  R[0][2] * R[1][1] * R[2][0] -
  R[0][1] * R[1][0] * R[2][2] -
  R[0][0] * R[1][2] * R[2][1];
  Dx =
  b[0] * (R[1][1] * R[2][2] - R[1][2] * R[2][1]) +
  b[1] * (R[2][1] * R[0][2] - R[2][2] * R[0][1]) +
  b[2] * (R[0][1] * R[1][2] - R[0][2] * R[1][1]) ;
  Dy =
  b[0] * (R[1][2] * R[2][0] - R[1][0] * R[2][2]) +
  b[1] * (R[2][2] * R[0][0] - R[2][0] * R[0][2]) +
  b[2] * (R[0][2] * R[1][0] - R[0][0] * R[1][2]) ;
  Dz =
  b[0] * (R[1][0] * R[2][1] - R[1][1] * R[2][0]) +
  b[1] * (R[2][0] * R[0][1] - R[2][1] * R[0][0]) +
  b[2] * (R[0][0] * R[1][1] - R[0][1] * R[1][0]) ;
  
  real v[3];
  v[ 0 ] = Dx/D;
  v[ 1 ] = Dy/D;
  v[ 2 ] = Dz/D;
  
  /// Now compute centroids coordinates
  v[ 0 ] = centroid_distance( a ) * v[ 0 ];
  v[ 1 ] = centroid_distance( a ) * v[ 1 ];
  v[ 2 ] = centroid_distance( a ) * v[ 2 ];
  
  // Update the output
  vadd ( v, ca2, cg );
  *radius = centroid_radius( a );
}//calculate_cg_atom

real
Utilities::centroid_torsional_angle ( aminoacid a ) {
  if (a==ala) return -138.45;
  if (a==arg) return -155.07;
  if (a==asn) return -144.56;
  if (a==asp) return -142.28;
  if (a==cys) return -142.28;
  if (a==gln) return -149.99;
  if (a==glu) return -147.56;
  if (a==gly) return -0;
  if (a==his) return -144.08;
  if (a==ile) return -151.72;
  if (a==leu) return -153.24;
  if (a==lys) return -153.03;
  if (a==met) return -159.50;
  if (a==phe) return -146.92;
  if (a==pro) return -105.62;
  if (a==ser) return -139.94;
  if (a==thr) return -142.28;
  if (a==trp) return -155.35;
  if (a==tyr) return -149.29;
  if (a==val) return -150.47;
  return 0;
}//centroid_torsional_angle

real
Utilities::centroid_chi2 ( aminoacid a ) {
  if (a==ala) return 110.53;
  if (a==arg) return 113.59;
  if (a==asn) return 117.73;
  if (a==asp) return 116.03;
  if (a==cys) return 115.36;
  if (a==gln) return 115.96;
  if (a==glu) return 115.98;
  if (a==gly) return 0;
  if (a==his) return 115.38;
  if (a==ile) return 118.17;
  if (a==leu) return 119.90;
  if (a==lys) return 115.73;
  if (a==met) return 115.79;
  if (a==phe) return 114.40;
  if (a==pro) return 123.58;
  if (a==ser) return 110.33;
  if (a==thr) return 111.67;
  if (a==trp) return 109.27;
  if (a==tyr) return 113.14;
  if (a==val) return 114.46;
  return 0;
}//centroid_chi2

real
Utilities::centroid_distance ( aminoacid a ) {
  if (a==ala) return 1.53;
  if (a==arg) return 3.78;
  if (a==asn) return 2.27;
  if (a==asp) return 2.24;
  if (a==cys) return 2.03;
  if (a==gln) return 2.85;
  if (a==glu) return 2.83;
  if (a==gly) return 0;
  if (a==his) return 3.01;
  if (a==ile) return 2.34;
  if (a==leu) return 2.62;
  if (a==lys) return 3.29;
  if (a==met) return 2.95;
  if (a==phe) return 3.41;
  if (a==pro) return 1.88;
  if (a==ser) return 1.71;
  if (a==thr) return 1.94;
  if (a==trp) return 3.87;
  if (a==tyr) return 3.56;
  if (a==val) return 1.97;
  return 0;
}//centroid_distance

int
Utilities::centroid_radius ( aminoacid a ) {
  if (a==ala) return 190;
  if (a==arg) return 280;
  if (a==asn) return 222;
  if (a==asp) return 219;
  if (a==cys) return 213;
  if (a==gln) return 241;
  if (a==glu) return 238;
  if (a==gly) return 120;
  if (a==his) return 249;
  if (a==ile) return 249;
  if (a==leu) return 249;
  if (a==lys) return 265;
  if (a==met) return 255;
  if (a==phe) return 273;
  if (a==pro) return 228;
  if (a==ser) return 192;
  if (a==thr) return 216;
  if (a==trp) return 299;
  if (a==tyr) return 276;
  if (a==val) return 228;
  return 100; // default
}//centroid_radius


