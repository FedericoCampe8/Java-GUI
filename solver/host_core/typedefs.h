/*********************************************************************
 * This file lists the global definitions used by FIASCO.
 *********************************************************************/
#ifndef FIASCO_TYPEDEFS__
#define FIASCO_TYPEDEFS__

#include <iostream>
#include <string>
#include <vector>

#define STATISTICS
//#define VERBOSE
#define JM_PREPROCESS

typedef double real;
typedef char flag;
typedef unsigned int uint;
typedef unsigned long long usize_t;
typedef std::pair<long double, uint> real_exp; // <NUMBER, EXP>

typedef real vec3[3];
typedef real point[3];
typedef real plane[3][3];
typedef std::pair<point, point> bound;

typedef real T_MAT[4][4];	/* Translation matrix */
typedef real QUAT[4];	        /* Quaternion */
typedef real R_MAT[3][3];	/* Rot matrix for vectors */
//----------------------------------------------------------------

enum _euler_angles {PHI=0, THETA=1, PSI=2};
enum _exploring_dir {up=1, down=-1};
enum _var_type {bottom = 0, vfrag = 1, vpair = 2};
enum _thread_type {worker = 1, scheduler = 2};
enum atom_type {N, CA, CB, O, H, S, CG, X}; // X = generic atom
enum aminoacid {ala, arg, asn, asp, cys, gln, glu, gly, his, ile, leu, 
		lys, met, phe, pro, ser, thr, trp, tyr, val, err};
enum fragment_type {standard=21, special=22, helix=23, sheet=24};
enum constr_type { __c_alldist, __c_distGEQ, c_distLEQ, __c_fragment, 
		   __c_pair, __c_energy, __c_bundle, __c_centroid, 
		   __c_ellipsoid, __c_look_ahead, __c_loop, 
		   __c_end_anchor_distance, __c_end_anchor_orientation,
		   constr_type_size};
enum t_stats {t_search, t_first_sol, t_preprocess, t_jm, t_statistics, t_noop, t_table, t_stat_size};
enum prot_struct_type{protein, p_loop, p_helix, p_sheet, prot_struct_size};


// E. Clementi, D.L.Raimondi, W.P. Reinhardt (1967).
// "Atomic Screening Constants from SCF Functions. II. 
// Atoms with 37 to 86 Electrons". J. Chem. Phys. 47: 1300.
// ditance bound C-C	120 - 154
enum atom_radii {
  rH = 120,
  rC = 170,
  rO = 152,
  rN = 155,
  rS = 180,
  // http://en.wikipedia.org/wiki/Atomic_radii_of_the_elements_(data_page)
  /* rH = 37, */
  /* rC = 77, */
  /* rO = 73, */
  /* rS = 102, */
  /* rN = 75, */
  //
  rCG = 100,
  r_a = 240, 
  r_r = 323,
  r_n = 340,
  r_d = 403, 
  r_c = 426,
  r_q = 240,
  r_e = 404,
  r_g = 316,
  r_h = 441,
  r_i = 348,
  r_l = 412,
  r_k = 340,
  r_m = 270,
  r_f = 398,
  r_p = 280,
  r_s = 280,
  r_t = 501,
  r_w = 280,
  r_y = 483,
  r_v = 473};

// A. Bondi (1964). "van der Waals Volumes and Radii". 
// J. Phys. Chem. 68: 441.
enum atom_van_der_waals_radii {
  wH = 120, 
  wC = 170, 
  wO = 152, 
  wN = 155, 
  wS = 180, 
  wCG = 200};

// P. Pyykkö, M. Atsumi (2009). "Molecular Single-Bond 
// Covalent Radii for Elements 1-118". 
// Chemistry: A European Journal 15: 186–197.
enum atom_covalent_radii {
  cH = 31, 
  cC = 76, 
  cO = 66, 
  cS = 105,
  cN = 71,
  cCG = 100};

// phi and psi clutering angle definitions.
// fragments with initial phi or terminal psi angle will be clustered in the 
// same bucket.
//50 0.872664626 //30  0.471238898 //15 0.261799388 //10 0.174532925
#define _pi 3.1415926535897932384626433832795
#define phi_cluster_angle 0.471238898 
#define psi_cluster_angle 0.471238898 
#define dist_C_O 1.23
#define dist_C_N 1.33
#define dist_N_Ca 1.46
#define dist_Ca_C 1.51
#define dist_epsilon 0.30
#define MAX_TIMEOUT 1000000000

#endif
