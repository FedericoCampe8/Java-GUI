#ifndef COCOS_TYPEDEFS__
#define COCOS_TYPEDEFS__

#include <stdio.h>
#include <iostream>
#include <string>

#define TIME_STATS

#define MAX_DIM 32
#define MAX_DOM_SIZE 500
#define MAX_TARGET_SIZE 350
#define MAX_ENERGY 1000
#define MAX_RMSD 1000
#define MAX_GIBBS_SET_SIZE 1000
#define MAX_BOOL_SIZE ((MAX_GIBBS_SET_SIZE/32)+1)
#define MAX_SAMPLE_SET_SIZE max( MAX_GIBBS_SET_SIZE, MAX_DOM_SIZE )
#define MAX_N_THREADS 1024
#define LOWER_PRIORITY 100
#define MAX_QUANTUM 86400
#define PI_VAL 3.1415926535897
#define CLOSE_TO_ZERO_VAL 0.00001

// Bit operations (on uint) and Utilities
#define ISBITSET(x,i) ((x[i>>5] & (1<<(i%32)))!=0)
#define SETBIT(x,i) x[i>>5]|=(1<<i%32)
#define CLEARBIT(x,i) x[i>>5]&= ~(1<<(i%32))
#define WHICHWARP(x) x>>5 
#define WHICHBIT(x)  x%32

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))
#define CURAND_CALL( x ) do { if((x)!=CURAND_STATUS_SUCCESS) { printf("Error at %s:%d\n",__FILE__,__LINE__); exit( 2 );}} while( 0 )

static void HandleError( cudaError_t err, const char *file, int line ) {
  if (err != cudaSuccess) {
    printf( "Error %d: %s in %s at line %d\n", err, cudaGetErrorString( err ), file, line );
    exit( 2 );
  }
}//HandleError

/* Using float reals we loose precision */
typedef float real;
//typedef double real;
typedef unsigned int uint;

typedef real vec3[3];
typedef real point[3];
typedef real R_MAT[3][3];

enum agent_type    { supervisor, coordinator, structure, worker };
enum ss_type       { helix, pi_helix, g_helix, turn, coil, sheet, other };
enum search_type   { icm, gibbs, montecarlo, complete, c_type_search };
enum atom_type     { N, CA, CB, O, H, S, CG, X };
enum aminoacid     { ala, arg, asn,
                     asp, cys, gln,
                     glu, gly, his,
                     ile, leu, lys,
                     met, phe, pro,
                     ser, thr, trp,
                     tyr, val, err
};

enum constr_type   { c_sang,
                     c_k_angle_shuffle,
                     c_all_dist,
                     c_k_rang,
                     c_cg,
                     c_mang,
                     c_dist,
                     c_type_size
};

enum constr_events { sing_event,
                     dmc_event,
                     all_events,
                     empty_event,
                     failed_event,
                     fix_prop,
                     single_prop,
                     events_size
};
enum energy_fields { f_hydrogen, f_contact, f_correlation, en_fields_size };
enum t_stats       { t_search, t_cuda, t_all_distant, t_cg, t_fragment_prop,
                     t_energy, t_rmsd, t_stat_size };

/*
 E. Clementi, D.L.Raimondi, W.P. Reinhardt (1967).
 "Atomic Screening Constants from SCF Functions. II.
 Atoms with 37 to 86 Electrons". J. Chem. Phys. 47: 1300.
 */
enum atom_radii    {
  rH = 120,
  rC = 170,
  rO = 152,
  rN = 155,
  rS = 180,
  rCG = 100,
  r_a = 190,
  r_r = 280,
  r_n = 222,
  r_d = 219,
  r_c = 213,
  r_q = 241,
  r_e = 238,
  r_g = 120,
  r_h = 249,
  r_i = 249,
  r_l = 249,
  r_k = 265,
  r_m = 255,
  r_f = 273,
  r_p = 228,
  r_s = 192,
  r_t = 216,
  r_w = 299,
  r_y = 276,
  r_v = 228
};






//#define INF_ENERGY 10000
//#define MAX_ATOMS MAX_TARGET_SIZE*4
//#define MAX_MEMORY 500000000 //Bytes
//#define MAX_NUM_STR 1000     // max = 65536



//#define NITROGEN_ATOM 0
//#define CALPHA_ATOM   1
//#define CBETA_ATOM    2
//#define OXYGEN_ATOM   3
//#define CARBON_ATOM   4
//#define SULPHUR_ATOM  5
//#define CENTROID_ATOM 6
//#define HYDROGEN_ATOM 7
//#define OTHER_ATOM    8

//#define dist_C_O  1.23
//#define dist_C_N  1.33
//#define dist_N_Ca 1.46
//#define dist_Ca_C 1.51



// Types of Secondary Structures
//#define HELIX    0
//#define PI_HELIX 1
//#define G_HELIX  2
//#define TURN     3
//#define COIL     4
//#define SHEET    5
//#define OTHER    6





//#define W_H 0
//#define W_CORR 1
//#define W_CON 2
//
//#define ca_up true
//#define ca_down false

//#define CUDA_STATISTICS






//typedef char flag;
//typedef real planes[3][3];
//
//typedef real T_MAT[4][4];	/* Translation matrix */
//typedef real QUAT[4];	    /* Quaternion */
//









//typedef struct {
//  int worker_agent_id;
//  int worker_agent_label;
//  real energy;
//  std::vector<std::vector<real> > structure;
//} seen_structures;


//enum fragment_type    { standard2 = 21, special2 = 22, helix2 = 23, sheet2 = 24, turn2 = 25, coil2 = 26 };
//enum fragment_type_seq { first, middle, last, refinement };
//enum plane { left_plane, right_plane, both };
////enum energy_fields { lenard_jones, h_distances_par, h_angles_par, contact, coulomb, torsional };
////enum constr_type { __c_alldist, __c_fragment, __c_angles_relation, __c_hbond, __c_cg, constr_type_size };
//
//enum trail_type {
//  _trail_pt, 
//  _trail_fr, 
//  _trail_cg,
//  _trail_en, 
//  _trail_en_cca, 
//  _trail_en_ccg, 
//  _trail_en_ori, 
//  _trail_en_tor, 
//  _trail_en_cor,
//  _trail_constr,
//  _trail_consistency,
//  _trail_grid_pt,
//  _trail_grid_pt_h
//};
//
//enum prot_struct_type{ protein, p_loop, p_helix, p_sheet, prot_struct_size };


/* A. Bondi (1964). "van der Waals Volumes and Radii". 
   J. Phys. Chem. 68: 441.
*/
enum atom_van_der_waals_radii {
  wH = 120, 
  wC = 170, 
  wO = 152, 
  wN = 155, 
  wS = 180, 
  wCG = 200};

/* P. Pyykkö, M. Atsumi (2009). "Molecular Single-Bond 
   Covalent Radii for Elements 1-118". 
   Chemistry: A European Journal 15: 186–197.
 */
enum atom_covalent_radii {
  cH = 31, 
  cC = 76, 
  cO = 66, 
  cS = 105,
  cN = 71,
  cCG = 100};

#endif
