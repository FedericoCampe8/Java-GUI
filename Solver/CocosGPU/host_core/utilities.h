#ifndef COCOS_UTILITIES__
#define COCOS_UTILITIES__

#include "globals.h"

class Atom;
namespace Utilities{
  /***************************************
   *  Input and set constraints options  *
   ***************************************/
  void set_search_labeling_strategies ();
  void set_all_distant_constraint ();
  void set_distance_constraint ();
  void set_centroid_constraint ();
  /***************************************
   *           Conversion tools          *
   ***************************************/
  std::string cv_string_to_str_type ( ss_type t );
  ss_type cv_string_to_str_type ( std::string );
  std::string cv_aa1_to_aa3 ( char a );
  std::string cv_aa3_to_aa1 ( std::string a );
  std::string cv_class_to_aa3 ( aminoacid a );
  aminoacid cv_aa_to_class ( char a );
  aminoacid cv_aa_to_class ( std::string a );
  int cv_class_to_n ( aminoacid a );
  atom_type get_atom_type( std::string name );
  
  /***************************************
   *      Offsets and Atom postions      *
   ***************************************/
  atom_type get_atom_type ( uint bbidx );
  atom_radii get_atom_radii ( uint bbidx );
  int get_aaidx_from_bbidx ( uint bbidx, atom_type type );
  int get_bbidx_from_aaidx ( uint aaidx, atom_type type );
  void calculate_aa_points( bool dir, real bb[] );
  
  /***************************************
   *        Overalp and Rotations        *
   ***************************************/
  void overlap_structures ( point& pa, point& pb, point& pc,
                           point * str_out,
                           int len=9 );
  void overlap_structures ( point& pa, point& pb, point& pc,
                           point * str_in, point * str_out,
                           int len=9, int offset=0 );
  void overlap_structures ( point * str_left, point * str_right, int aa_idx );
  void compute_normal_base ( point * backbone, real rot_m[3][3], real shift_v[3] );
  void change_coordinate_system ( point * backbone, real rot_m[3][3], real shift_v[3], int len=9 );
  void overlap ( point& p1, point& p2, point& p3, point * backbone, int len=9, int offset=3 );
  
  /***************************************
   *          I/O aux functions          *
   ***************************************/
  void output_pdb_format ( std::string, const std::vector<Atom>& );
  std::string output_pdb_format ( point* structure, int len=0, real rmsd=0 );
  std::string output_pdb_format ( real* structure, real rmsd=0 );
  
  int  get_format_digits ( real );
  std::string get_format_spaces ( real );
  
  /***************************************
   *               Display               *
   ***************************************/
  void print_debug ( std::string s );
  void print_debug ( std::string s1, std::string s2 );
  /*
   template <class T> void print_debug ( std::string s, T param );
   template <class T> void print_debug ( std::string s1, std::string s2, T param );
   */
  void dump ( real* p );           // Display a point
  void dump ( real* p1, real* p2 );    // Display interval
  void dump ( std::vector< std::vector <real> > str );
  
  void rotate_point_about_line( real* in_point, real theta_rad,
                               real* p1, real* p2,
                               real* out_point );
  void move_phi ( real * aa_points, real degree, int v_id, int ca_pos, int first_res, int threadIdx );
  void move_psi ( real * aa_points, real degree, int v_id, int ca_pos, int last_res, int threadIdx ) ;
  void copy_structure_from_to ( real* s1, real* s2, int nthreads);
  
  /// Other
  void calculate_cg_atom ( aminoacid a,
                          real* ca1, real* ca2, real* ca3,
                          real* cg, int* radius );
  real centroid_torsional_angle ( aminoacid a );
  real centroid_chi2 ( aminoacid a );
  real centroid_distance ( aminoacid a );
  int centroid_radius ( aminoacid a );
}//-

#endif
