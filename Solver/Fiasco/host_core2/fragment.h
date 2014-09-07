/*********************************************************************
 * Fragment
 *********************************************************************/
#ifndef FIASCO_FRAGMENT__
#define FIASCO_FRAGMENT__

#include "typedefs.h"
#include "anchor.h"
#include "atom.h"

#include <string>
#include <vector>

enum AssemblyDirection {RIGHT_TO_LEFT, LEFT_TO_RIGHT};
class Atom;
class Anchor;
class VariableFragment;

/* A Fragment as defined in the FA model.
 * note: The backbone chain for the Fragment reprentation is described
 * by the following regular expression:  C' O N Ca C' O (N Ca C' O)* N
 * @note: If n is the number of amino acid of a fragment, its full atom
 * description contains 4n+3 atoms.
 * Atoms are univocally determined by the index in the #bb# array.
 */
class Fragment {
 private:
  int id;
  fragment_type type;
  std::string pid;     // Protein ID
  uint offset;
  /* note: start and end are included */
  uint aa_start;       // first frag aa. idx in the sequence
  uint aa_end;	       // last frag aa.  idx in the sequence 
  uint bb_start;       // Frag first atom mapped in the target seq.
  uint bb_end;	       // Frag last atom mapped in the target seq.

  int frequency;       // Occurrence rate from clustering process
  real probability;    // Fragment Probability occurrence value
  real psi, phi;

  Anchor* _front_anchor;
  Anchor* _end_anchor;
  const VariableFragment* _of_variable; // link to the associated Fragment variable

 public:
  R_MAT rot_m;
  vec3  shift_v;  

  std::vector<aminoacid> aa_seq; // only one between this and
  std::vector<Atom> backbone;
  std::vector<Atom> sidechains;
  std::vector<Atom> centroid;

  Fragment();  
  Fragment (int _id, fragment_type _type, std::string prot_id, uint _offset,
	    uint _aa_start, uint _aa_end, uint _bb_start, uint _bb_end,
	    int _frequency, real _probability, std::vector<aminoacid> _aa_seq,
	    std::vector<Atom> _backbone, VariableFragment* variable = NULL);
  ~Fragment();
  Fragment (const Fragment& other);
  Fragment& operator= (const Fragment& other);
  
  bool operator() (const Fragment& fi, const Fragment& fj) 
    {return ( fi.probability > fj.probability); }

  int get_id() const {return id;}
  void set_id(const int k) {id = k; }
  std::string get_pid() const {return pid; }
  void set_pid(const std::string s) {pid = s; }
  fragment_type get_type() const {return type; }
  void set_type(const fragment_type t) {type = t; }
  uint get_offset() const {return offset; }
  void set_offset(const int o) {offset = o; }
  int get_frequency () const {return frequency; }
  void set_frequency (const int f) {frequency = f; }
  real get_probability () const {return probability; }
  void set_probability (const real p) {probability = p; }
  uint get_aa_s() const {return aa_start; }
  void set_aa_s(const int a) {aa_start = a; }
  uint get_aa_e() const {return aa_end; }
  void set_aa_e(const int a) {aa_end=a; }
  uint get_bb_s() const {return bb_start; }
  void set_bb_s(const int a) {bb_start = a; }
  uint get_bb_e() const {return bb_end; }
  void set_bb_e(const int a) {bb_end = a; }
  bool check_steric_clashes ();
  
  uint ncentroids() const {return centroid.size();}
  uint backbone_len() const {return backbone.size();}
  uint nres() const;

  void compute_phi();
  void compute_psi();
  real get_phi() const {return phi;}
  real get_psi() const {return psi;}

  void set_front_anchor();
  void set_end_anchor();
  Anchor* front_anchor() const  {return _front_anchor; }
  Anchor* end_anchor() const    {return _end_anchor; }
  const VariableFragment* of_variable() const {return _of_variable; }

  // utilities
  void copy_rot_mat(const Fragment& other);
  void copy_sh_vec(const Fragment& other);
  void dump ();

  // geometry
  void transform_aux_elements ();
  void compute_normal_base (int offset = 0);
  void change_coordinate_system ();
  void overlap (const point& p1, const point& p2, const point &p3, const Fragment& f, int offset = 2);   // DEPRECATED
  void overlap (const Anchor& hook, AssemblyDirection /*growing_chain_*/direction = LEFT_TO_RIGHT);
  void overlap (const Fragment& hook, AssemblyDirection /*growing_chain_*/direction = LEFT_TO_RIGHT);

  void transform (const R_MAT& R, const vec3& v);

};


#endif
