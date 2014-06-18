#ifndef FIASCO_ANCHOR_H
#define FIASCO_ANCHOR_H

#include "typedefs.h"
#include "utilities.h"
#include "mathematics.h"

#include <string.h>

class Fragment;
class VariableFragment;

class Anchor {
 private:
  bound C_coordinates;
  bound O_coordinates;
  bound N_coordinates;
  real phi, theta, psi;		/* Euler Angles relative to common reference */
  int variable_idx;		/* Variable idx associated to the [C' O'] of the 
				   current anchor (can be derived from var_fragment->idx) */
  int domain_elem_idx;		/* idx of the VFragment domain elment associated 
				   to fragment modelling this anchor */
  Fragment *_associated_fragment;
  VariableFragment* _associated_variable;

 public:
  Anchor() {};
  Anchor (const point &_C, const point& _O, const point &_N, 
	  //	  bool compute_euler_angles = true,
	  int var_idx=-1, Fragment* associated_fragment = NULL,
	  real _phi=0, real _theta=0, real _psi=0);
  Anchor (const point &_Cinf, const point& _Csup, 
	  const point &_Oinf, const point &_Osup, 
	  const point& _Ninf, const point &_Nsup, 
	  int var_idx=-1, Fragment* associated_fragment = NULL,
	  real _phi=0, real _theta=0, real _psi=0);
  ~Anchor() {};
  
  Anchor (const Anchor& other);
  Anchor& operator= (const Anchor& other);

  const point& get_C() const {return C_coordinates.first; }
  const point& get_O() const {return O_coordinates.first; }
  const point& get_N() const {return N_coordinates.first; }
  const bound& get_C_bounds () const {return C_coordinates; }
  const bound& get_O_bounds () const {return C_coordinates; }
  const bound& get_N_bounds () const {return C_coordinates; }
  real get_phi() const {return phi; }
  real get_theta() const {return theta; }
  real get_psi() const {return psi; }  
  void get_centroid (point& cg) const; 
  void set (const point& ptC, const point& ptO, const point& ptN);
  void set (const point& inf, const point& sup, atom_type atom);
  
  void set_associated_variable (int vidx);
  void set_associated_fragment (int vidx, int fidx);
  const Fragment* of_fragment () const {return _associated_fragment; }
  const VariableFragment* of_variable () const {return _associated_variable; }
  int associated_variable_idx () const {return variable_idx; }
  int associated_fragment_idx () const {return domain_elem_idx; }


  void get_normal (point& normal) const;
  Anchor& compute_euler_angles ();
  bool is_within_distance_of (const Anchor& other, real eps=0.0) const;
  bool is_within_bounds_of (const Anchor& other, real eps=0.0) const;
  bool is_within_orientation_of (const Anchor& other, real eps=0.0) const;
  bool is_degenerate();  
  void dump () const;
  
};//-

#endif
