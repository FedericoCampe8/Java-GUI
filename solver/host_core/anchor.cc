#include "anchor.h"
#include "typedefs.h"
#include "globals.h"
#include "logic_variables.h"
#include "variable_fragment.h"
#include "utilities.h"
#include "mathematics.h"
#include "fragment.h"

#include <iostream>
#include <string>
#include <cmath>

using namespace std;

Anchor::Anchor(const point &_C, const point& _O, const point &_N,
	       int vidx, Fragment* associated_fragment, 
	       real _phi, real _theta, real _psi) 
  : phi (_phi), theta (_theta), psi (_psi), variable_idx (vidx), 
    _associated_fragment (associated_fragment) {
  set (_C, _C, CB);
  set (_O, _O, O);
  set (_N, _N, N);
  if (phi == 0 && theta == 0 && psi == 0)
    compute_euler_angles();
  if (vidx >= 0)
    _associated_variable = &(g_logicvars->var_fragment_list[vidx]);
  else if (_associated_fragment)
    _associated_variable = const_cast<VariableFragment*>
                           (_associated_fragment->of_variable());

}
//-

Anchor::Anchor(const point &_Cinf, const point& _Csup, 
	       const point &_Oinf, const point &_Osup, 
	       const point& _Ninf, const point &_Nsup, 
	       int vidx, Fragment* associated_fragment,
	       real _phi, real _theta, real _psi)  
  : phi (_phi), theta (_theta), psi (_psi), variable_idx (vidx), 
    _associated_fragment (associated_fragment) {
  set (_Cinf, _Csup, CB);
  set (_Oinf, _Osup, O);
  set (_Ninf, _Nsup, N);
  
  if (phi == 0 && theta == 0 && psi == 0)  
    compute_euler_angles();
  if (vidx >= 0)
    _associated_variable = &(g_logicvars->var_fragment_list[vidx]);
  else if (_associated_fragment)
    _associated_variable = const_cast<VariableFragment*> 
                           (_associated_fragment->of_variable());
}
//-

// Anchor::~Anchor () {
//   //  delete[] array;
// }

Anchor::Anchor (const Anchor& other) {
  set (other.C_coordinates.first, other.C_coordinates.second, CB);
  set (other.O_coordinates.first, other.O_coordinates.second, O);
  set (other.N_coordinates.first, other.N_coordinates.second, N);
  phi = other.phi;
  theta = other.theta;
  psi = other.psi;

  variable_idx = other.variable_idx;
  domain_elem_idx = other.domain_elem_idx;
  _associated_fragment = other._associated_fragment;
  _associated_variable = other._associated_variable;
}
//-

Anchor&
Anchor::operator= (const Anchor& other) {
  if( this != &other) {
    set (other.C_coordinates.first, other.C_coordinates.second, CB);
    set (other.O_coordinates.first, other.O_coordinates.second, O);
    set (other.N_coordinates.first, other.N_coordinates.second, N);
    phi = other.phi;
    theta = other.theta;
    psi = other.psi;
    
    variable_idx = other.variable_idx;
    domain_elem_idx = other.domain_elem_idx;
    _associated_fragment = other._associated_fragment;
    _associated_variable = other._associated_variable;
  }
  return *this;
}
//-

void 
Anchor::set (const point& inf, const point& sup, atom_type atom) {
  if(atom == (atom_type)N) {
    memcpy (N_coordinates.first, inf, sizeof(point));
    memcpy (N_coordinates.second, sup, sizeof(point));
  }
  else if (atom == (atom_type)CB) {
    memcpy (C_coordinates.first, inf, sizeof(point));
    memcpy (C_coordinates.second, sup, sizeof(point));
  }
  else if (atom == (atom_type)O) {
    memcpy (O_coordinates.first, inf, sizeof(point));
    memcpy (O_coordinates.second, sup, sizeof(point));
  }
  compute_euler_angles();
}
//-


void 
Anchor::set (const point& ptC, const point& ptO, const point& ptN) {
  string dbg = "Anchor::set(3 points) - ";
  set(ptC, ptC, CB);
  set(ptN, ptN, N);
  set(ptO, ptO, O);
}
//-

void 
Anchor::set_associated_variable (int vidx) {
  variable_idx = vidx; 
  _associated_variable = &(g_logicvars->var_fragment_list[vidx]);
}
//-

void 
Anchor::set_associated_fragment (int vidx, int fidx) {
  variable_idx = vidx;   
  domain_elem_idx = fidx;
  _associated_variable = &(g_logicvars->var_fragment_list[vidx]);
  _associated_fragment = &(g_logicvars->var_fragment_list[vidx].domain[fidx]);
}
//-

// here we asume points are fixed. 
void
Anchor::get_centroid (point& cg) const {
  cg[0] = (C_coordinates.first[0] + O_coordinates.first[0] + N_coordinates.first[0]) / 3;
  cg[1] = (C_coordinates.first[1] + O_coordinates.first[1] + N_coordinates.first[1]) / 3;
  cg[2] = (C_coordinates.first[2] + O_coordinates.first[2] + N_coordinates.first[2]) / 3;
}
//-

Anchor&//void 
Anchor::compute_euler_angles() {
  point pC, pO, pN;
  if (this->is_degenerate()) {
    memcpy (pC, C_coordinates.first, sizeof(point));
    memcpy (pO, O_coordinates.first, sizeof(point));
    memcpy (pN, N_coordinates.first, sizeof(point));
  }
  else {
    for (int c=0; c<3; c++) {
      pC[c] = (C_coordinates.second[c] - C_coordinates.first[c])/2;
      pO[c] = (O_coordinates.second[c] - O_coordinates.first[c])/2;
      pN[c] = (N_coordinates.second[c] - N_coordinates.first[c])/2;
    }
  }

  vec3 u, v, normal;
  Math::vsub (pO, pC, u);
  Math::vsub (pN, pC, v);
  Math::vcross (u, v, normal);
  Math::round (normal);
  
  vec3 *n1 = &normal;
  vec3 *n2 = &g_reference_system.normal_xy;    

  phi   = acos (Math::abs (Math::vdot (*n1, *n2)) / 
		Math::abs ((Math::vnorm2 (*n1) * Math::vnorm2 (*n2)) ));
  
  n2 = &g_reference_system.normal_xz;
  theta = acos (Math::abs (Math::vdot (*n1, *n2)) / 
		Math::abs ((Math::vnorm2 (*n1) * Math::vnorm2 (*n2)) ));
  
  n2 = &g_reference_system.normal_yz;
  psi   = acos (Math::abs (Math::vdot (*n1, *n2)) / 
		Math::abs ((Math::vnorm2 (*n1) * Math::vnorm2 (*n2)) ));

  return *this;
}
//-

// This routine assumes the points are fixed (no bounds)
void 
Anchor::get_normal (point& normal) const {
  vec3 O_N, C_O;
  Math::vsub (get_O(), get_N(), O_N);
  Math::vsub (get_C(), get_O(), C_O);
  Math::vcross (O_N, C_O, normal);
}
//-

bool
Anchor::is_within_distance_of (const Anchor& other, real eps) const {
  return ( Math::in_range (get_C(), other.get_C(), eps) &&
	   Math::in_range (get_O(), other.get_O(), eps) &&
	   Math::in_range (get_N(), other.get_N(), eps));
}
//-

bool 
Anchor::is_within_bounds_of (const Anchor& other, real eps) const {
  return ( Math::in_range (get_C(), other.get_C_bounds(), eps) &&
	   Math::in_range (get_O(), other.get_O_bounds(), eps) &&
	   Math::in_range (get_N(), other.get_N_bounds(), eps));
}
//-

bool
Anchor::is_within_orientation_of (const Anchor& other, real eps) const {
  return (Math::in_range 
	  (get_phi(), other.get_phi()-eps, other.get_phi()+eps)       &&
	  Math::in_range 
	  (get_theta(), other.get_theta()-eps, other.get_theta()+eps) &&
	  Math::in_range 
	  (get_psi(), other.get_psi()-eps, other.get_psi()+eps));
}
//-

bool
Anchor::is_degenerate() {
  for (uint i = 0; i < 3; i++) {
    if ((C_coordinates.first[i] != C_coordinates.second[i]) ||
	(O_coordinates.first[i] != O_coordinates.second[i]) ||
	(N_coordinates.first[i] != N_coordinates.second[i]))
      return false;
  }
  return true;
}
//-

void 
Anchor::dump () const {
  std::cout << "Associated element: D(Var-F_" << variable_idx 
	    << ") = "<< domain_elem_idx << std:: endl;
  Utilities::dump (C_coordinates.first, C_coordinates.second, "C: ");
  Utilities::dump (O_coordinates.first, O_coordinates.second, "O: ");
  Utilities::dump (N_coordinates.first, N_coordinates.second, "N: ");
  std::cout << "Euler Angles: <" << phi << ", " << theta << ", " << psi << ">" << std::endl;
  if (_associated_fragment)
    cout << "Fragment Probability: " << _associated_fragment->get_probability() << std::endl;
}
//-

