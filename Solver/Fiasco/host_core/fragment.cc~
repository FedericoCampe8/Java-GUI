#include "typedefs.h"
#include "fragment.h"
#include "mathematics.h"
#include "utilities.h"
#include "anchor.h"

#include <iostream>
#include <cmath>

using namespace std;

Fragment::Fragment()
  : id (-1), type (standard), pid (""), offset (0), 
    aa_start (0), aa_end (0), bb_start (0), bb_end (0),
    frequency (0), probability (0), psi (0), phi (0),
    _front_anchor (NULL), _end_anchor (NULL), _of_variable (NULL) {
}//-

Fragment::Fragment
  (int _id, fragment_type _type, string prot_id, uint _offset,
   uint _aa_start, uint _aa_end, uint _bb_start, uint _bb_end,
   int _frequency, real _probability, vector<aminoacid> _aa_seq,
   vector<Atom> _backbone, VariableFragment* variable) :
    id (_id), 
    type (_type), 
    pid (prot_id), 
    offset (_offset),
    aa_start (_aa_start),
    aa_end   (_aa_end),
    bb_start (_bb_start),
    bb_end   (_bb_end),
    frequency (_frequency),
    probability (_probability),
    _front_anchor (NULL),
    _end_anchor (NULL),
    _of_variable (variable),
    aa_seq (_aa_seq),
    backbone (_backbone) {
  Math::set_identity(rot_m);
  Math::set_identity(shift_v); 
  transform_aux_elements();
}
//-

Fragment::~Fragment () {
  string dbg = "Fragment::~Fragment() - ";
  //  if (front_anchor())
    delete _front_anchor;
    // if (end_anchor())
    delete _end_anchor;
}
//-

Fragment::Fragment (const Fragment& other) {
  string dbg = "Fragment::Fragment() - ";
  id   = other.get_id();
  type = other.get_type();
  pid  = other.get_pid();
  aa_start = other.aa_start;
  aa_end   = other.aa_end;
  bb_start = other.bb_start;
  bb_end   = other.bb_end;
  offset   = other.get_offset();
  frequency   = other.get_frequency();
  probability = other.get_probability();
  phi = other.phi;
  psi = other.psi;
  
  aa_seq     = other.aa_seq;
  backbone   = other.backbone;
  sidechains = other.sidechains;
  centroid   = other.centroid;
  
  copy_sh_vec (other);
  copy_rot_mat (other);
  _front_anchor = NULL;
  _end_anchor = NULL;
  set_front_anchor ();
  set_end_anchor ();
  _of_variable = other._of_variable;
}
//-

Fragment&
Fragment::operator= (const Fragment& other) {
  if (this != &other) {
    id   = other.get_id();
    type = other.get_type();
    pid  = other.get_pid();
    aa_start = other.aa_start;
    aa_end   = other.aa_end;
    bb_start = other.bb_start;
    bb_end   = other.bb_end;
    offset   = other.get_offset();
    frequency   = other.get_frequency();
    probability = other.get_probability();
    phi = other.phi;
    psi = other.psi;
    
    aa_seq     = other.aa_seq;
    backbone   = other.backbone;
    sidechains = other.sidechains;
    centroid   = other.centroid;
    
    copy_sh_vec (other);
    copy_rot_mat (other);
    
    set_front_anchor ();
    set_end_anchor ();
    _of_variable = other._of_variable;
  }
  return *this;
}
//-

uint
Fragment::nres() const {
  uint nres = 0;
  for (uint i=0; i<backbone_len(); i++) {
    nres = backbone.at(i).is_type(CA) ? nres+1 : nres;
  }
  return nres;
}
//-

void 
Fragment::set_front_anchor() {
  if (!backbone.empty()) {
    if (_front_anchor) {
      _front_anchor->set (backbone[0].position, 
			  backbone[1].position,
			  backbone[2].position);
    }
    else { 
      _front_anchor = 
	new Anchor (backbone[0].position, 
		    backbone[1].position, 
		    backbone[2].position, -1, this);
    }
  }
}
//-

void 
Fragment::set_end_anchor() {
  size_t bb_len = backbone.size();
  if (!backbone.empty()) {
    if (_end_anchor) {
      _end_anchor->set (backbone[bb_len-3].position, 
			backbone[bb_len-2].position, 
			backbone[bb_len-1].position);
    }
    else {
      _end_anchor = new Anchor (backbone[bb_len-3].position,  
				backbone[bb_len-2].position, 
				backbone[bb_len-1].position, -1, this);
    }
  }
}
//-


// modify anchor positions
void 
Fragment::transform_aux_elements () {
  // Transform end anchors
  set_front_anchor();
  set_end_anchor();
  // compute_phi();
  // compute_psi();
} 
//-


bool 
Fragment::check_steric_clashes () {
  real d = 0;
  for (uint i = 0; i < backbone_len()-1; i++) {
    // dist C'-O
    if (backbone[i].type == CB && backbone[i+1].type == O) {
      d = Math::eucl_dist (backbone[i].position, backbone[i+1].position);
      if ( d > (dist_C_O + dist_epsilon)) {
	return false;
      }
    }
    // dist C'-N
    if (i < backbone_len()-2 &&
	backbone[i].type == CB && backbone[i+2].type == N) {
      d = Math::eucl_dist (backbone[i].position, backbone[i+2].position);
      if ( d > (dist_C_N + dist_epsilon)) {
	return false;
      }
    }
    // dist N-CA
    if (backbone[i].type == N && backbone[i+1].type == CA) {
      d = Math::eucl_dist(backbone[i].position, backbone[i+1].position);
      if ( d > (dist_N_Ca + dist_epsilon)) {
	return false;
      }
    }
    // dist CA-C'
    if (backbone[i].type == CA && backbone[i+1].type == CB) {
      d = Math::eucl_dist(backbone[i].position, backbone[i+1].position);
      if ( d > (dist_Ca_C + dist_epsilon)) {
	return false;
      }
    }
  }
  return true;
}//-



// Build the orthonormal (right) rotation matrix for a given Fragment.
void
Fragment::compute_normal_base (int offset) {
  vec3 x, y, z, v;

  // Build the plane for rotation 
  for (int i = 0; i < 3; i++) {
    v[i] = backbone.at(offset+1)[i] - backbone.at(offset)[i];
    z[i] = backbone.at(offset+2)[i] - backbone.at(offset+1)[i];
  }
  // Build the Orthogonal Base
  Math::vcross (z, v, y); 	// y orthogonal to z, v
  Math::vcross (y, z, x);	// x orthogonal to z and y
  // Normalize: Obtain the Orthonormal Base
  Math::vnorm (x);
  Math::vnorm (y);
  Math::vnorm (z);
  
  shift_v[0]=shift_v[1]=shift_v[2]=0;
  // Build the Rotation matrix (orthonormal)
  for (int i = 0; i < 3; i++) {
    rot_m[i][0] = x[i];
    rot_m[i][1] = y[i];
    rot_m[i][2] = z[i];
    shift_v[i]  -= backbone.at(offset)[i]; 
  }
}
//-

/* 
 * Normalize a Fragment f, given in an orhonormal base (f.R).
 * Compute R^-1 x (f - s0), where s0 is the translation vector to move
 * f first atom to <0,0,0> .
 * note: The inverse of the  R is its transposte (R orthonormal).
 * require: Global variable real tt[3], real sh[3]
 */
void
Fragment::change_coordinate_system () {
  real tt[3*backbone_len()];
  for (uint i = 0; i < backbone_len(); i++) {
    tt[3*i + 0] = backbone.at(i)[0];
    tt[3*i + 1] = backbone.at(i)[1];
    tt[3*i + 2] = backbone.at(i)[2];
  }

  // Translate the vector in 0,0,0 : tt = (f - s0)
  for (uint i = 0; i < backbone_len(); i++){
    tt[3*i + 0] += shift_v[0];
    tt[3*i + 1] += shift_v[1];
    tt[3*i + 2] += shift_v[2];
  } 

  // Rotate to the transport of R the fragment
  for (uint i = 0; i < backbone_len(); i++){
    for (uint j = 0; j < 3; j++){
      // R^-1 * tt (see note)
      backbone.at(i).position[j] =
      	rot_m[0][j] * tt[3*i]     +
      	rot_m[1][j] * tt[3*i + 1] +
      	rot_m[2][j] * tt[3*i + 2];
      if (abs(backbone.at(i).position[j]) < 1.0e-4) 
	backbone.at(i).position[j] = 0;
    }
  }  
 
  transform_aux_elements ();

}
//-

// OFFSET
// 0 -- 1      off%4+ 1
// 1 -- 2
// 2 -- 3
// 3 .// ?
// 4 -- 1      off%4 + 1
// 5 -- 2
// 6 -- 3

// DEPRECATED!! 
void
Fragment::overlap(const point& p1, const point& p2, const point &p3, 
		  const Fragment& f, int offset) {  
  vec3 x, y, z, v;
  // Build rotation plane
  Math::vsub(p2, p1, v);  // v = p2 - p1
  Math::vsub(p3, p2, z);  // z = p3 - p2
  // Build the Orthogonal Base */
  Math::vcross (z, v, y); // y orthogonal to z, v 
  Math::vcross (y, z, x); // x orthogonal to z and y
  // Normalize: Obtain the Orthonormal Base
  Math::vnorm (x);
  Math::vnorm (y);
  Math::vnorm (z);

  // Build the Rotation matrix (orthonormal)
  for (uint i = 0; i < 3; i++) {
    rot_m[i][0] = x[i];  
    rot_m[i][1] = y[i]; 
    rot_m[i][2] = z[i]; 
  }

  // f_t = RotMat(<p1,p3,p3>) x f 
  for (uint i = 0; i < f.backbone_len(); i++) {
    real px = rot_m[0][0] * f.backbone[i][0] +   
              rot_m[0][1] * f.backbone[i][1] +   
              rot_m[0][2] * f.backbone[i][2];   
    real py = rot_m[1][0] * f.backbone[i][0] +
              rot_m[1][1] * f.backbone[i][1] +
              rot_m[1][2] * f.backbone[i][2];   
    real pz = rot_m[2][0] * f.backbone[i][0] +
              rot_m[2][1] * f.backbone[i][1] +  
              rot_m[2][2] * f.backbone[i][2];  
    backbone[i].set_position(px,py,pz);
  }

  // c_t = RotMat(F1) x c
  for (uint i = 0; i < f.ncentroids(); i++) {
    real px = rot_m[0][0] * f.centroid[i].position[0] +   
              rot_m[0][1] * f.centroid[i].position[1] +   
              rot_m[0][2] * f.centroid[i].position[2];   
    real py = rot_m[1][0] * f.centroid[i].position[0] +
              rot_m[1][1] * f.centroid[i].position[1] +        
              rot_m[1][2] * f.centroid[i].position[2];   
    real pz = rot_m[2][0] * f.centroid[i].position[0] +
              rot_m[2][1] * f.centroid[i].position[1] +  
              rot_m[2][2] * f.centroid[i].position[2]; 
    centroid[i].set_position (px,py,pz);
  }
    
  // Set atom type and radius
  for (uint i = 0; i < f.backbone_len(); i++) {
    backbone[i].set_type(f.backbone[i].type);
  }

  // Translate the fragment so that it overlaps f on the plan <p1,p2,p3>
  // (superimposition on a3 and the third point of f) 
  
  // Set the traslation vector
  shift_v[0] = shift_v[1] = shift_v[2] = 0;
  point p;
  if ((offset % 4 + 1) == 1)
    memcpy(p, p1, sizeof(point));
  if ((offset % 4 + 1) == 2)
    memcpy(p, p2, sizeof(point));
  if ((offset % 4 + 1) == 3)
    memcpy(p, p3, sizeof(point));
  
  shift_v[0] = -backbone[offset][0] + p[0];
  shift_v[1] = -backbone[offset][1] + p[1];
  shift_v[2] = -backbone[offset][2] + p[2];
   
  for (uint i = 0; i < f.backbone_len(); i++) {
    backbone[i].position[0] += shift_v[0];
    backbone[i].position[1] += shift_v[1];
    backbone[i].position[2] += shift_v[2];
  }
  
  for (uint i = 0; i < f.ncentroids(); i++) {
    centroid[i].position[0] += shift_v[0];
    centroid[i].position[1] += shift_v[1];
    centroid[i].position[2] += shift_v[2];
  }
  
  transform_aux_elements (); 
}//-

void 
Fragment::overlap (const Fragment& hook, AssemblyDirection /*growing_chain_*/direction) {
  if (direction == LEFT_TO_RIGHT)
    overlap (*hook.end_anchor(), direction);
  else
    overlap (*hook.front_anchor(), direction);
}//-

void
Fragment::overlap (const Anchor& hook, AssemblyDirection /*growing_chain_*/direction) {

  //int first_CB = 0;
  int last_CB = backbone.size();
  while (!backbone[--last_CB].is_type(CB)) {; }

  if (direction == LEFT_TO_RIGHT) ;
    //   compute_normal_base (first_CB);
  else {
    compute_normal_base (last_CB);
    change_coordinate_system();
  }
  vec3 x, y, z, v;
  // Build rotation plane
  Math::vsub(hook.get_O(), hook.get_C(), v);
  Math::vsub(hook.get_N(), hook.get_O(), z);
  // Build the Orthogonal Base */
  Math::vcross (z, v, y); // y orthogonal to z, v 
  Math::vcross (y, z, x); // x orthogonal to z and y
  // Normalize: Obtain the Orthonormal Base
  Math::vnorm (x);
  Math::vnorm (y);
  Math::vnorm (z);

  // Build the Rotation matrix (orthonormal)
  for (uint i = 0; i < 3; i++) {
    rot_m[i][0] = x[i];  
    rot_m[i][1] = y[i]; 
    rot_m[i][2] = z[i]; 
  }

  // f_t = RotMat(<p1,p3,p3>) x f 
  for (uint i = 0; i < backbone_len(); i++) {
    real px = rot_m[0][0] * backbone[i][0] +   
              rot_m[0][1] * backbone[i][1] +   
              rot_m[0][2] * backbone[i][2];   
    real py = rot_m[1][0] * backbone[i][0] +
              rot_m[1][1] * backbone[i][1] +
              rot_m[1][2] * backbone[i][2];   
    real pz = rot_m[2][0] * backbone[i][0] +
              rot_m[2][1] * backbone[i][1] +  
              rot_m[2][2] * backbone[i][2];  
    backbone[i].set_position(px,py,pz);
  }

  // c_t = RotMat(F1) x c
  for (uint i = 0; i < ncentroids(); i++) {
    real px = rot_m[0][0] * centroid[i].position[0] +   
              rot_m[0][1] * centroid[i].position[1] +   
              rot_m[0][2] * centroid[i].position[2];   
    real py = rot_m[1][0] * centroid[i].position[0] +
              rot_m[1][1] * centroid[i].position[1] +        
              rot_m[1][2] * centroid[i].position[2];   
    real pz = rot_m[2][0] * centroid[i].position[0] +
              rot_m[2][1] * centroid[i].position[1] +  
              rot_m[2][2] * centroid[i].position[2]; 
    centroid[i].set_position (px,py,pz);
  }

  // Translate this fragment so that it overlaps f on the hooking plan
  shift_v[0] = shift_v[1] = shift_v[2] = 0;
  uint N_idx = 
    (direction == LEFT_TO_RIGHT) ? 2 // N-start
    : backbone_len() - 1;//3; // C-term /////N-end
  
  shift_v[0] = -backbone[N_idx][0] + hook.get_N()[0];
  shift_v[1] = -backbone[N_idx][1] + hook.get_N()[1];
  shift_v[2] = -backbone[N_idx][2] + hook.get_N()[2];
  
  for (uint i = 0; i < backbone_len(); i++) {
    backbone[i].position[0] += shift_v[0];
    backbone[i].position[1] += shift_v[1];
    backbone[i].position[2] += shift_v[2];
  }
  
  for (uint i = 0; i < ncentroids(); i++) {
    centroid[i].position[0] += shift_v[0];
    centroid[i].position[1] += shift_v[1];
    centroid[i].position[2] += shift_v[2];
  }
  
  transform_aux_elements ();  
}//-


// performs a transformation of the type: f = (R x f ) + v
void 
Fragment::transform (const R_MAT& R, const vec3& v) {
  vector <Atom> auxbb = backbone;
  vector <Atom> auxcg = centroid;
  
  // Move backbone atoms 
  for (uint i=0; i<backbone.size(); i++) {
    real *dst = backbone[i].position;
    for (int ii = 0; ii < 3; ii++) {
      dst[ii] = R[ii][0] * auxbb[i][0]
     	      + R[ii][1] * auxbb[i][1]
    	      + R[ii][2] * auxbb[i][2];
      dst[ii] += v[ii];
    }
  }

  //  Move Centroids
  for (uint i=0; i<centroid.size(); i++) {
    real *dst = centroid[i].position;
    for (int ii = 0; ii < 3; ii++) {
      dst[ii] = R[ii][0] * auxcg[i][0]
              + R[ii][1] * auxcg[i][1]
              + R[ii][2] * auxcg[i][2];
      dst[ii] += v[ii];
    }
  }
  
  // TRAIL THIS CHANGE!
  // save current transformations
  for (uint i=0; i<3; i++){
    shift_v[i]=v[i];
    for (uint ii=0; ii<3; ii++)
      rot_m[i][ii] = R[i][ii];
  }

  transform_aux_elements ();
}
//-

void 
Fragment::copy_rot_mat(const Fragment& other) {
  for (int i=0; i<3; i++)
    for (int ii=0; ii<3; ii++) 
      rot_m[i][ii] = other.rot_m[i][ii];
}
//-

void 
Fragment::copy_sh_vec(const Fragment& other) {
  for (int i=0; i<3; i++) 
    shift_v[i] = other.shift_v[i];
}
//-

// The phi-angle is the torsion angle around the N-CA bond
// is the torsion angle between the planes 
// involving the backbone atoms C'--N--CA--C'
void
Fragment::compute_phi() {
  vec3 a, b, c, d, n1, n2;
  uint C1 = 0, N = 2, CA = 3, C2 = 4;

  // plane1 C'-N-Ca 
  Math::vsub(backbone[N].position, backbone[C1].position, a);  // a = N - C1'
  Math::vsub(backbone[CA].position, backbone[C1].position, b);  // b = CA - N
  
  // plane2 N-Ca-C' 
  Math::vsub(backbone[CA].position, backbone[N].position, c);  // c = CA - N'
  Math::vsub(backbone[C2].position, backbone[N].position, d); // d = CA - C2'

  // Normal to plane1 and plane2
  Math::vcross (a, b, n1); 
  Math::vcross (c, d, n2);
  
  phi = acos((Math::vdot(n1, n2))/(Math::vnorm2(n1)*Math::vnorm2(n2)));
}
//-

// The psi-angle is the torsion angle around the CA-C bond
// is the torsion angle between the planes 
// involving the backbone atoms  N--CA--C'--N
void
Fragment::compute_psi() {
  vec3 a, b, c, d, n1, n2;
  uint l = backbone.size();
  uint N1 = l-5, CA = l-4, C = l-3, N2 = l-1; 

  // plane1 N-Ca-C'
  Math::vsub(backbone[CA].position, backbone[N1].position, a);  // a = CA - N1
  Math::vsub(backbone[C].position, backbone[N1].position, b);   // b = C' - CA

  // plane2 Ca-C'-N 
  Math::vsub(backbone[C].position, backbone[CA].position, c);  // c = C' - CA
  Math::vsub(backbone[N2].position, backbone[CA].position, d);  // d = N2 - C'

  // Normal to plane1 and plane2
  Math::vcross (a, b, n1); 
  Math::vcross (c, d, n2);
  
  psi = acos((Math::vdot(n1, n2))/(Math::vnorm2(n1)*Math::vnorm2(n2)));
}
//-

void
Fragment::dump() {
  cout << "FRAGMENT id (" << id << ") INFO" << endl;
    cout << "  Type:" << type << " PID:" << pid << endl;
  cout << " Freq:"<< frequency << " Prob:" << probability << endl;
  cout << " phi: " << phi << " psi: " << psi << endl; 
  cout << "  backbone len: " << backbone_len();
  cout << "  AA start/end: ["<< aa_start << ", "<< aa_end << "]";
  cout << "  BB start/end: ["<< bb_start << ", "<< bb_end << "]" << endl;
  cout << "  Dihedral angles: phi: " << phi << " psi: " << psi << endl;

  cout << "  Classes : ";
  // @todo: drop this or
  for (uint i=0; i < aa_seq.size(); i++)
    cout << aa_seq.at(i) << " ";
  cout << endl;

  for (uint i=0; i < aa_seq.size(); i++)
    cout << aa_seq.at(i) << " ";
  cout << endl;

  /* Atoms */
  cout << "Backbone:" << endl;
  for (uint i=0; i<backbone_len(); i ++) {
    backbone.at(i).dump();
  }
  /* printf("CG:\n"); */
  /* for (i=0; i<F->ncg; i++) */
  /*   atom_displ (F->cg[i]); */
  /* printf("OA:\n"); */
  /* for (i=0; i<F->noa; i++) */
  /*   atom_displ (F->oa[i]); */
}//-
