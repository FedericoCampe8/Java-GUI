#ifndef COCOS_RMSD_FAST__
#define COCOS_RMSD_FAST__

#include "globals.h"

namespace Rmsd_fast{
  /*
   * calculate_rotation_rmsd()
   *
   *   given two lists of x,y,z coordinates, constructs
   *    - mov_com: the centre of mass of the mov list
   *    - mov_to_ref: vector between the com of mov and ref
   *    - U: the rotation matrix for least-squares, usage of
   *         of the matrix U[3][3] is
   *           for (i=0; i<3; i++)
   *           {
   *             rotated_v[i] = 0.0;
   *             for (j=0; j<3; j++)
   *               rotated_v[i] += U[i][j] * v[j];
   *           }
   *    - rmsd: measures similarity between the vectors
   */
  void calculate_rotation_rmsd(double ref_xlist[][3],
                               double mov_xlist[][3],
                               int n_list,
                               double mov_com[3],
                               double mov_to_ref[3],
                               double U[3][3],
                               double* rmsd);
  
  /*
   * fast_rmsd()
   *
   * Fast calculation of rmsd w/o calculating a rotation matrix,
   * adapted from the BTK by Chris Saunders 11/2002.
   */
  void fast_rmsd(double ref_xlist[][3],
                 double mov_xlist[][3],
                 int n_list,
                 double* rmsd);
  
  void get_rmsd ( real* beam_str, real* beam_energies,
                  real* validity_solutions,
                  real* known_prot, int nres,
                  int scope_first, int scope_second,
                  int n_blocks );
  
  real get_rmsd ( real* my_prot,
                  real* known_prot, int nres,
                  int scope_first, int scope_second );
  
  void normalize(double a[3]);
  double dot(double a[3], double b[3]);
  void cross(double a[3], double b[3], double c[3]);
  void setup_rotation(double ref_xlist[][3],
                      double mov_xlist[][3],
                      int n_list,
                      double mov_com[3],
                      double mov_to_ref[3],
                      double R[3][3],
                      double* E0);
  int jacobi3(double a[3][3], double d[3], double v[3][3], int* n_rot);
  int diagonalize_symmetric(double matrix[3][3],
                            double eigen_vec[3][3],
                            double eigenval[3]);
  
  int calculate_rotation_matrix(double R[3][3],
                                double U[3][3],
                                double E0,
                                double* residual);
  
}; //Rmsd_fast

#endif