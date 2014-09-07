/*********************************************************************
 * RMSD - Root Mean Squared Deviation
 * Compute RMSD of two given vectors of points
 *********************************************************************/
#ifndef FIASCO_RMSD__
#define FIASCO_RMSD__

#include <math.h>
#include <stdio.h>
#include "typedefs.h"

class Protein;
class LogicVariables;
class Fragment;
class Atom;

namespace Rmsd{
  void alloc(int size=-1);
  void dealloc ();

  real rmsd_compare(uint a1, uint a2);
  real rmsd_compare(std::vector<Atom> a1, std::vector<Atom> a2);
  void setup_rotation(real ref_xlist[][3], real mov_xlist[][3], 
                        int n_list, real mov_com[3], real mov_to_ref[3],
                        real R[3][3], real* E0);  
  
    
  real rmsd_superimpose(point v1[], int len1, point v2[], int len2, int offset=0);
  void calculate_centroids(int n, point x[], int offset, point p);
  void jacobi(real q[][5], int n, real d[], real v[][5]);
  void eigsrt(real d[], real v[][5], int n);
  void rotate(real a[][5], real s, real tau, int i, int j,int k, int l);
  
  /*
   * fast_rmsd()
   *
   * Fast calculation of rmsd w/o calculating a rotation matrix,
   * adapted from the BTK by Chris Saunders 11/2002. 
   */
  real fast_rmsd(real ref_xlist[][3], real mov_xlist[][3], int n_list);
  void setup_rotation(real ref_xlist[][3], real mov_xlist[][3], 
                        int n_list, real mov_com[3], real mov_to_ref[3],
                        real R[3][3], real* E0);  
  void fast_rmsd2(double ref_xlist[][3],
                 double mov_xlist[][3], 
                 int n_list,
                 double* rmsd);
    
  void setup_rotation2(double ref_xlist[][3],
                double mov_xlist[][3], 
                int n_list,
                double mov_com[3],
                double mov_to_ref[3],
                double R[3][3],
                double* E0);
    
  void vcrossd(double *a, double *b, double *c);
  double vdotd(double *x, double *y);  
    
}; //Rmsd


#endif
