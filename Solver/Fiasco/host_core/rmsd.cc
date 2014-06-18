#include "rmsd.h"
#include "globals.h"
#include "logic_variables.h"
#include "variable_point.h"
#include "variable_fragment.h"
#include "protein.h"
#include "utilities.h"
#include "mathematics.h"
#include "atom.h"

#include <iostream>
#include <cassert>
#include <cmath> 
#include <stdlib.h>

using namespace std;
using namespace Rmsd;

point *tertiary_compare_1;
point *tertiary_compare_2;

void Rmsd::alloc(int size) {
  if (size < 0)
    size = g_known_prot.get_bblen();

  tertiary_compare_1 = (point*) calloc(size, sizeof(point));
  tertiary_compare_2 = (point*) calloc(size, sizeof(point));
}//-

void Rmsd::dealloc() {
  delete tertiary_compare_1;
  delete tertiary_compare_2;
}//-

// FRAG
// C' O N Ca C' O N 
// PROT
//      N Ca C' O N Ca C' O
real
Rmsd::rmsd_compare(vector<Atom> atoms1, vector<Atom> atoms2) {
  assert (atoms1.size() == atoms2.size());
  size_t size = atoms1.size();
  for(uint i=0; i<size; i++){
    memcpy(&tertiary_compare_1[i], atoms1[i].position, sizeof(point));
  }
  for(uint i=0; i<size; i++){
    memcpy(&tertiary_compare_2[i], atoms2[i].position, sizeof(point));
  }
  return rmsd_superimpose (tertiary_compare_1, size, tertiary_compare_2, size, 0);
}//-


// Compare a protein part [start-end] with the logic var points [start-end]
// extreems included
real
Rmsd::rmsd_compare(uint start, uint end){
  uint len = end-start+1; //g_known_prot.get_bblen();
  
  for(uint i=0; i<len; i++){
    memcpy(&tertiary_compare_1[i], &g_known_prot[i+start].position, sizeof(point));
  }

    for(uint i=0; i<len; i++){
      memcpy(&tertiary_compare_2[i], 
	     &g_logicvars->var_point_list[i+start].lower_bound, sizeof(point));
    }
    
    real rmsd = rmsd_superimpose(tertiary_compare_1, len, tertiary_compare_2, len, 0);
    return rmsd;
    
}//rmsd_Compare


/**
 * @brief Superimpose
 * @param v1 A list of 3D coordinates (original protein)
 * @param len1 length of the vector v1
 * @param v2 A list of 3D coordinates (target protein or fragment)
 * @param len2 length of the vector v2
 * @param offset is the idx of the point variable from where start the
 * comparison. it starts from 0...n-1-len2
 * @note We assume v1 is the original protein and v2 a fragment
 */
real
Rmsd::rmsd_superimpose(point v1[], int len1, point v2[], int len2, int offset){
    real rmsd;
    int i,j;
    int n = len1 > len2 ? len1 : len2;
    //real* xc = NULL;
    //real* yc = NULL;
    point xc, yc;

    /* Declaring matrices */
    real D[n][3], d[n][3]; //gli scarti!
    real Q[5][5];
    real eigenvec[5][5], eigenval[5];
    
    calculate_centroids(n, v1, offset, xc);
    calculate_centroids(n, v2, 0, yc);

    /*calcolo degli scarti! */
    int i1, i2;
    i1 = offset; i2 = 0;
    while(i2 < n){
        for(j=0; j<3; j++){
            D[i2][j] = real((v1[i1][j] - xc[j]) + (v2[i2][j] - yc[j]));
            d[i2][j] = real((v1[i1][j] - xc[j]) - (v2[i2][j] - yc[j]));
        }
        i1++;
        i2++;
    }
    
    for(i=0; i<=4; i++)
      for (j=0; j<=4; j++)
          Q[i][j] = 0;

    for(i=0; i<n; i++){
      Q[1][1] += d[i][0]*d[i][0] + d[i][1]*d[i][1] + d[i][2]*d[i][2];
      Q[2][2] += d[i][0]*d[i][0] + D[i][1]*D[i][1] + D[i][2]*D[i][2];
      Q[3][3] += D[i][0]*D[i][0] + d[i][1]*d[i][1] + D[i][2]*D[i][2];
      Q[4][4] += D[i][0]*D[i][0] + D[i][1]*D[i][1] + d[i][2]*d[i][2];
      Q[1][2] += D[i][1]*d[i][2] - D[i][2]*d[i][1];
      Q[1][3] += D[i][2]*d[i][0] - D[i][0]*d[i][2];
      Q[1][4] += D[i][0]*d[i][1] - D[i][1]*d[i][0];
      Q[2][3] += d[i][0]*d[i][1] - D[i][0]*D[i][1];
      Q[2][4] += d[i][0]*d[i][2] - D[i][0]*D[i][2];
      Q[3][4] += d[i][1]*d[i][2] - D[i][1]*D[i][2];
    }
    
    Q[2][1] = Q[1][2];
    Q[3][1] = Q[1][3];
    Q[4][1] = Q[1][4];
    Q[3][2] = Q[2][3];
    Q[4][2] = Q[2][4];
    Q[4][3] = Q[3][4];
    
    // Factorization step!
    jacobi(Q, 4, eigenval, eigenvec);
    eigsrt(eigenval, eigenvec, 4);

    // construction of rotation matrix to superimpose the two sequences 
    // This is not used!
#ifdef ROTATE
    real T[3][3];
    T[0][0] = eigenvec[1][4]*eigenvec[1][4] +
            eigenvec[2][4]*eigenvec[2][4] -
            eigenvec[3][4]*eigenvec[3][4] -
            eigenvec[4][4]*eigenvec[4][4];
    T[0][1] = 2*(eigenvec[2][4]*eigenvec[3][4] + eigenvec[1][4]*eigenvec[4][4]);
    T[0][2] = 2*(eigenvec[2][4]*eigenvec[4][4] - eigenvec[1][4]*eigenvec[3][4]);
    T[1][0] = 2*(eigenvec[2][4]*eigenvec[3][4] - eigenvec[1][4]*eigenvec[4][4]);
    T[1][1] = eigenvec[1][4]*eigenvec[1][4] +
            eigenvec[3][4]*eigenvec[3][4] -
            eigenvec[2][4]*eigenvec[2][4] -
            eigenvec[4][4]*eigenvec[4][4];
    T[1][2] = 2*(eigenvec[3][4]*eigenvec[4][4] + eigenvec[1][4]*eigenvec[2][4]);
    T[2][0] = 2*(eigenvec[2][4]*eigenvec[4][4] + eigenvec[1][4]*eigenvec[3][4]);
    T[2][1] = 2*(eigenvec[3][4]*eigenvec[4][4] - eigenvec[1][4]*eigenvec[2][4]);
    T[2][2] = eigenvec[1][4]*eigenvec[1][4] +
            eigenvec[4][4]*eigenvec[4][4] -
            eigenvec[3][4]*eigenvec[3][4] -
            eigenvec[2][4]*eigenvec[2][4];
#endif
    rmsd = sqrt(eigenval[4]/n);
    return rmsd;
}//rmsd_superimpose

void
Rmsd::calculate_centroids(int n, point x[], int offset, point p){
    int i,j;
    for (i=0; i<3; i++){
        p[i] = 0;
        for(j=offset; j < n + offset; j++){
        p[i] += x[j][i];
        }
    p[i] /= real(n);
    }
}//calculate_centroids

/*
  Computes all eigenvalues and eigenvectors of a real symmetric matrix
  a[1..n][1..n].
  On output, elements of a above the diagonal are destroyed.
  d[1..n] returns the eigenvalues of a.
  v[1..n][1..n] is a matrix whose columns contain, on output, the
  normalized eigenvectors of a.
  Note:
  This function is taken from the book numerical recipes in C.
  There is a problem, i.e. there the vectors are stored as v[1..n]
  and not as v[0..n-1]. This must also taken into account while
  passing the matrices a and v! */
void
Rmsd::jacobi(real a[][5], int n, real d[], real v[][5]){
    int j,iq,ip,i;
    real tresh,theta,tau,t,sm,s,h,g,c;

    real b[n+1];  //vector(1,n);
    real z[n+1];  //vector(1,n);

    /* Initialize to the identity matrix. */
    for(ip=1; ip<=n; ip++){
        for(iq=1; iq<=n; iq++)
            v[ip][iq]=0.0;
        v[ip][ip]=1.0;
    }
    /* Initialize b and d to the diagonal of a. */
    for(ip=1; ip<=n; ip++){
        b[ip] = d[ip] = a[ip][ip];
        z[ip] = 0.0;
        /* This vector will accumulate terms of the form tapq as in equation (11.1.14) */
    }

    for(i=1; i<=50; i++){
    /* Sum off-diagonal elements. */
        sm = 0.0;
        for(ip=1;ip<=n-1;ip++){
            for(iq=ip+1;iq<=n;iq++)
                sm += fabs(a[ip][iq]);
        }

    /* The normal return, which relies on quadratic convergence to machine underflow. */
    if(sm == 0.0)
        return;

    /* ...on the first three sweeps. */
    if(i < 4)
      tresh = 0.2*sm/(n*n);
    /* ...thereafter. */
    else
      tresh = 0.0;

    for(ip=1; ip<=n-1; ip++){
        for(iq=ip+1; iq<=n; iq++){
            g = 100.0*fabs(a[ip][iq]);

            /*After four sweeps, skip the rotation if the off-diagonal
            element is small. */
            if(i > 4 && (real)(fabs(d[ip])+g) == (real)fabs(d[ip]) && (real)(fabs(d[iq])+g) == (real)fabs(d[iq]))
                a[ip][iq] = 0.0;
            else if (fabs(a[ip][iq]) > tresh){
                h = d[iq]-d[ip];
                if((real)(fabs(h)+g) == (real)fabs(h))
                    t = (a[ip][iq])/h; 			//t = 1/(2?)
            else{
                theta = 0.5*h/(a[ip][iq]); //Equation (11.1.10).
                t = 1.0/(fabs(theta)+sqrt(1.0+theta*theta));
                if(theta < 0.0)
                    t = -t;
            }
        c = 1.0/sqrt(1+t*t);
        s = t*c;
        tau = s/(1.0+c);
        h = t*a[ip][iq];
        z[ip] -= h;
        z[iq] += h;
        d[ip] -= h;
        d[iq] += h;
        a[ip][iq] = 0.0;

	  // Case of rotations 1 = j < p.
	  for (j=1;j<=ip-1;j++)
            rotate(a,s,tau,j,ip,j,iq);
	  // Case of rotations p < j < q.
	  for (j=ip+1;j<=iq-1;j++)
            rotate(a,s,tau,ip,j,j,iq);
	  // Case of rotations q < j = n.
	  for (j=iq+1;j<=n;j++)
            rotate(a,s,tau,ip,j,iq,j);
	  for (j=1;j<=n;j++)
            rotate(v,s,tau,j,ip,j,iq);
    }
      }
    }

    // Update d with the sum of tapq, and reinitialize z.
    for(ip=1;ip<=n;ip++) {
        b[ip] += z[ip];
        d[ip] = b[ip];
        z[ip] = 0.0;
    }
  }

  std::cout << "ERROR! Too many iterations in routine jacobi" << std::endl;
}//jacobi


void
Rmsd::rotate(real a[][5], real s, real tau, int i, int j,int k, int l){
    real g, h;
    g = a[i][j];
    h = a[k][l];
    a[i][j] = g-s*(h+g*tau);
    a[k][l] = h+s*(g-h*tau);
}//__rotate

/*
Given the eigenvalues d[1..n] and eigenvectors v[1..n][1..n] as output from jacobi (§11.1)
or tqli (§11.3), this routine sorts the eigenvalues into descending order, and rearranges
the columns of v correspondingly. The method is straight insertion.
Also this routine is taken from the numerical recipes in C book.
Beware of how vectors are managed. */
void
Rmsd::eigsrt(real d[], real v[][5], int n){
  int k,j,i;
  real p;
  for(i=1; i<n; i++){
    p = d[k=i];
    for(j=i+1; j<=n; j++)
      if(d[j] >= p)
	p = d[k=j];
    
    if(k != i){
      d[k] = d[i];
      d[i] = p;
      for(j=1; j<=n; j++){
	p = v[j][i];
	v[j][i] = v[j][k];
	v[j][k] = p;
      }
    }
  }
}//eigsrt

/*
 * Fast calculation of rmsd w/o calculating a rotation matrix.
 *
 *   Chris Saunders 11/2002 - Fast rmsd calculation by the method of 
 * Kabsch 1978, where the required eigenvalues are found by an 
 * analytical, rather than iterative, method to save time. 
 * The cubic factorization used to accomplish this only produces 
 * stable eigenvalues for the transpose(R]*R matrix of a typical 
 * protein after the whole matrix has been normalized. Note that 
 * the normalization process used here is completely empirical 
 * and that, at the present time, there are **no checks** or 
 * warnings on the quality of the (potentially unstable) cubic 
 * factorization. 
 *
 */

real
Rmsd::fast_rmsd(real ref_xlist[][3], 
                real mov_xlist[][3], 
                int n_list)
{ 
    real R[3][3];
    real d0,d1,d2,e0,e1,f0;
    real omega;
    real mov_com[3];
    real mov_to_ref[3];
    
    /* cubic roots */
    real r1,r2,r3;
    real rlow;
    
    real v[3];
    real Eo, residual;
    
    setup_rotation(ref_xlist, mov_xlist, n_list, 
                   mov_com, mov_to_ref, R, &Eo);
    
    /* 
     * check if the determinant is greater than 0 to
     * see if R produces a right-handed or left-handed
     * co-ordinate system.
     */
    Math::vcross(&R[1][0], &R[2][0], v);
    if (Math::vdot(&R[0][0], v) > 0.0)
        omega = 1.0;
    else
        omega = -1.0;
    
    /*
     * get elements we need from tran(R) x R 
     *  (funky matrix naming relic of first attempt using pivots)
     *          matrix = d0 e0 f0
     *                      d1 e1
     *                         d2
     * divide matrix by d0, so that cubic root algorithm can handle it 
     */
    
    d0 =  R[0][0]*R[0][0] + R[1][0]*R[1][0] + R[2][0]*R[2][0];
    
    d1 = (R[0][1]*R[0][1] + R[1][1]*R[1][1] + R[2][1]*R[2][1])/d0;
    d2 = (R[0][2]*R[0][2] + R[1][2]*R[1][2] + R[2][2]*R[2][2])/d0;
    
    e0 = (R[0][0]*R[0][1] + R[1][0]*R[1][1] + R[2][0]*R[2][1])/d0;
    e1 = (R[0][1]*R[0][2] + R[1][1]*R[1][2] + R[2][1]*R[2][2])/d0;
    
    f0 = (R[0][0]*R[0][2] + R[1][0]*R[1][2] + R[2][0]*R[2][2])/d0;
    
    /* cubic roots */
    {
        real B, C, D, q, q3, r, theta;
        /*
         * solving for eigenvalues as det(A-I*lambda) = 0
         * yeilds the values below corresponding to:
         * lambda**3 + B*lambda**2 + C*lambda + D = 0
         *   (given that d0=1.)
         */
        B = -1.0 - d1 - d2;
        C = d1 + d2 + d1*d2 - e0*e0 - f0*f0 - e1*e1;
        D = e0*e0*d2 + e1*e1 + f0*f0*d1 - d1*d2 - 2*e0*f0*e1;
        
        /* cubic root method of Viete with all safety belts off */
        q = (B*B - 3.0*C) / 9.0;
        q3 = q*q*q;
        r = (2.0*B*B*B - 9.0*B*C + 27.0*D) / 54.0;
        theta = acos(r/sqrt(q3));
        r1 = r2 = r3 = -2.0*sqrt(q);
        r1 *= cos(theta/3.0);
        r2 *= cos((theta + 2.0*M_PI) / 3.0);
        r3 *= cos((theta - 2.0*M_PI) / 3.0);
        r1 -= B / 3.0;
        r2 -= B / 3.0;
        r3 -= B / 3.0;
    }
    
    /* undo the d0 norm to get eigenvalues */
    r1 = r1*d0;
    r2 = r2*d0;
    r3 = r3*d0;
    
    /* set rlow to lowest eigenval; set other two to r1,r2 */
    if (r3<r1 && r3<r2)
    {
        rlow = r3;
    }
    else if (r2<r1 && r2<r3)
    {
        rlow = r2; 
        r2 = r3;
    } 
    else 
    { 
        rlow = r1; 
        r1 = r3;
    }
    
    residual = Eo - sqrt(r1) - sqrt(r2) - omega*sqrt(rlow);
    
    /* Debug */
    if(sqrt( residual*2.0 / (n_list) ) > 100 || sqrt( residual*2.0 / (n_list) ) < 0){
        cout << "residual " << residual << endl;
        cout << "Eo " << Eo << endl;
        cout << "sqrt(r1) " << sqrt(r1) << " r1 " << r1 << endl;
        cout << "sqrt(r2) " << sqrt(r2) << " r2 " << r2 << endl;
        cout << "r3 " << r3 << endl;
        cout << "rlow " << rlow << endl;
        cout << "omega " << omega << endl;
        cout << "rlow " << rlow << endl;
        cout << "molt " << omega*sqrt(rlow) << endl;
    }
    
    return sqrt( residual*2.0 / (n_list) ); 
}//fast_rmsd


void 
Rmsd::fast_rmsd2(double ref_xlist[][3],
                 double mov_xlist[][3], 
                 int n_list,
                 double* rmsd)
{ 
    double R[3][3];
    double d0,d1,d2,e0,e1,f0;
    double omega;
    double mov_com[3];
    double mov_to_ref[3];
    
    /* cubic roots */
    double r1,r2,r3;
    double rlow;
    
    double v[3];
    double Eo, residual;
    
    setup_rotation2(ref_xlist, mov_xlist, n_list, 
                    mov_com, mov_to_ref, R, &Eo);
    
    /* 
     * check if the determinant is greater than 0 to
     * see if R produces a right-handed or left-handed
     * co-ordinate system.
     */
    vcrossd(v, &R[1][0], &R[2][0]);
    if (vdotd(&R[0][0], v) > 0.0)
        omega = 1.0;
    else
        omega = -1.0;
    
    /*
     * get elements we need from tran(R) x R 
     *  (funky matrix naming relic of first attempt using pivots)
     *          matrix = d0 e0 f0
     *                      d1 e1
     *                         d2
     * divide matrix by d0, so that cubic root algorithm can handle it 
     */
    
    d0 =  R[0][0]*R[0][0] + R[1][0]*R[1][0] + R[2][0]*R[2][0];
    
    d1 = (R[0][1]*R[0][1] + R[1][1]*R[1][1] + R[2][1]*R[2][1])/d0;
    d2 = (R[0][2]*R[0][2] + R[1][2]*R[1][2] + R[2][2]*R[2][2])/d0;
    
    e0 = (R[0][0]*R[0][1] + R[1][0]*R[1][1] + R[2][0]*R[2][1])/d0;
    e1 = (R[0][1]*R[0][2] + R[1][1]*R[1][2] + R[2][1]*R[2][2])/d0;
    
    f0 = (R[0][0]*R[0][2] + R[1][0]*R[1][2] + R[2][0]*R[2][2])/d0;
    
    /* cubic roots */
    {
        double B, C, D, q, q3, r, theta;
        /*
         * solving for eigenvalues as det(A-I*lambda) = 0
         * yeilds the values below corresponding to:
         * lambda**3 + B*lambda**2 + C*lambda + D = 0
         *   (given that d0=1.)
         */
        B = -1.0 - d1 - d2;
        C = d1 + d2 + d1*d2 - e0*e0 - f0*f0 - e1*e1;
        D = e0*e0*d2 + e1*e1 + f0*f0*d1 - d1*d2 - 2*e0*f0*e1;
        
        /* cubic root method of Viete with all safety belts off */
        q = (B*B - 3.0*C) / 9.0;
        q3 = q*q*q;
        r = (2.0*B*B*B - 9.0*B*C + 27.0*D) / 54.0;
        theta = acos(r/sqrt(q3));
        r1 = r2 = r3 = -2.0*sqrt(q);
        r1 *= cos(theta/3.0);
        r2 *= cos((theta + 2.0*M_PI) / 3.0);
        r3 *= cos((theta - 2.0*M_PI) / 3.0);
        r1 -= B / 3.0;
        r2 -= B / 3.0;
        r3 -= B / 3.0;
    }
    
    /* undo the d0 norm to get eigenvalues */
    r1 = r1*d0;
    r2 = r2*d0;
    r3 = r3*d0;
    
    /* set rlow to lowest eigenval; set other two to r1,r2 */
    if (r3<r1 && r3<r2)
    {
        rlow = r3;
    }
    else if (r2<r1 && r2<r3)
    {
        rlow = r2; 
        r2 = r3;
    } 
    else 
    { 
        rlow = r1; 
        r1 = r3;
    }
    
    residual = Eo - sqrt(r1) - sqrt(r2) - omega*sqrt(rlow);
    residual = (residual < 0.0001) ? 0 : residual;
    *rmsd = sqrt( (double) residual*2.0 / ((double) n_list) );
		
    assert (*rmsd == *rmsd); 
}//fast_rmsd

/*
 * setup_rotation() 
 *
 *      given two lists of x,y,z coordinates, constructs
 * the correlation R matrix and the E value needed to calculate the
 * least-squares rotation matrix.
 */

void 
Rmsd::setup_rotation(real ref_xlist[][3],
                     real mov_xlist[][3], 
                     int n_list,
                     real mov_com[3],
                     real mov_to_ref[3],
                     real R[3][3],
                     real* E0)
{
    int i, j, n;
    real ref_com[3];
    
    /* calculate the centre of mass */
    for (i=0; i<3; i++)
    { 
        mov_com[i] = 0.0;
        ref_com[i] = 0.0;
    }
    
    for (n=0; n<n_list; n++) 
        for (i=0; i<3; i++)
        { 
            mov_com[i] += mov_xlist[n][i];
            ref_com[i] += ref_xlist[n][i];
        }
    
    for (i=0; i<3; i++)
    {
        mov_com[i] /= n_list;
        ref_com[i] /= n_list;
        mov_to_ref[i] = ref_com[i] - mov_com[i];
    }
    
    /* shift mov_xlist and ref_xlist to centre of mass */
    for (n=0; n<n_list; n++) 
        for (i=0; i<3; i++)
        { 
            mov_xlist[n][i] -= mov_com[i];
            ref_xlist[n][i] -= ref_com[i];
        }
    
    /* initialize */
    for (i=0; i<3; i++)
        for (j=0; j<3; j++) 
            R[i][j] = 0.0;
    *E0 = 0.0;
    
    for (n=0; n<n_list; n++) 
    {
        /* 
         * E0 = 1/2 * sum(over n): y(n)*y(n) + x(n)*x(n) 
         */
        for (i=0; i<3; i++)
            *E0 +=  mov_xlist[n][i] * mov_xlist[n][i]  
            + ref_xlist[n][i] * ref_xlist[n][i];
        
        /*
         * correlation matrix R:   
         *   R[i,j) = sum(over n): y(n,i) * x(n,j)  
         *   where x(n) and y(n) are two vector sets   
         */
        for (i=0; i<3; i++)
        {
            for (j=0; j<3; j++)
                R[i][j] += mov_xlist[n][i] * ref_xlist[n][j];
        }
    }
    *E0 *= 0.5;
}//setup_rotation

void 
Rmsd::setup_rotation2(double ref_xlist[][3],
                      double mov_xlist[][3], 
                      int n_list,
                      double mov_com[3],
                      double mov_to_ref[3],
                      double R[3][3],
                      double* E0)
{
    int i, j, n;
    double ref_com[3];
    
    /* calculate the centre of mass */
    for (i=0; i<3; i++)
    { 
        mov_com[i] = 0.0;
        ref_com[i] = 0.0;
    }
    
    for (n=0; n<n_list; n++) 
        for (i=0; i<3; i++)
        { 
            mov_com[i] += mov_xlist[n][i];
            ref_com[i] += ref_xlist[n][i];
        }
    
    for (i=0; i<3; i++)
    {
        mov_com[i] /= n_list;
        ref_com[i] /= n_list;
        mov_to_ref[i] = ref_com[i] - mov_com[i];
    }
    
    /* shift mov_xlist and ref_xlist to centre of mass */
    for (n=0; n<n_list; n++) 
        for (i=0; i<3; i++)
        { 
            mov_xlist[n][i] -= mov_com[i];
            ref_xlist[n][i] -= ref_com[i];
        }
    
    /* initialize */
    for (i=0; i<3; i++)
        for (j=0; j<3; j++) 
            R[i][j] = 0.0;
    *E0 = 0.0;
    
    for (n=0; n<n_list; n++) 
    {
        /* 
         * E0 = 1/2 * sum(over n): y(n)*y(n) + x(n)*x(n) 
         */
        for (i=0; i<3; i++)
            *E0 +=  mov_xlist[n][i] * mov_xlist[n][i]  
            + ref_xlist[n][i] * ref_xlist[n][i];
        
        /*
         * correlation matrix R:   
         *   R[i,j) = sum(over n): y(n,i) * x(n,j)  
         *   where x(n) and y(n) are two vector sets   
         */
        for (i=0; i<3; i++)
        {
            for (j=0; j<3; j++)
                R[i][j] += mov_xlist[n][i] * ref_xlist[n][j];
        }
    }
    *E0 *= 0.5;
}//setup_rotation


void 
Rmsd::vcrossd(double *a, double *b, double *c) {
    a[0] = b[1] * c[2] - b[2] * c[1];
    a[1] = b[2] * c[0] - b[0] * c[2];
    a[2] = b[0] * c[1] - b[1] * c[0];
}//-

double 
Rmsd::vdotd(double *x, double *y) {
    return (x[0]*y[0] + x[1]*y[1] + x[2]*y[2]);
}//-
