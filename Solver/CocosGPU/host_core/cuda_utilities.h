/***************************************
 *             Cuda Utilities          *
 ***************************************/
#ifndef COCOS_CUDA_UTILITIES__
#define COCOS_CUDA_UTILITIES__

#include "globals.h"
#include "cuda_math.h"

/* @note: "rotate_point_about_line" 
 * see http://paulbourke.net/geometry/rotate/
 * for more details.
 */
__forceinline__ __device__
void rotate_point_about_line( real* in_point, real theta_rad,
                              real* p1, real* p2,
                              real* out_point ) {
  real u[3];
  real q2[3];
  real scale;
  
  /// Step 1
  out_point[ 0 ] = in_point[ 0 ] - p1[ 0 ];
  out_point[ 1 ] = in_point[ 1 ] - p1[ 1 ];
  out_point[ 2 ] = in_point[ 2 ] - p1[ 2 ];
  u[ 0 ]         = p2[ 0 ] - p1[ 0 ];
  u[ 1 ]         = p2[ 1 ] - p1[ 1 ];
  u[ 2 ]         = p2[ 2 ] - p1[ 2 ];

  scale = sqrt ( ( u[0]*u[0] ) + ( u[1]*u[1] ) + ( u[2]*u[2] ) );
  if ( scale > CLOSE_TO_ZERO_VAL ) {
    u[0] /= scale;
    u[1] /= scale;
    u[2] /= scale;
  }
  else
    u[0] = u[1] = u[2] = 0.0;
  
  /// Step 2
  scale = sqrt ( ( u[1]*u[1] ) + ( u[2]*u[2] ) );
  if ( scale > CLOSE_TO_ZERO_VAL ) {
    q2[0] = out_point[0];
    q2[1] = out_point[1] * u[2] / scale - out_point[2] * u[1] / scale;
    q2[2] = out_point[1] * u[1] / scale + out_point[2] * u[2] / scale;
  }
  else {
    q2[0] = out_point[0];
    q2[1] = out_point[1];
    q2[2] = out_point[2];
    scale = 0.0;
  }

  /// Step 3
  out_point[0] = q2[0] * scale - q2[2] * u[0];
  out_point[1] = q2[1];
  out_point[2] = q2[0] * u[0] + q2[2] * scale;
  
  /// Step 4
  q2[0] = out_point[0] * cos( theta_rad ) - out_point[1] * sin( theta_rad );
  q2[1] = out_point[0] * sin( theta_rad ) + out_point[1] * cos( theta_rad );
  q2[2] = out_point[2];
  
  /// Inverse of step 3
  out_point[0] =   q2[0] * scale + q2[2] * u[0];
  out_point[1] =   q2[1];
  out_point[2] = - q2[0] * u[0] + q2[2] * scale;
  
  /// Inverse of step 2
  if ( scale > CLOSE_TO_ZERO_VAL ) {
    q2[0] =   out_point[0];
    q2[1] =   out_point[1] * u[2] / scale + out_point[2] * u[1] / scale;
    q2[2] = - out_point[1] * u[1] / scale + out_point[2] * u[2] / scale;
  }
  else {
    q2[0] = out_point[0];
    q2[1] = out_point[1];
    q2[2] = out_point[2];
  }
  
  /// Inverse of step 1
  out_point[ 0 ] = q2[ 0 ] + p1[ 0 ];
  out_point[ 1 ] = q2[ 1 ] + p1[ 1 ];
  out_point[ 2 ] = q2[ 2 ] + p1[ 2 ];
}//rotate_point_about_line

__forceinline__ __device__
void move_phi ( real * aa_points, real degree, int v_id, int ca_pos, int first_res, int is_thread_idx=-1 ) {
  int my_thread = (is_thread_idx >= 0) ? is_thread_idx : threadIdx.x;
  if ( my_thread >= first_res ) {
    int ca_res = ( my_thread * 5 + 1 ) * 3;
    if ( my_thread < v_id ) {
      ///ROTATE LEFT-CA
      rotate_point_about_line( &aa_points[ ca_res ], degree,
                               &aa_points[ ca_pos ], &aa_points[ ca_pos-3 ],
                               &aa_points[ ca_res ] );
      ///ROTATE LEFT-N
      rotate_point_about_line( &aa_points[ ca_res-3 ], degree,
                               &aa_points[ ca_pos ], &aa_points[ ca_pos-3 ],
                               &aa_points[ ca_res-3 ] );
    }
    
    ///ROTATE LEFT-H
    rotate_point_about_line( &aa_points[ ca_res+9 ], degree,
                             &aa_points[ ca_pos ], &aa_points[ ca_pos-3 ],
                             &aa_points[ ca_res+9 ] );
    
    if ( my_thread > 0 ) {
      ///ROTATE LEFT-O
      rotate_point_about_line( &aa_points[ ca_res-9 ], degree,
                               &aa_points[ ca_pos ], &aa_points[ ca_pos-3 ],
                               &aa_points[ ca_res-9 ] );
      ///ROTATE LEFT-C
      //if ( my_thread > v_id + 1 ) // Added
      rotate_point_about_line( &aa_points[ ca_res-12 ], degree,
                               &aa_points[ ca_pos ], &aa_points[ ca_pos-3 ],
                               &aa_points[ ca_res-12 ] );
    }
  }// first_res
}//move_phi

__forceinline__ __device__
void move_psi ( real * aa_points, real degree, int v_id, int ca_pos, int last_res, int is_thread_idx=-1 ) {
  int my_thread = (is_thread_idx >= 0) ? is_thread_idx : threadIdx.x;
  if ( my_thread < last_res ) {
    int ca_res = ( my_thread * 5 + 1 ) * 3;
    if ( my_thread > v_id ) {
      ///ROTATE RIGHT-CA
      rotate_point_about_line( &aa_points[ ca_res ], degree,
                               &aa_points[ ca_pos ], &aa_points[ ca_pos+3 ],
                               &aa_points[ ca_res ] );
      ///ROTATE RIGHT-C
      rotate_point_about_line( &aa_points[ ca_res+3 ], degree,
                               &aa_points[ ca_pos ], &aa_points[ ca_pos+3 ],
                               &aa_points[ ca_res+3 ] );
      
    }
    
    ///ROTATE RIGHT-O
    rotate_point_about_line( &aa_points[ ca_res+6 ], degree,
                             &aa_points[ ca_pos ], &aa_points[ ca_pos+3 ],
                             &aa_points[ ca_res+6 ] );
    
    if ( my_thread > v_id ) {
      ///ROTATE RIGHT-H
      rotate_point_about_line( &aa_points[ ca_res+9 ], degree,
                               &aa_points[ ca_pos ], &aa_points[ ca_pos+3 ],
                               &aa_points[ ca_res+9 ] );
    }
    
    if ( my_thread < last_res-1 ) {//-1
      ///ROTATE RIGHT-N
      rotate_point_about_line( &aa_points[ ca_res+12 ], degree,
                               &aa_points[ ca_pos ], &aa_points[ ca_pos+3 ],
                               &aa_points[ ca_res+12 ] );
    }
  }/// last_res
}//move_psi

__forceinline__ __device__
real
centroid_chi2 ( aminoacid a ) {
  if (a==ala) return 110.53;
  if (a==arg) return 113.59;
  if (a==asn) return 117.73;
  if (a==asp) return 116.03;
  if (a==cys) return 115.36;
  if (a==gln) return 115.96;
  if (a==glu) return 115.98;
  if (a==gly) return 0;
  if (a==his) return 115.38;
  if (a==ile) return 118.17;
  if (a==leu) return 119.90;
  if (a==lys) return 115.73;
  if (a==met) return 115.79;
  if (a==phe) return 114.40;
  if (a==pro) return 123.58;
  if (a==ser) return 110.33;
  if (a==thr) return 111.67;
  if (a==trp) return 109.27;
  if (a==tyr) return 113.14;
  if (a==val) return 114.46;
  return 0;
}//centroid_chi2

__forceinline__ __device__
real
centroid_torsional_angle ( aminoacid a ) {
  if (a==ala) return -138.45;
  if (a==arg) return -155.07;
  if (a==asn) return -144.56;
  if (a==asp) return -142.28;
  if (a==cys) return -142.28;
  if (a==gln) return -149.99;
  if (a==glu) return -147.56;
  if (a==gly) return -0;
  if (a==his) return -144.08;
  if (a==ile) return -151.72;
  if (a==leu) return -153.24;
  if (a==lys) return -153.03;
  if (a==met) return -159.50;
  if (a==phe) return -146.92;
  if (a==pro) return -105.62;
  if (a==ser) return -139.94;
  if (a==thr) return -142.28;
  if (a==trp) return -155.35;
  if (a==tyr) return -149.29;
  if (a==val) return -150.47;
  return 0;
}//centroid_torsional_angle

__forceinline__ __device__
real
centroid_distance ( aminoacid a ) {
  if (a==ala) return 1.53;
  if (a==arg) return 3.78;
  if (a==asn) return 2.27;
  if (a==asp) return 2.24;
  if (a==cys) return 2.03;
  if (a==gln) return 2.85;
  if (a==glu) return 2.83;
  if (a==gly) return 0;
  if (a==his) return 3.01;
  if (a==ile) return 2.34;
  if (a==leu) return 2.62;
  if (a==lys) return 3.29;
  if (a==met) return 2.95;
  if (a==phe) return 3.41;
  if (a==pro) return 1.88;
  if (a==ser) return 1.71;
  if (a==thr) return 1.94;
  if (a==trp) return 3.87;
  if (a==tyr) return 3.56;
  if (a==val) return 1.97;
  return 0;
}//centroid_distance

__forceinline__ __device__
int
centroid_radius ( aminoacid a ) {
  if (a==ala) return 190;
  if (a==arg) return 280;
  if (a==asn) return 222;
  if (a==asp) return 219;
  if (a==cys) return 213;
  if (a==gln) return 241;
  if (a==glu) return 238;
  if (a==gly) return 120;
  if (a==his) return 249;
  if (a==ile) return 249;
  if (a==leu) return 249;
  if (a==lys) return 265;
  if (a==met) return 255;
  if (a==phe) return 273;
  if (a==pro) return 228;
  if (a==ser) return 192;
  if (a==thr) return 216;
  if (a==trp) return 299;
  if (a==tyr) return 276;
  if (a==val) return 228;
  return 100; // default
}//-

__forceinline__ __device__
void
calculate_cg_atom ( aminoacid a, real* ca1, real* ca2, real* ca3, real* cg, int* radius ) {
  /// Placement of the centroid using dist, chi2, e tors
  /// v1 is the normalized vector w.r.t. ca1, ca2
  real v1[3];
  vsub ( ca2, ca1, v1 );
  vnorm ( v1 );
  
  /// v2 is the normalized vector w.r.t. ca2, ca3
  real v2[3];
  vsub ( ca3, ca2, v2 );
  vnorm ( v2 );
  
  /// Compute v1 (subtracting the component along v2)
  /// in order to obtain v1 and v2 orthogonal each other
  real x = vdot ( v1, v2 );
  v1[ 0 ] = v1[ 0 ] - x * v2[ 0 ];
  v1[ 1 ] = v1[ 1 ] - x * v2[ 1 ];
  v1[ 2 ] = v1[ 2 ] - x * v2[ 2 ];
  vnorm ( v1 );
  
  /// Compute v3 orthogonal to v1 and v2
  real v3[3];
  vcross ( v1, v2, v3 );
  
  /// Using Cramer method
  real factor;
  real b[3];
  real R[3][3];
  real D, Dx, Dy, Dz;
  real tors   = centroid_torsional_angle ( a ) * PI_VAL/180;
  b[0] = cos( (centroid_chi2 ( a )) * PI_VAL/180 );
  factor = sqrt( 1 - ( b[0] * b[0] ) );
  b[1] = sin( tors ) * factor ;
  b[2] = cos( tors ) * factor ;
  
  R[0][0] = v2[0];
  R[0][1] = v2[1];
  R[0][2] = v2[2];
  
  R[1][0] = v3[0];
  R[1][1] = v3[1];
  R[1][2] = v3[2];
  
  R[2][0] = -v1[0];
  R[2][1] = -v1[1];
  R[2][2] = -v1[2];
  
  D =
  R[0][0] * R[1][1] * R[2][2] +
  R[0][1] * R[1][2] * R[2][0] +
  R[0][2] * R[1][0] * R[2][1] -
  R[0][2] * R[1][1] * R[2][0] -
  R[0][1] * R[1][0] * R[2][2] -
  R[0][0] * R[1][2] * R[2][1];
  Dx =
  b[0] * (R[1][1] * R[2][2] - R[1][2] * R[2][1]) +
  b[1] * (R[2][1] * R[0][2] - R[2][2] * R[0][1]) +
  b[2] * (R[0][1] * R[1][2] - R[0][2] * R[1][1]) ;
  Dy =
  b[0] * (R[1][2] * R[2][0] - R[1][0] * R[2][2]) +
  b[1] * (R[2][2] * R[0][0] - R[2][0] * R[0][2]) +
  b[2] * (R[0][2] * R[1][0] - R[0][0] * R[1][2]) ;
  Dz =
  b[0] * (R[1][0] * R[2][1] - R[1][1] * R[2][0]) +
  b[1] * (R[2][0] * R[0][1] - R[2][1] * R[0][0]) +
  b[2] * (R[0][0] * R[1][1] - R[0][1] * R[1][0]) ;
  
  real v[3];
  v[ 0 ] = Dx/D;
  v[ 1 ] = Dy/D;
  v[ 2 ] = Dz/D;
  
  /// Now compute centroids coordinates
  v[ 0 ] = centroid_distance( a ) * v[ 0 ];
  v[ 1 ] = centroid_distance( a ) * v[ 1 ];
  v[ 2 ] = centroid_distance( a ) * v[ 2 ];
  
  // Update the output
  vadd ( v, ca2, cg );
  *radius = centroid_radius( a ); 
}//calculate_cg_atom

__forceinline__ __device__
int
cv_class_to_n ( aminoacid a ) {
  if (a==ala) return 0;
  if (a==arg) return 1;
  if (a==asn) return 2;
  if (a==asp) return 3;
  if (a==cys) return 4;
  if (a==gln) return 5;
  if (a==glu) return 6;
  if (a==gly) return 7;
  if (a==his) return 8;
  if (a==ile) return 9;
  if (a==leu) return 10;
  if (a==lys) return 11;
  if (a==met) return 12;
  if (a==phe) return 13;
  if (a==pro) return 14;
  if (a==ser) return 15;
  if (a==thr) return 16;
  if (a==trp) return 17;
  if (a==tyr) return 18;
  if (a==val) return 19;
  return -1;
}//cv_class_to_n


__forceinline__ __device__
atom_radii
get_atom_radii( int bbidx ) {
  switch ( bbidx%5 ) {
    case N:
      return rN;
    case CA:
      return rC;
    case CB:
      return rC;
    case O:
      return rO;
    case H:
      return rH;
  }
  return rC;
}//get_atom_radii

__forceinline__ __device__
void
copy_structure_from_to ( real* from_s, real* to_s ) {
  to_s[ 15*threadIdx.x      ] = from_s[ 15*threadIdx.x      ];
  to_s[ 15*threadIdx.x + 1  ] = from_s[ 15*threadIdx.x + 1  ];
  to_s[ 15*threadIdx.x + 2  ] = from_s[ 15*threadIdx.x + 2  ]; /// N
  to_s[ 15*threadIdx.x + 3  ] = from_s[ 15*threadIdx.x + 3  ];
  to_s[ 15*threadIdx.x + 4  ] = from_s[ 15*threadIdx.x + 4  ];
  to_s[ 15*threadIdx.x + 5  ] = from_s[ 15*threadIdx.x + 5  ]; /// Ca
  to_s[ 15*threadIdx.x + 6  ] = from_s[ 15*threadIdx.x + 6  ];
  to_s[ 15*threadIdx.x + 7  ] = from_s[ 15*threadIdx.x + 7  ];
  to_s[ 15*threadIdx.x + 8  ] = from_s[ 15*threadIdx.x + 8  ]; /// C
  to_s[ 15*threadIdx.x + 9  ] = from_s[ 15*threadIdx.x + 9  ];
  to_s[ 15*threadIdx.x + 10 ] = from_s[ 15*threadIdx.x + 10 ];
  to_s[ 15*threadIdx.x + 11 ] = from_s[ 15*threadIdx.x + 11 ]; /// O
  to_s[ 15*threadIdx.x + 12 ] = from_s[ 15*threadIdx.x + 12 ];
  to_s[ 15*threadIdx.x + 13 ] = from_s[ 15*threadIdx.x + 13 ];
  to_s[ 15*threadIdx.x + 14 ] = from_s[ 15*threadIdx.x + 14 ]; /// H
}//copy_structure_from_to

#endif


