/*********************************************************************
 * This search engine implements a Docking sampling of the
 * search space. Docking is performed by MonteCarlo sampling.
 *********************************************************************/
#ifndef COCOS_DOCKING__
#define COCOS_DOCKING__

#include "montecarlo.h"

class DOCKING : public MONTECARLO {
private:
  /// Coordinates of the center of the cube
  real _center_x;
  real _center_y;
  real _center_z;
  real _radius; // Angstrom
  real _side;
  real _oc_tree_height;
  real _oc_tree_side;
  std::vector <  std::vector < real > > _centers_coords;
  /// Energy value
  real _energy_value;
  /// Other and info
  size_t _n_seeds;
  size_t _n_total_sols;
  
  /// Printing options
  bool _to_print;
  real _failed_constraints;
  
  void print_str_step_by_step ( int );
  bool verify_conditions      ( int );
  
public:
  DOCKING ( MasAgent* mas_agt );
  ~DOCKING ();
  void search ();
  
  int choose_label ( WorkerAgent* w );
  
  /// Set the parameters for the centroid of the cube to subdivide as an oc_tree
  void set_parameters ( std::vector < std::vector < real > >& coords );
  void set_parameters ( real x, real y, real z, real radius, real height = 4 );
};

#endif
