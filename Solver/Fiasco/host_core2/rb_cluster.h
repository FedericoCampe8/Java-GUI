/*********************************************************************
 * Class: RBcluster
 * A cluster is an abstraction which is used to group a set of Anchors.
 *********************************************************************/
#ifndef FIASCO_RB_CLUSTER__
#define FIASCO_RB_CLUSTER__

#include "typedefs.h"

#include <iostream>
#include <vector>

class Anchor;
class R_Cluster;
class Linked_Anchor;

class RB_Cluster {
 private:
  std::vector<R_Cluster> clusters;
  size_t clusters_size;
  real radius; // the maximum radius size for a cluster
  real beta;   // the beta-cluster angle tollerance in radiants 
  uint numof_bins;

 public:
  RB_Cluster (uint max_clusters, real _radius, real _beta);

  void insert (const Anchor& obj, Linked_Anchor* parent, int pos=-1);
  void insert (const Linked_Anchor &obj, int pos=-1);

  R_Cluster* get_r_cluster (uint pos);
  
  void clear();
  void erase();
  size_t size () const {return clusters_size; }
  size_t capacity() const {return clusters.size(); }
  uint get_numof_bins () {return numof_bins; }
  real get_beta() {return beta; }
  real get_radius() {return radius; }
};
  
#endif
