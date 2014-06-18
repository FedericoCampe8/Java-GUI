/*********************************************************************
 * K Medoids
 * The K Medoids object implements the k-means clustering on the 
 * anchor space by taking account of anchor positions and angles
 * given by the plane N-C-O associated to each anchor.
 *********************************************************************/
#ifndef FIASCO_K_MEDOIDS__
#define FIASCO_K_MEDOIDS__

#include "typedefs.h"
#include "b_cluster.h"

#include <vector>
#include <set>

class Anchor;
class RB_Cluster;

/* These structures are used to create clusters w.r.t. end-anchor
 * orientation. The result will be integrated to the clustering
 * w.r.t. radius distance.
 */
struct Linked_Anchor; // defined in b_cluster.h
struct cmp_linked_anchor;             // defined in b_cluster.h
typedef std::set<Linked_Anchor, cmp_linked_anchor> 
  set_of_anchors_in_cluster;

class K_Medoids {
 private:
  uint k_min; // minumum number of clusters
  uint k_max; // maximum number of clusters
  real radius;
  real beta;
  RB_Cluster* rb_clusters;

  void dump_mem();

 public:
  K_Medoids (int kmin, uint kmax, real r, real b);
  ~K_Medoids();

  void set_numof_clusters(uint _k_min, uint _k_max);
  void set_radius(real r);
  void set_beta (real b);
  std::pair<uint, uint> get_numof_clusters ();
  real get_radius () const;
  real get_beta () const;

  void get_linked_anchors
    (std::vector<Linked_Anchor>& linked_anchors,	  /* out */
     uint& numof_linked_anchors,			  /* out */
     const set_of_anchors_in_cluster& anchors_to_cluster); /* in */

  void rb_cluster_to_linked_anchors 
    (std::vector<Linked_Anchor>& linked_anchors, /* out */
     uint& numof_linked_anchors);		 /* out */

  void make_clusters 
    (std::vector<Linked_Anchor>& linked_anchors, /* out */
     uint& numof_linked_anchors,		 /* out */
     set_of_anchors_in_cluster& anchors_to_cluster); 
};
//-  

#endif
