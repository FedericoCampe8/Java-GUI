/*********************************************************************
 * Class: R_Cluster
 * A cluster is an abstraction which is used to group a set of Anchors.
 *********************************************************************/
#ifndef FIASCO_R_CLUSTER__
#define FIASCO_R_CLUSTER__

#include "typedefs.h"
#include "rb_cluster.h"

#include <set>
#include <vector>

class Anchor;
class RB_Cluster;
class B_Cluster;
class Linked_Anchor; // defined in b_cluster.h

/* The elements of an r_clusters are Anchors for wich their centroids are 
 * within 2r Angstroms from the Anchor representative.
 * The elements of an r_cluster are futher grouped in b_clusters. A b_cluster
 * is a cluster which groups anchors with a similar orientation plane, 
 * with respect to the initial reference system.
 */
class R_Cluster {
 private:
  std::vector<B_Cluster> r_cluster;
  std::vector<uint> active_b_clusters;
  size_t representative_idx;
  bool _is_valid;

 protected:
  RB_Cluster* linked_rb_cluster;
  size_t find_bin (real phi, real theta, real psi) const;

 public:
  R_Cluster (RB_Cluster* rb_cluster, uint numof_bins=1);
  // Rule of the 3
  R_Cluster (const R_Cluster& other);
  R_Cluster& operator= (const R_Cluster& other);	    
  ~R_Cluster();

  bool operator () (const R_Cluster& lhs, const R_Cluster& rhs) const; 
  void insert(const Anchor& obj, Linked_Anchor* parent = NULL);
  void insert(const Linked_Anchor& obj);
  void set_valid (bool b = true);
  bool is_valid () const;
  size_t size();
  size_t capacity();
  void clear();
  void erase();
  const Anchor* get_representative() const;
  Linked_Anchor* get_parent_representative() const;
  const Anchor* get_b_representative(uint b) const;
  Linked_Anchor* get_parent_b_representative(uint b) const;
};
//-

#endif
