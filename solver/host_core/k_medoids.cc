#include "k_medoids.h"
#include "globals.h"

#include "jm_constraint.h"
#include "anchor.h"
#include "rb_cluster.h"
#include "r_cluster.h"
#include "b_cluster.h"
#include "statistics.h"

#include <algorithm>
#include <cassert>
#include <iostream>
#include <sstream>
#include <set>
#include <map>
#include <vector> 
#include <limits>

#include <stdio.h>
using namespace std;

// The cluster parameters where:
// [kmin, kmax] represent the minimum and maximum number of clusters 
//   that may be created at each run.
// r is the maximum radius of the sphere containing points in a cluster
// b is the angle in radiant for the partitions within an r-cluster.
K_Medoids::K_Medoids (int kmin, uint kmax, real r, real b) :
  k_min (kmin), k_max (kmax), radius (r), beta (b) {
  rb_clusters = new RB_Cluster(kmax, r, b);
}//-

K_Medoids::~K_Medoids () {
  rb_clusters->clear();
  delete rb_clusters;
}//-

void 
K_Medoids::set_numof_clusters(uint _k_min, uint _k_max) {
  k_min = _k_min; 
  k_max = _k_max;
}//-

void 
K_Medoids::set_radius(real r) {
  radius = r;
}//-

void 
K_Medoids::set_beta (real b) {
  beta = b;
}//-

std::pair<uint, uint> 
K_Medoids::get_numof_clusters (){
  return std::make_pair (k_min, k_max); 
}//-

real 
K_Medoids::get_radius () const {
  return radius; 
}//-

real 
K_Medoids::get_beta () const {
  return beta; 
}//-


/* 
 * Create a set of clusters starting from a set of anchors, placing
 * each anchor in a different cluster, but at the same time computing
 * the correct beta id for the equivalence function of the clusters.
 */
void
K_Medoids::get_linked_anchors
  (std::vector<Linked_Anchor>& linked_anchors,  // out 
   uint& numof_linked_anchors,	      // out
   const set_of_anchors_in_cluster& anchors_to_cluster)	// in
{
  string dbg = "K_medoids::anchors_to_clusters - ";
  int beta = 0;
  set_of_anchors_in_cluster::iterator it;
  for (it = anchors_to_cluster.begin(); it != anchors_to_cluster.end(); ++it, beta++) { 
    linked_anchors[beta].representative = it->representative;
    linked_anchors[beta].parent = it->parent;
  }
  numof_linked_anchors = beta;
}//-

/*
 * The vector<ClusterOrientation> structure represent the product 
 * of clustering acting on positions and angles - namely, a 
 * ClusterOrientation is a map containing fragments belonging to 
 * same orientation bucket. The vector of Cluster Orientation represent
 * the intersection of the positional clustering with the orientation 
 * clustering. 
 * In other words, we first cluter by spatial positions (vector of 
 * ClusterOrientation NULL) hence we group together fragments with same
 * ClusterOrientation (shrinking the size of the vector).
 * clustering of each cluster
 * TODO: Limit the size expansion of the output vector of cluster
 */
void
K_Medoids::rb_cluster_to_linked_anchors 
  (std::vector<Linked_Anchor>& linked_anchors,  // out 
   uint& numof_linked_anchors) 	        // out
{
  string dbg = "cluster_utils::orientation_buckets_to_clusters - ";
  
  // this will limit the set of clusters taken to k_max
  uint b_limit = (int)(k_max / rb_clusters->size());
  uint cluster_idx =0, b_size = 0, b_incr = 0;

  for (uint r = 0; r < rb_clusters->size(); r++) {
    b_size = rb_clusters->get_r_cluster(r)->size();
    b_incr = Math::max (1, (int)(b_size/b_limit));
    
    for (uint b = 0, c = 0; b < b_size && c < b_limit; b += b_incr, c++) {
      linked_anchors[cluster_idx].representative = 
	*(rb_clusters->get_r_cluster(r)->get_b_representative(b));
      linked_anchors[cluster_idx].parent =
	rb_clusters->get_r_cluster(r)->get_parent_b_representative(b);
      cluster_idx++;
    }
  }
  numof_linked_anchors = cluster_idx;
}
//-

void
K_Medoids::make_clusters 
  (vector<Linked_Anchor>& linked_anchors,     // out
   uint& numof_linked_anchors,	      // out
   set_of_anchors_in_cluster& anchors_to_cluster)
{
  string dbg = "K_medoids::cluster() - ";
  
  // Not enought end-anchors
  if (anchors_to_cluster.size() <= k_min) { 
    //   cout << dbg << "not enouth end anchors\n";
    get_linked_anchors (linked_anchors, numof_linked_anchors, anchors_to_cluster); 
  }

  /* 
   * Select the first K-medoids from the set fo anchors.
   * The medoids are selected only if their centroid is separated by 
   * a distance of at least 2r.
   */
  uint jump = (anchors_to_cluster.size() / k_min);
  int count = 0;
  // cout << "anchor_to cluster sizE: " << anchors_to_cluster.size()
  //      << "jump: " << jump << endl;

  point cluster_anchor_cg, anchor_it_cg;
  while (!anchors_to_cluster.empty() && rb_clusters->size() < k_min) {

    set_of_anchors_in_cluster::iterator 
      anchor_it = anchors_to_cluster.begin();
    const Anchor *anchor = &((*anchor_it).representative);
    
    // check clusters are within distance 2r
    anchor->get_centroid (anchor_it_cg);
    for (size_t r = 0; r < rb_clusters->size(); r++) {
      const Anchor *cluster_anchor = rb_clusters->get_r_cluster(r)->get_representative(); 
      cluster_anchor->get_centroid (cluster_anchor_cg);

      real cg_distance = Math::eucl_dist (cluster_anchor_cg, anchor_it_cg);
      if (cg_distance <= 2*radius) {
	continue;
      }
    }//-

    // Create a new medoid
    rb_clusters->insert (*anchor_it);

    // remove anchor from the input anchor set
    set_of_anchors_in_cluster::iterator anchor_to_erase = anchor_it;
    if (rb_clusters->size() >= k_min) {
      anchors_to_cluster.erase (anchor_to_erase);	
      break;
    }
    // Jump of N/k positions in the anchor set
    for (uint pos=0; pos < jump; pos++, anchor_it++, count++) {
      if (anchor_it == anchors_to_cluster.end()) {
	anchor_it = anchors_to_cluster.begin(); // restart!
	break;			     
      }
    }
    // (*anchor_to_erase).second = NULL;
    anchors_to_cluster.erase (anchor_to_erase);	
  }
  //-
  //  std::cout << dbg << " AFTER FIRST STEP rb_clusters: " << rb_clusters->size() << std::endl;

  /* 
   * If the numer of medoids distant at least 2r each other, is 
   * smaller then k_min, we return the set of 'clusters' generated
   * so far --> anchor_to_cluster is empty
   */
  if (rb_clusters->size() < k_min) {
    // std::cout << dbg << " not enough points wrt to clusters\n ";
    rb_cluster_to_linked_anchors (linked_anchors, numof_linked_anchors);
    // std::cout << dbg << " AFTER FIRST STEP rb_clusters: " << rb_clusters->size() << std::endl;
  }
  //-

  /*
   * All the remaining anchors are placed in the correct buckets.
   * If an anchor does not fall in any cluster, we create a new one. 
   * This step undergo untill the size of the cluster vector 
   * reaches k_max. If the size of the cluster vector exceeds k_max 
   * all other anchors not satisfying the 2*r distance requirement 
   * are placed in the cluster whom representant minimizes the distance 
   * between the anchor cecntroids.
   */
  set_of_anchors_in_cluster::iterator anchor_it = anchors_to_cluster.begin();
  for (; anchor_it != anchors_to_cluster.end(); anchor_it++) {

    const Anchor *anchor = &((*anchor_it).representative); 
    pair<int, real> best_cluster = make_pair (0, 100000);    // <cluster_index, distance> 
   
    // Get the closest representative to current end-anchor to cluster
    anchor->get_centroid (anchor_it_cg);
    for (uint r = 0; r < rb_clusters->size(); r++) {
      const Anchor* cluster_anchor = rb_clusters->get_r_cluster(r)->get_representative();
      cluster_anchor->get_centroid (cluster_anchor_cg);
      real distance = Math::eucl_dist (anchor_it_cg, cluster_anchor_cg);

      if (distance < best_cluster.second) {
	best_cluster.second = distance;
	best_cluster.first = r;
      }
    }
    
    // If distance is too far away, make a new cluster 
    if (best_cluster.second > 2*radius && rb_clusters->size() < k_max) {
      rb_clusters->insert (*anchor_it);
    }
    else {
      // Insert the anchor in the designated orientation bucket
      rb_clusters->insert (*anchor_it, best_cluster.first);
    }
  }
  //-

  //  std::cout << dbg << " AFTER K-medioindg rb_clusters: " << rb_clusters->size() << std::endl;
  // Copy all the orientation clusters in the correct (output) bucktes
  rb_cluster_to_linked_anchors  (linked_anchors, numof_linked_anchors);
  // dump_mem();
  // std::cout << dbg << "AFTER CLUSTER-TO_LINK anchor size returned: " << numof_linked_anchors << std::endl;
  // getchar();

  rb_clusters->clear();


}//-


void K_Medoids::dump_mem() {
  std::cout << "=== MEMORY ALLOCATION IN K-MEDOIDS ===\n";
  std::cout << "RB_cluster["<< rb_clusters->size() << " / " 
	    << rb_clusters->capacity() << "]\n";
  
  for (int j=0; j<rb_clusters->size(); j++) {
    R_Cluster* rcluster = rb_clusters->get_r_cluster(j);
    std::cout << "R_cluster["<<j<<"]: " 
	      << rcluster->size() << " / " 
	      << rcluster->capacity() << std::endl;
  }
  std::cout << "---------------------------------------\n";
  
}//-
