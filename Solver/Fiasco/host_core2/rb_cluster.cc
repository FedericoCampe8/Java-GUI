#include "rb_cluster.h"
#include "r_cluster.h"
#include "b_cluster.h"
#include "mathematics.h"

#include <cassert>
#include <iterator>
#include <cmath>

// allocate k_max objects
RB_Cluster::RB_Cluster (uint max_clusters, real _radius, real _beta) 
  : clusters_size(0), radius (_radius), beta (_beta), numof_bins (ceil(Math::CONST_2PI / beta)) 
{
  clusters.resize (max_clusters, R_Cluster(this, numof_bins));
}//-


/*  
 * An element is inserted into an existing bucket if its centroid is within a distance 
 * of at most 2r.
*/
void
RB_Cluster::insert (const Anchor& obj, Linked_Anchor* parent, int pos) {
  if (pos < 0) {
    assert (clusters_size <= clusters.size());
    // std::cout <<"RB_Cluster::insert R_cluster[ " <<  clusters_size 
    // 	      << "]" << " size: " << clusters[clusters_size].size()
    // 	      << " capacity: " << clusters[clusters_size].capacity() << std::endl;
    clusters[clusters_size++].insert(obj, parent);
  }
  else {
    // std::cout << "RB_cluster inserting in pos:" << pos << std::endl;
    clusters[pos].insert(obj, parent);
  }
}//-

void
RB_Cluster::insert (const Linked_Anchor &obj, int pos) {
  this->insert (obj.representative, obj.parent, pos);
}//-

void 
RB_Cluster::clear() {
  for (int i=0; i < clusters_size; i++) 
    clusters[i].clear();
  clusters_size = 0;
}//-

void 
RB_Cluster::erase() {
  for (int i=0; i < clusters_size; i++) 
    clusters[i].erase();
  clusters_size = 0;
}

R_Cluster* 
RB_Cluster::get_r_cluster (uint pos) {
  if (pos < clusters_size)
    return &(clusters[pos]);
  return NULL;
}//-
