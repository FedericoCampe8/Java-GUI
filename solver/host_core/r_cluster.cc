#include "r_cluster.h"
#include "b_cluster.h"
#include "anchor.h"

#include <algorithm>
#include <cassert>
#include <iterator>
#include <cmath>

R_Cluster::R_Cluster(RB_Cluster* rb_cluster, uint bins)
  : representative_idx (0), _is_valid (true), 
    linked_rb_cluster (rb_cluster)
{
  r_cluster.resize (bins*bins*bins);
}//-

R_Cluster::R_Cluster (const R_Cluster& other) {
  r_cluster = other.r_cluster;
  representative_idx = other.representative_idx;
  active_b_clusters = other.active_b_clusters;
  _is_valid = other._is_valid;
  linked_rb_cluster = other.linked_rb_cluster;
}//-

R_Cluster& 
R_Cluster::operator= (const R_Cluster& other) {
  if (this != &other) {
    r_cluster = other.r_cluster;
    representative_idx = other.representative_idx;
    active_b_clusters = other.active_b_clusters;
    _is_valid = other._is_valid;
    linked_rb_cluster = other.linked_rb_cluster;
  }
  return *this;
}//-

R_Cluster::~R_Cluster() {}//-

bool
R_Cluster::operator () (const R_Cluster& lhs, const R_Cluster& rhs) const { 
  return (lhs.get_representative()->associated_fragment_idx() <
	  rhs.get_representative()->associated_fragment_idx());
}//-

void
R_Cluster::insert (const Anchor& obj, Linked_Anchor* parent) {
  size_t bin = 
    find_bin (obj.get_phi(), obj.get_theta(), obj.get_psi());
  // std::cout <<"R_Cluster::insert R_cluster[ " <<  bin 
  // 	    << "]" << " size: " << r_cluster[bin].size()
  // 	    << " capacity: " << r_cluster[bin].capacity() << std::endl;

  // THIS IS ONLY FOR TEST -- NEED A MORE EFFICIENT DS -- SET? OR
  // VECTOR WITH BINARY SEARCH
  std::vector<uint>::iterator it;
  it = std::find (active_b_clusters.begin(), active_b_clusters.end(), bin);
  if (it == active_b_clusters.end()) {
    r_cluster[bin].insert(obj, parent);
    active_b_clusters.push_back(bin);
    // set cluster representative, if not set yet
    if (!representative_idx)
      representative_idx = bin;
  }
  

}//-

void
R_Cluster::insert (const Linked_Anchor& obj) {
  R_Cluster::insert (obj.representative, obj.parent);
}//-


void 
R_Cluster::set_valid (bool b) {
  _is_valid = b;
}//-

bool 
R_Cluster::is_valid () const {
  return _is_valid;
}//-

size_t
R_Cluster::size() {
  return active_b_clusters.size();
}//-

size_t
R_Cluster::capacity () {
  return r_cluster.size();
}//-

void 
R_Cluster::clear() {
  std::vector<B_Cluster>::iterator it;
  for (it = r_cluster.begin(); it != r_cluster.end(); ++it) {
    it->clear();
  }
  active_b_clusters.clear();
  representative_idx = 0;
  _is_valid = true;
}//-

void 
R_Cluster::erase() {
  std::vector<B_Cluster>::iterator it;
  for (it = r_cluster.begin(); it != r_cluster.end(); ++it) {
    it->erase();
  }
  r_cluster.clear();
  active_b_clusters.clear();
  representative_idx = 0;
  _is_valid = true;
}//-

const Anchor*
R_Cluster::get_representative() const {
  // std::cout << "R_Cluster::get_representative[" 
  // 	    << representative_idx << "]\n";
  return (r_cluster[representative_idx].get_representative());
}//-

Linked_Anchor* 
R_Cluster::get_parent_representative() const {
  // std::cout << "R_Cluster::get_parent_representative[" 
  // 	    << representative_idx << "]\n";
  return (r_cluster[representative_idx].get_parent_representative());
}//-

const Anchor*
R_Cluster::get_b_representative (uint b) const {
  assert(b <= active_b_clusters.size());
  return (r_cluster[active_b_clusters[b]].get_representative());
}//-

Linked_Anchor* 
R_Cluster::get_parent_b_representative(uint b) const {
  assert(b <= active_b_clusters.size());
  return (r_cluster[active_b_clusters[b]].get_parent_representative());
}//-

/* 
 * utility function:
 * Finds the bin index in R/2pi given the euleran angles 
 * in radiant and the beta-epsilon parameter
 */
size_t 
R_Cluster::find_bin (real phi, real theta, real psi) const {
  size_t numof_bins = Math::CONST_2PI / linked_rb_cluster->get_beta();
  return (phi * (numof_bins * numof_bins) +
	  theta * (numof_bins) +
	  psi);
}//-
//------------------------------------------------
