#include "b_cluster.h"
#include "mathematics.h"

#include <iostream>
#include <utility>

using namespace std;

B_Cluster::B_Cluster ()
  : phi(make_pair(0,0)), theta(make_pair(0,0)), psi(make_pair(0,0)), 
    b_cluster_size(0), representative_idx(0) { 
}//-

B_Cluster::B_Cluster (const B_Cluster& other) {
  phi   = other.phi;
  theta = other.theta;
  psi   = other.psi;
  representative_idx = other.representative_idx;
  objects = other.objects;
  b_cluster_size = other.b_cluster_size;
}//-

B_Cluster&
B_Cluster::operator= (const B_Cluster& other) {
  if (this != &other) {
    phi   = other.phi;
    theta = other.theta;
    psi   = other.psi;
    representative_idx = other.representative_idx;
    objects = other.objects;
    b_cluster_size = other.b_cluster_size;
  }
  return *this;
}//-

void 
B_Cluster::set_b_bounds 
  (real _phi, real _theta, real _psi, real eps) {
  phi.first  = _phi - Math::abs(eps);
  phi.second = _phi + Math::abs(eps);
  theta.first  = _theta - Math::abs(eps);
  theta.second = _theta + Math::abs(eps);
  psi.first  = _psi - Math::abs(eps);
  psi.second = _psi + Math::abs(eps);
}//-

std::pair<real, real>
B_Cluster::get_phi() const {
  return phi;
}//-

std::pair<real, real>
B_Cluster::get_theta() const {
  return theta;
}//-

std::pair<real, real>
B_Cluster::get_psi() const {
  return psi;
}//-

bool
B_Cluster::is_valid() const {
  return (b_cluster_size > 0);
}//-

/*
 * The check for the presence/absence of an anchor in the b-bucket is 
 * not performed -- NOT USEFUL as we ensure the buckets to be flushed
 * at the end of the clustering step.
 */
void 
B_Cluster::insert (const Anchor &obj, Linked_Anchor* parent) {
  if (b_cluster_size < objects.size()) {
    // std::cout << "B_Cluster::insert by copy\n";
    objects[b_cluster_size] = Linked_Anchor (obj, parent);
  }
  else {
    // std::cout << "B_Cluster::insert by pushing\n"; 
    objects.push_back (Linked_Anchor (obj, parent));
  } 
  b_cluster_size++;
  // update representative based on occurrence probability measure
  if (obj.of_fragment()->get_probability() > 
      objects[representative_idx].representative.of_fragment()->get_probability())
    representative_idx = b_cluster_size-1;
}//-

void 
B_Cluster::insert (const Linked_Anchor &obj) {
  insert (obj.representative, obj.parent);
}//-

const Anchor*
B_Cluster::get_representative () const {
  // std::cout << "B_Cluster::get_representative - b_cluster_size: " << b_cluster_size
  // 	    << " repr: " << representative_idx << std::endl;
  
  return (b_cluster_size > 0 ) ? 
    &(objects[representative_idx].representative)
    : NULL;
}//-

Linked_Anchor*
B_Cluster::get_parent_representative () const {
  return (b_cluster_size > 0 ) ? 
    objects[representative_idx].parent 
    : NULL;
}//-

size_t 
B_Cluster::size () const {
  return b_cluster_size;
}//-

size_t 
B_Cluster::capacity () const {
  return objects.size();
}//-

void 
B_Cluster::incr_size(size_t n) {
  b_cluster_size += n; 
}//-

void 
B_Cluster::decr_size(size_t n ) {
  b_cluster_size -= n; 
}//-

void
B_Cluster::clear () {
  b_cluster_size = 0; 
  representative_idx = 0;
}//-

void
B_Cluster::erase () { 
  objects.clear();
  B_Cluster::clear();
}//-
