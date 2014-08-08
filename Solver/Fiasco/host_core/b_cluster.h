/*********************************************************************
 * Class: RBcluster
 * A cluster is an abstraction which is used to group a set of Anchors.
 *********************************************************************/
#ifndef FIASCO_B_CLUSTER__
#define FIASCO_B_CLUSTER__

#include "typedefs.h"
#include "anchor.h"
#include "fragment.h"

#include <vector>

class Linked_Anchor {
 public:
  Anchor representative;
  Linked_Anchor* parent;
  bool is_valid;
  
  Linked_Anchor () 
    : parent (NULL), is_valid (true) { };

  Linked_Anchor (const Anchor& a, Linked_Anchor* p = NULL) 
    : representative (a), parent (p), is_valid (true) { };

  Linked_Anchor (const Linked_Anchor& other) {
    representative = other.representative;
    parent = other.parent;
    is_valid = other.is_valid;
  }//-

  Linked_Anchor& operator= (const Linked_Anchor& other) {
    if (this != &other) {
      representative = other.representative;
      parent = other.parent;
      is_valid = other.is_valid;
    }
    return *this;
  }//-

  ~Linked_Anchor () {};

  bool operator() (const Linked_Anchor& li, const Linked_Anchor& lj) {
    if (li.parent && lj.parent) 
      return ((li.representative.of_fragment()->get_id() < lj.representative.of_fragment()->get_id()) || 
	       li.parent->representative.of_fragment()->get_id() < lj.parent->representative.of_fragment()->get_id());
    else 
      return (li.representative.of_fragment()->get_id() < lj.representative.of_fragment()->get_id());
  }//-
};

/* 
 * B-equivalence-class for the sim-parameter \beta angle.
 */
class B_Cluster {
 private: 
  std::pair<real, real> phi, theta, psi; // \pm \b_epsilon
  std::vector< Linked_Anchor > objects;
  size_t b_cluster_size;
  int representative_idx;

 public:
  B_Cluster ();
  // Rule of the 3  
  B_Cluster (const B_Cluster& other);
  B_Cluster& operator= (const B_Cluster& other);
  ~B_Cluster() {};
  
  void set_b_bounds(real _phi, real _theta, real _psi, real eps);
  std::pair<real, real> get_phi   () const;
  std::pair<real, real> get_theta () const;
  std::pair<real, real> get_psi   () const;
  bool is_valid () const;
  void insert (const Anchor &obj, Linked_Anchor* parent);  
  void insert (const Linked_Anchor &obj);

  const Anchor* get_representative() const;
  Linked_Anchor* get_parent_representative() const;
  size_t size () const;
  size_t capacity () const;
  void incr_size(size_t n = 1);
  void decr_size(size_t n = 1);
  void clear ();
  void erase ();

};
//-

struct cmp_linked_anchor
  : public std::binary_function <Linked_Anchor, Linked_Anchor, bool> {
  bool operator () (const Linked_Anchor& lhs, 
		    const Linked_Anchor& rhs) const { 
    return (lhs.representative.of_fragment()->get_probability() >=
	    rhs.representative.of_fragment()->get_probability());
  }
};
//-


struct cmp_linked_anchor_id
  : public std::binary_function <Linked_Anchor, Linked_Anchor, bool> {
  bool operator () (const Linked_Anchor* _lhs, const Linked_Anchor* _rhs) const { 
    
    Linked_Anchor* lhs = const_cast<Linked_Anchor*> (_lhs);
    Linked_Anchor* rhs = const_cast<Linked_Anchor*> (_rhs);
    
    while (lhs->parent && rhs->parent) {
      int lhs_id = lhs->representative.of_fragment()->get_id();
      int rhs_id = rhs->representative.of_fragment()->get_id();
      if (lhs_id != rhs_id) {
        return lhs_id < rhs_id;
      }
      else {
        lhs = lhs->parent;
        rhs = rhs->parent;
      }
    }
    
    return (lhs->representative.of_fragment()->get_id() <= 
	    rhs->representative.of_fragment()->get_id());
  }
};
//-



#endif
