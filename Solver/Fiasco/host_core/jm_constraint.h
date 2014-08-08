/*********************************************************************
 * Joint Multibody Constraint and Filtering
 * > if front-anchor and end-anchor FIXED -> activate BIDIRECTIONAL JM 
 *   filtering
 * > if front-anchor only or end-anchor only FIXED -> activate MONODI-
 *   RECTIONAL JM filtering
 *
 * JM CONSTRAINT Syntax
 * --jm aa_s {-> | <- | <->} aa_f :
 *   numof-clusters= K_min K_max 
 *   sim-params= R B 
 *   tolerances= eps_R eps_B
 *
 * where:
 *   - aa_s: is the amino acid (starting from 0) 
 *     corresponding to the front-anchor of the 
 *     flexible chain.
 *   - aa_f: is the amino acid relative to the 
 *     end-anchor of the flexible chain.
 *   - '->' | '<-' | '<->', defines the propagation direction,
 *     either MONO- or BI-directional
 *   - K_min, K_max: are the minumum / maximum number 
 *     of cluster to generate
 *   - R, B: are the sim-clustering parameters 
 *     eps_R, eps_B: are the maximum tolerances radius 
 *     (eps_R) and orientation (eps_B) within which an 
 *     end-anchor can be placed. 
 *
 * DEFAULT VALUES:
 *   - K_min = max {50, dom-size}
 *   - K_max = 500
 *   - R = 0.5 A
 *   - B = 15 deg
 *   - eps_R = 1.5 A
 *   - eps_R = 60 deg
 *
 * EXAMPLE:
 * --jm 88 -> 96 :
 *   numof-clusters= 60 200
 *   sim-param= 0.5 15 
 *   tolerances= 1.5 30
 *
 * References: 
 * F. Campeotto, A. Dal Palu', A. Dovier, F. Fioretto, E. Pontelli. 
 * A Filtering Technique for Fragment Assembly-based Proteins Loop 
 * Modeling with Constraints. Proceedings of CP 2012.
 *********************************************************************/
#ifndef FIASCO_JM_CONSTRAINT_
#define FIASCO_JM_CONSTRAINT_

#include "typedefs.h"
#include "constraint.h"
#include "fragment.h" // for search_direction
#include "k_medoids.h"
#include "anchor.h"
#include "bitset.h"

#include <vector>

//class Anchor;
//class Bitset;
//class K_Medoids;
//struct Linked_Anchor;

class TableConstraint;

class JMConstraint : public Constraint {
 private:
  Anchor _front_anchor;
  Anchor _end_anchor;
  bool _front_anchor_is_fixed;
  bool _end_anchor_is_fixed;
  bool head_of_jm_multibody;

  real ths_anchor_distance;	/* ~cluster-params: r */
  real ths_anchor_orientation;	/* ~cluster-params: beta */

  K_Medoids* Kmedoids_alg;
  TableConstraint* table;

  std::vector< std::vector< Linked_Anchor> > clusters_left_to_right;
  std::vector< std::vector< Linked_Anchor> > clusters_right_to_left;
  std::vector<uint> clusters_left_to_right_size;
  std::vector<uint> clusters_right_to_left_size;
  std::vector<Bitset> domains_filtered;

  bool cluster_rigid_bodies (
     std::vector<std::vector<Linked_Anchor> >& linked_anchors,
     std::vector<uint>& linked_anchors_size,
     uint chain_front_aa, uint chain_end_aa, 
     AssemblyDirection direction);

  bool get_JMconsistent_rigid_bodies (
     const std::vector<std::vector<Linked_Anchor> >& linked_anchors,
     const std::vector<uint>& linked_anchors_size);

  /* void get_JMconsistent_rigid_bodies  */
  /*   (std::vector<std::vector<Linked_Anchor> >& clusters_L2R,  */
  /*    std::vector<std::vector<Linked_Anchor> >& clusters_R2L) const; */

  void dump_mem() const;

 protected:
  friend class AlldistantConstraint;
  friend class CentroidConstraint;
  friend class UniformConstraint;
  friend class EndAnchorDifferentConstraint;
  std::vector<Constraint*> synergy_constraints;
   
 public:   
  JMConstraint (int argc, char* argv[], int& parse_pos, int& scanner);
  ~JMConstraint () {delete Kmedoids_alg; }
  
  bool front_anchor_is_fixed () const {return _front_anchor_is_fixed; }
  bool end_anchor_is_fixed ()   const {return _end_anchor_is_fixed; }
  const Anchor* front_anchor () const {return &_front_anchor; }
  const Anchor* end_anchor ()   const {return &_end_anchor; }
  void add_synergy_constraint (Constraint* c) {synergy_constraints.push_back(c); }
  void flush_synergy_constraints () {synergy_constraints.clear(); }
  
  // Virtual Methods
  bool propagate (int trailtop);
  bool consistency ();
  bool check_cardinality (size_t& backjump) {return true; }
  bool synergic_consistency 
    (const point& p, atom_type t = CA, int aa_idx = 0) {return true; }
  void reset_synergy () {};
  void dump(bool all=true);

  // TMP
  bool table_done;

};
  
#endif 
