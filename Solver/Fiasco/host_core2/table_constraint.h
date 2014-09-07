/*********************************************************************
 * Table Constraint
 * This is a generic constraint used to assign variable values for a large 
 * (not necessarily contiguous) set of variables, all at once.
 * 
 * The table constraint is set on a set of variables S \subseteq V and it 
 * is woken up once one of V_i \in S is selected and all other variables
 * are not ground.
 * The labeling choices for a tuple <v_i, ..., v_j> are stored in the 
 * 'table', which is stored, accessed and modified column by column.
 *********************************************************************/

#ifndef FIASCO_TABLE_CONSTRAINT_
#define FIASCO_TABLE_CONSTRAINT_

#include "constraint.h"
#include "bitset.h"
#include <vector>
#include <map>

#define NOT_INITIALIZED 999999

class TableConstraint : public Constraint {
 private:
  // nota usa una struttura dati piu' efficiente per riordinare 
  std::vector< std::vector <int> > table;
  uint _nrows, _ncols;
  std::map <int, int> variables_indexes;
  int current_row;
  
 public:
  TableConstraint (uint v_start, uint v_end, uint nrow);
  //TableConstraint (std::vector<uint> variables, uint nrow);
  void set (uint row, uint col, int value);
  void reset ();
  void reset_row (uint row);
  uint get_nrows () {return _nrows; }
  uint get_ncols () {return _ncols; }  
  void radix_sort();
  
  bool propagate(int trailtop);   // here backjump the index is needed.
  bool consistency();
  void post_backtrack_process (); // update exploration table row  
  bool check_cardinality (size_t& backjump) {return true; } 
  bool synergic_consistency (const point& p, atom_type t = CA, int aa_idx = 0);
  void reset_synergy() {};
  void dump(bool all=true);
};//-

#endif
