/*********************************************************************
 * Flexible Sequence First
 * This search engine implements a  DFS to explore a prop-labelling 
 * tree with a variable selection strategy optimzed to loop closure.
 *
 *********************************************************************/
#ifndef FLEXSEQ_FIRST_SEARCH_H
#define FLEXSEQ_FIRST_SEARCH_H

#include "search_engine.h"
#include "search.h"
#include <vector>
#include <iostream>

class VariableFragment;
class Loop;

class FlexSeqFirstSearch : public SearchEngine {
 private:
  uint height;
  uint expl_level;

  void goto_next_level(const VariableFragment *v);
  void goto_prev_level(const VariableFragment *v);

protected:
  VariableFragment* select_flex_variable (const Loop* loop);
  const Loop* select_flexible_sequence ();

  VariableFragment* variable_selection ();
  bool labeling(VariableFragment *v);
  void process_solution();

 public:
  FlexSeqFirstSearch (int argc, char* argv[]);
  ~FlexSeqFirstSearch();
  
  void reset ();
  void search ();
  void dump_statistics (std::ostream &os = std::cout);
};

#endif
