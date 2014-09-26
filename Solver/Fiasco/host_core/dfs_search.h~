/*********************************************************************
 * Search Engine Object
 * This search engine implements a simple DFS to explore a 
 * prop-labelling tree.
 *
 * TODO:
 * 1. Move natom_ground in LogicVariables
 *********************************************************************/
#ifndef SEARCH_DEPTH_FIRST_SEARCH_ENGINE_H
#define SEARCH_DEPTH_FIRST_SEARCH_ENGINE_H

#include "search_engine.h"
#include "typedefs.h"
#include <vector>
#include <iostream>

class VariableFragment;

class DepthFirstSearchEngine : public SearchEngine {
 private:
  uint height;
  uint expl_level;

  void goto_next_level(const VariableFragment *v);
  void goto_prev_level(const VariableFragment *v);

protected:
  VariableFragment* variable_selection ();
  bool labeling(VariableFragment *v);
  void process_solution();

 public:
  DepthFirstSearchEngine (int argc, char* argv[]);
  ~DepthFirstSearchEngine();
  
  void reset ();
  void search ();
  void dump_statistics (std::ostream &os = std::cout);
};

#endif
