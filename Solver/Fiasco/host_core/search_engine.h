/*********************************************************************
 * Search Engine Object
 * Abstract search engine class.
 *********************************************************************/
#ifndef SEARCH_ENGINE_H
#define SEARCH_ENGINE_H

#include "typedefs.h"
#include <vector>
#include <iostream>

class VariableFragment; // Change to FiniteDomainVariable

class SearchEngine {
 private:
  bool _abort_search;
 
 protected:
  const size_t NO_BACKJUMPING;
  size_t backjump_trailtop;
  size_t max_numof_solutions;
  std::vector<int> curr_labeling;
  std::vector<int> next_labeling;
  
  // enum {FAILED, SOLVED, IN_PROGRESS};
  // virtual void initialize() {}
  virtual VariableFragment* variable_selection () = 0;
  virtual bool labeling (VariableFragment *v) = 0;
  virtual void process_solution () = 0;

 public:
  SearchEngine();
  virtual ~SearchEngine ();
  
  virtual void search () = 0;
  virtual void reset ();
  
  virtual void dump_statistics (std::ostream &os = std::cout) const;

  void abort ();
  bool aborted () const;
};

#endif
