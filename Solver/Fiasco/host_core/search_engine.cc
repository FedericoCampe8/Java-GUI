#include "search_engine.h"
#include "globals.h"
#include "statistics.h"
#include "output.h"

#include <limits>
#include <sstream>
#include <stdlib.h> // for atoi


SearchEngine::SearchEngine() 
  : _abort_search(false), 
    NO_BACKJUMPING (std::numeric_limits<size_t>::max()),
    backjump_trailtop (NO_BACKJUMPING) {
}//-

SearchEngine::~SearchEngine () {
}//-

void
SearchEngine::reset () {
  _abort_search = false;
  backjump_trailtop = NO_BACKJUMPING;
}//-

void 
SearchEngine::abort () {
  g_statistics->stopwatch (t_statistics);
  g_output->dump();
  // g_statistics->dump();
  g_statistics->set_timer (t_statistics);
  _abort_search = true;
}//-

bool 
SearchEngine::aborted () const {
  return _abort_search; 
}//-

void
SearchEngine::dump_statistics (std::ostream &os) const {
}//-
