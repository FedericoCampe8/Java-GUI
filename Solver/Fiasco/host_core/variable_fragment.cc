#include "variable_fragment.h"
#include "globals.h"
#include "logic_variables.h"
#include "variable_point.h"
#include "utilities.h"
#include "bitset.h"

#include <cassert>
#include <cmath>

using namespace std;
using namespace Utilities;


domain_frag_info::domain_frag_info() :
  frag_mate_idx (-1), explored (false) {
}//-

domain_frag_info::domain_frag_info (const domain_frag_info& other) {
  frag_mate_idx = other.frag_mate_idx;
  explored = other.explored;
  frag_mate_info.resize(other.frag_mate_info.size());
  for (uint i = 0; i < frag_mate_info.size(); i++) {
    frag_mate_info[i].first  = other.frag_mate_info[i].first;
    frag_mate_info[i].second = other.frag_mate_info[i].second;
  }
}//-

domain_frag_info& 
domain_frag_info::operator= (const domain_frag_info& other) {
  if (this != & other) {
    frag_mate_idx = other.frag_mate_idx;
    explored = other.explored;
    frag_mate_info.resize(other.frag_mate_info.size());
    for (uint i = 0; i < frag_mate_info.size(); i++) {
      frag_mate_info[i].first  = other.frag_mate_info[i].first;
      frag_mate_info[i].second = other.frag_mate_info[i].second;
    }
  }
  return *this;
}//-


VariableFragment::VariableFragment(uint vidx, std::vector<Fragment> D) {
  domain = D;
  domain_info.resize(D.size());
  labeled      = -1;
  ground       = false;
  failed       = false;
  changed      = false;
  last_trailed = 0;
  _assembly_direction = LEFT_TO_RIGHT;
  idx = vidx;
}//-

VariableFragment::VariableFragment(uint vidx) {
  idx = vidx;
  labeled      = -1;
  ground       = false;
  failed       = false;
  changed      = false;
  _assembly_direction = LEFT_TO_RIGHT;
  last_trailed = 0;
}//-

VariableFragment::VariableFragment(const VariableFragment& other){
    idx          = other.idx;
    labeled      = other.labeled;
    ground       = other.ground;
    failed       = other.failed;
    changed      = other.changed;
    _assembly_direction = other._assembly_direction;

    last_trailed = other.get_last_trailed();

    domain       = other.domain;
    domain_info  = other.domain_info;
    constr_dep   = other.constr_dep;
}//-

// Hack! This is expensive!
VariableFragment&
VariableFragment::operator=(const VariableFragment& other){
  if (this != &other) {
    idx          = other.idx;
    labeled      = other.labeled;
    ground       = other.ground;
    failed       = other.failed;
    changed      = other.changed;
    last_trailed = other.last_trailed;
    _assembly_direction = other._assembly_direction;
    
    domain       = other.domain;
    domain_info  = other.domain_info;
    constr_dep   = other.constr_dep;
  }
  return *this;
}//-

Fragment
VariableFragment::operator[] (uint index) const {
  assert (index < domain.size());
  return domain[index];
}//-

Fragment
VariableFragment::at (uint index) const {
  assert (index < domain.size());
  return domain[index];  
}//-

void 
VariableFragment::reset () {
  labeled      = -1;
  ground       = false;
  failed       = false;
  changed      = false;
  last_trailed = 0;
  reset_domain();
  _assembly_direction = LEFT_TO_RIGHT;
  // Constraint dependencies should not be removed. 
  // Otherwise, the constraints will loose their
  // pointer to the current variable

  // constr_dep.clear();

}//-

void 
VariableFragment::add_domain_elem (const Fragment& f) {
  domain.push_back(f); 
  domain_frag_info e; 
  e.frag_mate_idx = 0;
  e.explored = false;
  domain_info.push_back(e);
}//-

void
VariableFragment::rm_domain_elem (uint i) {
  if (i<domain.size()) {
    domain.erase (domain.begin()+i);
    domain_info.erase(domain_info.begin()+i);
  }
}//-

void 
VariableFragment::set_domain(const vector<domain_frag_info>& d) {
  assert (d.size() == domain_info.size());
  for (uint i=0; i<domain_info.size(); i++)
    domain_info[i] = d[i];
}//-

void 
VariableFragment::set_domain_explored(const Bitset& d) {
  for (uint i=0; i<domain_info.size(); i++)
    domain_info[i].explored = d[i];
}//-

void 
VariableFragment::set_domain_singleton(uint d) {
  for (uint i=0; i<domain_info.size(); i++)
    domain_info[i].explored = true;  
  domain_info[d].explored = false;
}//-

void 
VariableFragment::set_domain_explored(const vector<bool>& d) {
  assert (d.size() == domain_info.size());
  for (uint i=0; i<domain_info.size(); i++)
    domain_info[i].explored = d[i];
}//-

void 
VariableFragment::get_domain_explored(vector<bool>& d) const{
  if (d.empty())
    d.resize(domain_info.size());
  assert (d.size() == domain_info.size());
  for (uint i=0; i<domain_info.size(); i++)
    d[i] = domain_info[i].explored;
}//-

bool 
VariableFragment::is_domain_explored (uint idx) const {
  return domain_info[idx].explored; 
}//-

size_t
VariableFragment::domain_size () const {
  return domain.size(); 
}//-

void
VariableFragment::reset_domain() {
  for (uint i=0; i<domain_info.size(); i++) {
    domain_info[i].explored = false;
    domain_info[i].frag_mate_idx = 0;
  }
}//-

frag_info
VariableFragment::get_domain_elem(uint i) const {
  assert (i<domain_info.size());
  int idx = domain_info[i].frag_mate_idx;
  return domain_info[i].frag_mate_info[idx];
}//-

void
VariableFragment::set_labeled(int i) {
  assert(i<(int)domain_info.size());
  labeled = i; 
  if(i>=0) {
    domain_info[i].frag_mate_idx = domain_info[i].frag_mate_info.size()-1;
  }
}//-

void
VariableFragment::reset_label () { 
  labeled = -1;
  for (uint i=0; i<domain_info.size(); i++) {
    domain_info[i].frag_mate_idx = -1;
  }
}//-

void 
VariableFragment::skip_label(uint i) {
  assert(i<domain_info.size());
  domain_info[i].explored = true;
}//-

bool 
VariableFragment::labeling() {
  string dbg = "VariableFragment::labeling() - "; 
  if (labeled > (int)domain_info.size()-1) { // was >= WHY? --- check carefully!! (Dec. 2012)
    return false;
  }

  if (labeled == -1) {
    labeled = 0;
    if (!domain_info[0].explored) {
      domain_info[labeled].explored = true;
      return true;
    }
  }

  while (domain_info[labeled].explored) {
    labeled++;
    if (labeled > (int)domain_info.size()-1) {
      return false;
    }
  }
  
  // check boundle relation
  if (is_in_bundle() &&
      domain_info[labeled].frag_mate_idx < 
      (int)domain_info[labeled].frag_mate_info.size()-1) {
    domain_info[labeled].frag_mate_idx++;  

    if (domain_info[labeled].frag_mate_idx ==
	(int)domain_info[labeled].frag_mate_info.size()-1)
      domain_info[labeled].explored = true;

    return true;
  }
  else { // current labeling 
    domain_info[labeled].explored = true;
    return true;
  }
  return false;
}//-

int 
VariableFragment::get_label () const {
  return labeled;
}//-

int 
VariableFragment::get_next_label() const {
  // all domain elements already explored
  if (labeled > (int)domain_info.size()-1)   // again it was >= WHY?? check!!
    return -1;
  
  int next = labeled;
    
  if (next == -1)
    next = 0;
  
  while (domain_info[next].explored) { 
    next++;
    if (next > (int)domain_info.size()-1)
      return -1;  
  }
  return next;

}//-

void 
VariableFragment::set_ground () {
  ground = true; 
  changed = true;
}//-

void 
VariableFragment::set_ground (bool b) {
  ground = b;
}//-

void 
VariableFragment::set_changed (bool b) {
  changed = b;
}//-

void 
VariableFragment::set_failed (bool b) {
  failed = b;
}//-

// test if current domain is ground
void
VariableFragment::test_ground() {
  ground = true;
  for (uint i = domain[labeled].get_bb_s(); 
       i <= domain[labeled].get_bb_e(); i++)
    if (!g_logicvars->var_point_list[i].is_ground()) {
      ground = false;
      return;
    }
}//-

// test if current variable is failed
void
VariableFragment::test_failed() {
  failed = 0;
  for (uint i = domain[labeled].get_bb_s(); 
       i <= domain[labeled].get_bb_e(); i++)
    if (!g_logicvars->var_point_list[i].is_failed()) {
      failed = 1;
      return;
    }
}//-

bool 
VariableFragment::is_ground () const {
  return (ground == 0) ? false : true;
}//-

bool 
VariableFragment::is_failed () const {
  return (failed == 0) ? false : true;
}//-

bool 
VariableFragment::is_changed () const {
  return (changed == 0) ? false : true;
}//-

bool
VariableFragment::is_in_bundle() const {
  if (labeled < 0 || labeled > (int)domain_info.size()-1)
    return false;
  // check pair relation
  if ( domain_info[labeled].frag_mate_info.size()>0)
    return true;
  
  return false;
}//-

void 
VariableFragment::set_assembly_direction (AssemblyDirection dir){
  _assembly_direction = dir;
}//-

AssemblyDirection 
VariableFragment::assembly_direction () const {
  return _assembly_direction;
}//-

int 
VariableFragment::get_idx () const {
  return idx;
}//-

int 
VariableFragment::get_last_trailed () const {
  return last_trailed;
}//-

void 
VariableFragment::set_last_trailed (int lt) {
  last_trailed=lt;
}//-

void
VariableFragment::dump () {
  cout << "VARIABLE FRAGMENT: (" << idx << ") ";
  cout << "Label: " << labeled << "\t";
  if (is_ground()) cout << " GROUND ";
  if (is_failed()) cout << " FAILED ";
  if (is_changed()) cout << " CHANGED ";
  cout << "Last Trailed: " << last_trailed << endl;
  cout << "Domain dimension: " << domain.size() << endl;  
}//-

