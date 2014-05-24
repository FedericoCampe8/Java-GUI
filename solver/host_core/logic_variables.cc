#include "logic_variables.h"
#include "variable_point.h"
#include "variable_fragment.h"
#include "typedefs.h"
#include "atom.h"
#include "fragment.h"
#include "protein.h"
#include "globals.h"
#include "utilities.h"

#include <iostream>
#include <string>
#include <vector>
#include <cassert>
#include <algorithm>

using namespace std;
using namespace Utilities;

LogicVariables::LogicVariables 
(int argc, char* argv[]) {

  size_t domain_size = 0;
  for (int narg=0; narg < argc; narg++) {
    if (!strcmp ("--domain-size",argv[narg]) || 
	!strcmp ("-d",argv[narg])) {
      domain_size = atoi(argv[narg + 1]);
      break;
    }
  }
  
  size_t backbone_length = g_target.get_bblen();
  for (uint i = 0; i < backbone_length; i++) {
    VariablePoint p(i);
    var_point_list.push_back(p);
  }
  size_t nres = g_target.get_nres();
  for (uint i = 0; i < nres-2; i++ ) {
    Atom a(0,0,0, CG, backbone_length+i);
    var_cg_list.push_back(a);
  }
  
  // Populate Variable FRAGMENTS
  populate_fragment_variables (g_target, g_assembly_db);

  // Resize domains
  if (domain_size > 0) {
    for (uint i=0; i < var_fragment_list.size(); i++) {
      if (var_fragment_list[i].domain.size() > domain_size) {
	var_fragment_list[i].domain.resize (domain_size);
	var_fragment_list[i].domain_info.resize (domain_size);
      }
    }
  }

  // Restrict domain of Special Fragments
  for (uint i = 0; i < var_fragment_list.size(); i++)  {
    VariableFragment *VF = &var_fragment_list[i];
    for (uint ii = 0; ii < VF->domain_size(); ii++) {
      if (VF->domain[ii].get_type() == special){
	VF->domain.resize(1);
	VF->domain_info.resize(1);
      }
    }
  }

}//-

void 
LogicVariables::reset () {
  for (uint i=0; i<var_point_list.size(); i++)
    var_point_list[i].reset();

  for (uint i=0; i<var_fragment_list.size(); i++)
    var_fragment_list[i].reset();

  for (uint i=0; i<var_cg_list.size(); i++)
    var_cg_list[i].set_position(0,0,0);
}//-

void 
LogicVariables::reset_allvars_changed() {
  for (uint i=0; i<var_point_list.size(); i++)
    var_point_list.at(i).set_changed(false);
  for (uint i=0; i<var_fragment_list.size(); i++)
    var_fragment_list.at(i).set_changed(false);
  // for (uint i=0; i<var_pair_list.size(); i++)
  //   var_pair_list.at(i).set_changed(false);
}//-

// Populate the Variable set, extracting the compatible fragments from
// the Fragment assembly DB. Compatibility refers to the protein seq
void 
LogicVariables::populate_fragment_variables_md(Protein target, 
                  const vector< vector<Fragment> >& assembly_db, 
		  int numMd, vector<Fragment>& assembly_db_out){
    
  cout << "num db " << numMd << endl;
  if(numMd > 0){
    for (int i = 3; i >= 0; i--){
      for (uint j = 0; j < assembly_db.at(i).size(); j++)
	assembly_db_out.push_back(assembly_db.at(i).at(j));
    }
    
    /* Populate the variables */
    cout << "populate the variables: start" << endl;
    populate_fragment_variables(target, assembly_db_out);
    cout << "populate the variables: end" << endl;
  }else{
    populate_fragment_variables(target, assembly_db.at(0));
  }
}//populate_fragment_variables_md

void
LogicVariables::populate_fragment_variables(Protein target, 
		       const vector<Fragment>& assembly_db){
  
  uint prot_len = target.get_nres();
    
  bool is_match;
  // Compare input amino acid class with each element of the Fagment DB
  for (uint i = 0; i < prot_len; i++) {
    VariableFragment f_var(i);
    // iterate on the set of fragments
    for (uint ii = 0; ii < assembly_db.size(); ii++) {
      is_match = true;
        
      uint f_len = assembly_db.at(ii).nres();
        
        if ( i + (f_len - 1) >= target.seq_code.size()) {
            continue;
        }
        
      for(uint k = 0; k < f_len; k++) {
          
        if( target.seq_code.at(i+k) !=
           assembly_db.at(ii).aa_seq.at(k) ){
            is_match = false;	// Different class sequence
            break;
        }
      }
        
      // If a match has been found copy fragment into logic vars
      if(is_match) {
	Fragment f(assembly_db.at(ii));
	if (assembly_db[ii].get_type() != special) {
	  int bbs = i > 0 ? 
	    get_bbidx_from_aaidx(i-1, CB) 
	    : 0;
	  int bbe = i+(f_len-1) < g_target.get_nres()-1 ? 
	    bbs+(4*f_len+2) 
	    : g_target.get_bblen()-1;
	  
	  f.set_aa_s(i);
	  f.set_aa_e(i+f_len-1);
	  f.set_bb_s(bbs);
	  f.set_bb_e(bbe);
	}
        // Add fragment f into domain of var fragment
        f_var.add_domain_elem(f);
    }
        
    }//ii
    assert (f_var.domain_size() > 0);
    
    sort (f_var.domain.begin(), f_var.domain.end(), Fragment());
    var_fragment_list.push_back(f_var);
  }//i
}//-


/*
 * Syntactic sugar to count the number of a variable of type atom
 * that have been already chosen.
 * Used by VariableSelection in DFS
 */
int
LogicVariables::natom_ground (int atomType) const {
  int count = 0;
  for (uint i=0; i < g_target.get_bblen(); i++) {
    if(g_logicvars->var_point_list[i*4+atomType].is_ground() == 1)
      count++;
    else break;
  }
  return count;
}//-
