/*********************************************************************
 * Authors: Ferdinando Fioretto (ffiorett@cs.nmsu.edu)               *
 *          Federico Campeotto (campe8@cs.nmsu.edu)                  *
 *          Alessandro Dal Palu', Enrico Pontelli, Agostino Dovier   *
 * (C) Copyright 2010-2011                                           *
 *                                                                   *
 * This file is part of FIASCO.                                      *
 *                                                                   *
 * FIASCO is free software; you can redistribute it and/or           *
 * modify it under the terms of the GNU General Public License       *
 * as published by the Free Software Foundation;                     *
 *                                                                   *
 * FIASCO is distributed WITHOUT ANY WARRANTY; without even the      *
 * implied  warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR  *
 * PURPOSE. See the GNU General Public License for more details.     *
 *                                                                   *
 * You should have received a copy of the GNU General Public License *
 * along with this program; if not, see http://www.gnu.org/licenses. *
 *                                                                   *
 *********************************************************************/
#ifndef FIASCO_CONSTRAINT_STORE_
#define FIASCO_CONSTRAINT_STORE_

#include "typedefs.h"
#include "constraint.h"
#include "variable_fragment.h"
#include "variable_point.h"

#include <vector>

class ConstraintStore {
 public:
  std::vector<Constraint*> store; // HACK!! this should be a priority queue!!
  std::vector<VariablePoint*> changed_point_vars;
  std::vector<VariableFragment*> changed_fragment_vars;
  // List of constraints to be checked after variable is changed -> 
  // a vector for each type of variable
  std::vector<Constraint*> constr_dep_point;
  std::vector<Constraint*> constr_dep_fragment;

  // Indexes for the vectors. We use them in order to acquire efficency
  uint nchanged_point;
  uint nchanged_frag;

  ConstraintStore() {};
  ~ConstraintStore() {};

   bool propagate (size_t trailtop);
   bool arc_consistency3 (size_t trailtop);
   bool check_cardinality_constraints (size_t& backjump);
   void add (Constraint* c);
   void remove(Constraint* c);
   Constraint* fetch();
   void reset ();

   /* @todo: check id into constraints -- never initialized */
   void add_constr_dep_var_point_changed();
   void add_constr_dep_var_fragment_changed();
   void upd_changed(VariablePoint *point); 
   void upd_changed(VariableFragment *frag);

   void dump();
};//-

#endif
