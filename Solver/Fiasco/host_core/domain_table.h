/*********************************************************************
 * Authors: Ferdinando Fioretto (ffiorett@cs.nmsu.edu)               *
 *          Federico Campeotto  (campe8@nmsu.edu)                    *
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
#ifndef FIASCO_DOMAIN_TABLE_
#define FIASCO_DOMAIN_TABLE_

#include "typedefs.h"
#include "globals.h"

#include <string>
#include <vector>

using namespace std;
//using namespace Globals;

class DomainTable{
 public:
  /* [Matrix] Domain of current variable */
  vector< vector<bool> > vdom_expl;  // var domain explored
  vector<_var_type> var_type;
  vector<int> var_idx;		// current var index in node 
  vector<int> label;		// current var element labeld
  vector<_thread_type> thread;

  DomainTable();
  ~DomainTable(){};
};

#endif
