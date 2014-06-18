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
#ifndef FIASCO_INTERVAL_
#define FIASCO_INTERVAL_

#include "typedefs.h"
#include "fragment.h"
#include "globals.h"

#include <iostream>
#include <queue>
#include <vector>

using namespace std;
//using namespace Globals;

class Range {
 public:
  Range(){};
  ~Range(){};
  pair<int, int> range;  
  int length() {return (range.second - range.first) + 1;}
};

class Intervals {
 public:
  std::vector<Range> ints;
  
  Intervals () {
    Range r;
    r.range.first = 0;
    r.range.second = 0;//g_target.get_nres();
    ints.push_back(r);
  };
  Intervals(uint size) {
    Range r;
    r.range.first = 0;
    r.range.second = size;
    ints.push_back(r);
  }//-
  ~Intervals() {};
  
  Range operator[]  (uint idx) const;
  
  void add(int a, int b);
  void remove(int idx);
  void update (Fragment& f, _var_type vtype, _exploring_dir d);

  uint size() const {return ints.size();}
  const pair<int, int> first();	// first interval to consider
  const pair<int, int> last();	// last interval to consider
  const pair<int, int> smaller();	// smaller interval
  const pair<int, int> larger();	// larger interval
  int sumground();
};

#endif
