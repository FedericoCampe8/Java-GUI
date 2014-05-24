#include "bitset.h"
#include <vector>
#include <stdio.h>

Bitset::Bitset(size_t size, bool val) {
  bitmask.resize (size, val);
}
//-

Bitset::Bitset(const Bitset& other) {
  bitmask = other.bitmask;
}
//-

Bitset& 
Bitset::operator= (const Bitset& other) {
  if (this == &other)
    return *this;
  bitmask = other.bitmask;
  return *this;
}
//-
  
size_t 
Bitset::count (bool b) const {
  register size_t ct = 0;
  for (size_t i=0; i<bitmask.size(); i++)
    ct += (bitmask[i] == b) ? 1 : 0;
  return ct;
}
//-

Bitset&
Bitset::filp () {
  for (size_t i=0; i<bitmask.size(); i++)
    bitmask[i] = !bitmask[i];//^= 1
  return *this;
}
//-

bool 
Bitset::none () const {
  return (count() == 0);
}
//-

bool 
Bitset::operator[] (size_t pos) const {
  return bitmask[pos];
}
//-

void 
Bitset::reset () {
  for (size_t i=0; i<bitmask.size(); i++)
    bitmask[i] = false; //^=0
}
//-

void 
Bitset::resize (size_t size) {
  bitmask.resize(size, false);
}
//-

void
Bitset::set () {
  for (size_t i=0; i<bitmask.size(); i++)
    bitmask[i] = true; //&= 1
}
//-

void 
Bitset::set (size_t pos, bool val) {
  bitmask[pos] = val;
}
//-

size_t
Bitset::size () const {
  return bitmask.size();
}
//-

bool 
Bitset::test (size_t pos) const {
  return bitmask[pos];
}
//-

void
Bitset::dump() {
  std::cout << "Bitmask: ";
  for (int i = 0; i < bitmask.size(); i++)
    if (bitmask[i]) std::cout << "T ";
    else std::cout << "F ";
  std::cout << std::endl;
}
//-
