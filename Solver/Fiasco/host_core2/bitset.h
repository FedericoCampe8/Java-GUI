/*********************************************************************
 * object: Bitset
 * c++ Bitset implementation 
 *********************************************************************/
#ifndef FIASCO_BITSET_
#define FIASCO_BITSET_

#include <vector>
#include <iostream>

class Bitset {
 protected:
  std::vector<bool> bitmask;

 public:
  Bitset() {};
  Bitset(size_t size, bool val=false);
  virtual ~Bitset() {};
  Bitset(const Bitset& other);
  Bitset& operator= (const Bitset& other);
  
  size_t count (bool b = true) const;
  Bitset& filp ();
  bool none () const;
  bool operator[] (size_t pos) const;
  void reset ();
  void resize (size_t size);
  void set ();
  void set (size_t pos, bool val = true);  
  size_t size () const;
  bool test (size_t pos) const; 
  virtual void dump();
};//-

#endif
