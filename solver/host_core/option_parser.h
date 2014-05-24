/* 
 * OptionParser
 * The option parser is an helper class which is used to pass the 
 * input options to other objects. For example constraints. 
 */

#ifndef FIASCO_OPTION_PARSER_
#define FIASCO_OPTION_PARSER_

#include "typedefs.h"

#include <algorithm>
#include <string>
#include <vector>
#include <map>
#include <ios>
#include <ext/tree.hh>
#include <ext/tree_util.hh>
#include <ext/boost/any.hpp>

//TypeNamer prints out names of types.
template <class T>
struct TypeNamer {
    static std::string name() {
        return typeid(T()).name();
    }
};

template <>
struct TypeNamer<int> {
    static std::string name() {
        return "int";
    }
};

template <>
struct TypeNamer<size_t> {
    static std::string name() {
        return "size_t";
    }
};

template <>
struct TypeNamer<double> {
    static std::string name() {
        return "double";
    }
};

template <>
struct TypeNamer<bool> {
    static std::string name() {
        return "bool";
    }
};

template<>
struct TypeNamer<point> {
  static std::string name() {
    return "point";
  }
};


template <>
struct TypeNamer<std::string> {
    static std::string name() {
        return "string";
    }
};

template <class T>
struct TypeNamer<std::vector<T> > {
    static std::string name() {
        return "list of " + TypeNamer<T>::name();
    }
};


class OptionsParser {
 private:
  std::map<std::string, boost::any> storage;
  
 public:
  template <class T>
    void set(std::string key, T value) {
    storage[key] = value;
  }//-
  
  template <class T>
    T get (std::string key) const {
    std::map<std::string, boost::any>::const_iterator it;
    it = storage.find(key);
    if (it == storage.end()) {
      std::cout << "attempt to retrieve nonexisting object of name "
		<< key << " (type: " << TypeNamer<T>::name() << ")"
		<< " from Options. Aborting." << std::endl;
      exit(1);
    }
    try {
      T result = boost::any_cast<T> (it->second);
      return result;
    } catch (const boost::bad_any_cast &bac) {
      std::cout << "Invalid conversion while retrieving config options!"
		<< std::endl
		<< key << " is not of type " << TypeNamer<T>::name()
		<< std::endl << "exiting" << std::endl;
      exit(1);
    }
  }//-
  
  template <class T>
    void verify_list_non_empty (std::string key) const {
    std::vector<T> temp_vec = get<std::vector<T> >(key);
    if (temp_vec.empty()) {
      std::cout << "Error: unexpected empty list!"
		<< std::endl
		<< "List " << key << " is empty"
		<< std::endl;
      exit(1);
    }
  }//-
  
  template <class T>
    std::vector<T> get_list (std::string key) const {
    return get<std::vector<T> >(key);
  }//-
  
  int get_enum (std::string key) const {
    return get<int>(key);
  }//-
  
  bool contains (std::string key) const {
    return storage.find(key) != storage.end();
  }//-
  
};

#endif

