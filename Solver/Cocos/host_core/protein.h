#ifndef COCOS_PROTEIN__
#define COCOS_PROTEIN__

#include <string>
#include <vector>

#include "typedefs.h"
#include "atom.h"


/*
 * This class is used
 * to manage the protein representation
 */
class Protein {
private:
  std::string _id;
  std::string _name;
  uint _nres;	        // Protein length
  uint _offset;       // Offset related to the first AA number 
  bool _is_h_defined;
 
  // Primary Structure
  std::string _sequence;
  std::vector< aminoacid > _seq_code;
  // Tertiary Structure
  std::vector< Atom > _tertiary;
public:

  
  Protein ();
  Protein( std::string filename );
  ~Protein();

  Atom& operator[]  ( const int index ){ return _tertiary[index]; }

  void load_protein ( std::string filename, std::string chain = "A" );
  void load_sequence ( std::string seq );
  void set_sequence ( std::string seq );
  uint get_nres() const;
  std::string get_sequence ();
  std::vector< aminoacid > get_sequence_code ();
  uint get_bblen() const;
  std::vector< Atom > get_tertiary ();
   
  
  
  
  
  
  void set_id (std::string pid) { _id = pid; }
  std::string get_id() const { return _id; }
  void set_name (std::string pname) { _name = pname; }
  std::string get_name() const { return _name; }
  void set_nres(int n) { _nres = n; }
  
  void set_offset (int off) { _offset = off; }
  uint get_offset() const { return _offset; }
  
  real get_minium_energy();
  
  
  void print_sequence ();
  void dump ();
};

#endif
