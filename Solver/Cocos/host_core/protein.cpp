#include "protein.h"

#include "globals.h"
#include "energy.h"
#include "utilities.h"
#include "atom.h"

using namespace std;
using namespace Utilities;

Protein::Protein( ) :
  _nres( 0 ),
  _is_h_defined ( false ) {
}//-

Protein::Protein( string filename ) :
_nres( 0 ) {
  load_protein ( filename );
}//-

Protein::~Protein () {
  _seq_code.clear();
  _tertiary.clear();
}//-

void 
Protein::load_protein (string filename, string chain) {
  
  // Tokens used to parse the string from the pdb file
  const string dbref = "DBREF ";
  const string seq   = "SEQRES";
  const string atom  = "ATOM  ";
  const string end   = "ENDMDL";
  bool set_chain = false;
  string aa;
  string line, token, buf, proper_chain;
  aminoacid c;
  real x, y, z;
  atom_type type = X;
  char ok = 0;

  ifstream protein_file ( filename.c_str() );
  if ( !protein_file.is_open() ) {
    cout << "#log: Protein - unable to open " << filename << endl;
    exit( 1 );
  }
  
  _name = filename.substr( 0, filename.size() - 4 );
  while ( protein_file.good() ) {
    getline( protein_file, line );
    token = line.substr(0, 6);
    
    if (token == dbref)
      _id = line.substr(7, 4);
    else if ( token == end )
      break;
    else if ( token == atom ) { // Atom found
      ok = 0;
      buf = line.substr( 12, 5 );
      
      /*
       * @note: 
       * this is needed in order to avoid
       * to add more than one atom of the same type
       * when there are different chains in the same
       * pdb file.
       */
      if ( Utilities::get_atom_type( buf ) == type ) continue;
      else type = get_atom_type ( buf );
      
      buf = line.substr ( 16, 1 );
      if(buf == " " || buf == "A") ok = 1;
      
      // Protein chain
      buf = line.substr( 21, 1 ); // chain
      
      if ( !set_chain ) {
        proper_chain = line.substr( 21, 1 );
        if ( proper_chain.compare( chain ) != 0 ) chain = proper_chain;
        set_chain = true;
      }
      
      ok *= ( buf == chain ) ? 1 : 0;
      if ( !ok ) type = X;
      
      // offset protein
      if ( _nres == 0 ) _offset = atoi ( line.substr(22, 5).c_str() );
      
      buf = line.substr( 26, 1 ); // res_ins
      
      // coordinates
      x = atof( line.substr( 30, 8 ).c_str() );
      y = atof( line.substr( 38, 8 ).c_str() );
      z = atof( line.substr( 46, 8 ).c_str() );
      
      if ( ok ) {
        if ( type == N  || type == CA  ||  type == CB  || type == O || type == H ) {
	  if ( type == H ) _is_h_defined = true;
          Atom a ( x, y, z, type );
          _tertiary.push_back( a );
          if ( type == CA ) {
            buf = line.substr( 17, 3 );  // Amino acid name
            if ( cv_aa3_to_aa1(buf).compare( "err" ) ) {
              _sequence.append( cv_aa3_to_aa1( buf ) );
              c = cv_aa_to_class( buf );
              if ( c != err ) _seq_code.push_back( c );
            }
            _nres++;
          }//type CA
        }//type
      }//ok
    }
  }//while
  
  // Handle Error
  if ( (_tertiary.size() <= _nres*( 5-1 ) || _tertiary.size() > _nres*5) && _is_h_defined ) {
    for ( uint i = 0; i < _tertiary.size(); i++ ) _tertiary.at( i ).dump();
    cout << endl;
    cout << "nres " << _nres << " tert.size " << _tertiary.size() << endl;
    cout << "4*nres " << _nres*5 << endl;
  }
  
  // Clear last and first atoms differnt from O and N respectively 
  if ( (_tertiary.size() <= _nres*(5-1) || _tertiary.size() > _nres*5) && _is_h_defined ) {
    while ( _tertiary.at(_tertiary.size()-1).type != O ) _tertiary.pop_back();
    while ( _tertiary.at( 0 ).type != N ) _tertiary.erase(_tertiary.begin());
  }
  
  if ( _is_h_defined ) {
    assert ( _tertiary.size() >  _nres*(5-1) && _tertiary.size() <= _nres*5 );
    assert ( _nres <= MAX_TARGET_SIZE );
  }
  
  protein_file.close();
}//load_protein

void
Protein::load_sequence ( string seq ) {
  cout << "Parser for Primary Structure (e.g., FASTA format): ToDo\n";
  std::string seq_well_form;
  if ( seq.compare(seq.length()-1, 1, "X") == 0 ) {
    seq_well_form = seq.substr( 0, seq.length()-1 );
  }
  else {
    seq_well_form = seq;
  }
  _sequence = seq_well_form;
  _nres = seq_well_form.length();
  for (uint i = 0; i < _nres; i++)
    _seq_code.push_back( Utilities::cv_aa_to_class( _sequence[ i ] ) );
}//load_target_sequence

void
Protein::set_sequence ( string seq ) {
  assert ( seq.length() <= MAX_TARGET_SIZE );
  std::string seq_well_form;
  if ( seq.compare(seq.length()-1, 1, "X") == 0 ) {
    seq_well_form = seq.substr( 0, seq.length()-1 );
  }
  else {
    seq_well_form = seq;
  }
  _sequence = seq_well_form;
  _nres = seq_well_form.length();
  for (uint i = 0; i < _nres; i++)
    _seq_code.push_back( Utilities::cv_aa_to_class( _sequence[ i ] ) );
}//set_sequence

string
Protein::get_sequence () {
  return _sequence;
}//get_sequence

uint
Protein::get_bblen () const {
  return _nres*5;
}//get_nres

uint
Protein::get_nres () const {
  return _nres;
}//get_nres

vector< Atom >
Protein::get_tertiary () {
  return _tertiary;
}//get_nres

std::vector< aminoacid >
Protein::get_sequence_code () {
  return _seq_code;
}//get_sequence_code

real
Protein::get_minium_energy () {
  real* plain_str = (real*) malloc ( _nres * 15 * sizeof(real) );
  for ( int i = 0; i < _nres * 5; i++ ) {
    plain_str[ 3*i + 0 ] = _tertiary[ i ][ 0 ];
    plain_str[ 3*i + 1 ] = _tertiary[ i ][ 1 ];
    plain_str[ 3*i + 2 ] = _tertiary[ i ][ 2 ];
  }
  
  int n_threads = 32;
  while ( n_threads < _nres ) n_threads += 32;
  n_threads = n_threads*2 + 32;
  real native_energy[ 1 ];
  real state_native[ 1 ];
  state_native[ 0 ] = 1;
  get_energy( plain_str, native_energy,
              state_native,
              gd_params.secondary_s_info,
              gd_params.h_distances, gd_params.h_angles,
              gd_params.contact_params, gd_params.aa_seq,
              gd_params.tors, gd_params.tors_corr,
              8, 22, 7,
              0, (5 * _nres) - 1,
              _nres, 0, _nres-1,
              0, 1, n_threads );
  return native_energy[ 0 ];
}//get_minium_energy

void
Protein::print_sequence () {
  cout << "AA Sequence: " << endl;
  cout << "--------------------------------------" << endl;
  cout << "1  - ";
  for ( uint i = 0; i < _sequence.length(); i++ ) {
    if ( ( i > 0 ) && ( i % 30 == 0 ) )
      cout << endl << i << " - ";
    cout << _sequence[ i ];
  }
  cout << endl;
  cout << "--------------------------------------" << endl;
}//print_sequence

void 
Protein::dump(){
  cout << "Protein: " << _name << " (id: " << _id << ") :: ";
  cout << "Seq: " << _sequence << endl;
  cout << "N.res: " << _nres << " backbone len: " << get_bblen() << endl;
  for (uint i = 0; i < _tertiary.size(); i++) _tertiary.at(i).dump();
}//dump
