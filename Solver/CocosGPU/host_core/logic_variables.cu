#include "logic_variables.h"
#include "atom.h"
#include "utilities.h"

using namespace std;

LogicVariables::LogicVariables() :
_dbg         ( "#log: Logic Variables - " ),
cp_structure ( NULL ) {
}//-

LogicVariables::~LogicVariables() {
  delete [] cp_structure;
  cp_variables.clear();
}//-

void
LogicVariables::init_logic_variables () {
  populate_logic_variables ();
  populate_point_variables ();
}//init_logic_variables

void
LogicVariables::populate_logic_variables () {
  Utilities::print_debug ( _dbg, "Populate FD Vars" );
  
  ss_type type;
  bool to_break;
  for ( int i = 0; i < gh_params.n_res; i++ ) {
    for ( uint ii = 0; ii < gh_params.mas_des.size(); ii++ ) {
      type = other;
      to_break = false;
      if ( gh_params.mas_des[ii].vars_bds.size() > 0 ) {
        for ( int iii = 0; iii < gh_params.mas_des[ii].vars_bds.size(); iii++) {
          if ( (i >= gh_params.mas_des[ii].vars_bds[iii].first) &&
              (i <= gh_params.mas_des[ii].vars_bds[iii].second) ) {
            type = gh_params.mas_des[ii].sec_str_type;
            to_break = true;
            break;
          }
        }//iii
      }
      else {
        for ( int iii = 0; iii < gh_params.mas_des[ii].vars_list.size(); iii++) {
          if ( i == gh_params.mas_des[ii].vars_list[iii] ) {
            type = gh_params.mas_des[ii].sec_str_type;
            to_break = true;
            break;
          }
        }//iii
      }
      if ( to_break ) break;
    }//ii
    cp_variables.push_back( new AminoAcid ( type ) );
  }//i
}//populate_logic_variables

void
LogicVariables::populate_point_variables () {
  Utilities::print_debug ( _dbg, "Populate Point Vars" );
  
  assert ( cp_variables.size() > 0 );
  int num_points = cp_variables.size() * 5;
  point* cp_structure_aux = new point[ num_points ];
  point pa, pb, pc;
  point * right_points = new point[ 9 ];
  point * out_points   = new point[ 9 ];
  int t = 0, start, end;
  // Init first points
  for ( int j = 3; j < 9; j++ ) {
    cp_structure_aux[ t ][ 0 ] = cp_variables[ 0 ]->operator[](j)[ 0 ];
    cp_structure_aux[ t ][ 1 ] = cp_variables[ 0 ]->operator[](j)[ 1 ];
    cp_structure_aux[ t ][ 2 ] = cp_variables[ 0 ]->operator[](j)[ 2 ];
    t++;
  }
  /// Fix H atom
  cp_structure_aux[ 4 ][ 0 ] = cp_variables[ 0 ]->operator[](2)[ 0 ];
  cp_structure_aux[ 4 ][ 1 ] = cp_variables[ 0 ]->operator[](2)[ 1 ];
  cp_structure_aux[ 4 ][ 2 ] = cp_variables[ 0 ]->operator[](2)[ 2 ];
  
  
  // Init all structure
  for ( int i = 1; i < cp_variables.size(); i++ ) {
    int atom_idx = Utilities::get_bbidx_from_aaidx ( i-1, CB );
    for (int j = 0; j < 3; j++ ) {
      pa[ j ] = cp_structure_aux[ atom_idx      ][ j ];
      pb[ j ] = cp_structure_aux[ atom_idx  + 1 ][ j ];
      pc[ j ] = cp_structure_aux[ atom_idx  + 3 ][ j ];
    }
    
    t = 0;
    for ( int j = 0; j < 9; j++) {
      right_points[ t ][0] = cp_variables[ i ]->operator[](j)[0];
      right_points[ t ][1] = cp_variables[ i ]->operator[](j)[1];
      right_points[ t ][2] = cp_variables[ i ]->operator[](j)[2];
      t++;
    }
    for ( int j = 0; j < 9; j++)
      out_points[ j ][ 0 ] = out_points[ j ][ 1 ] = out_points[ j ][ 2 ] = 0;
    
    Utilities::overlap_structures ( pa, pb, pc, right_points, out_points, 9 );
    
    t = 4;
    start = Utilities::get_bbidx_from_aaidx ( i, CA );
    end   = start + 5;
    if ( i == cp_variables.size()-1 ) end = start + 4;
    for ( int j = start; j < end; j++ ) {
      cp_structure_aux[ j ][ 0 ] = out_points[ t ][ 0 ];
      cp_structure_aux[ j ][ 1 ] = out_points[ t ][ 1 ];
      cp_structure_aux[ j ][ 2 ] = out_points[ t ][ 2 ];
      t++;
    }
    /// Fix H atom
    cp_structure_aux[ start + 3 ][ 0 ] = out_points[ 2 ][ 0 ];
    cp_structure_aux[ start + 3 ][ 1 ] = out_points[ 2 ][ 1 ];
    cp_structure_aux[ start + 3 ][ 2 ] = out_points[ 2 ][ 2 ];
  }//i
  
  cp_structure = new real[ num_points*3 ];
  
  if ( gh_params.translate_str ) {
    gh_params.translation_point[ 1 ] -= cp_structure_aux[ (int) gh_params.translation_point[ 0 ] ][ 0 ];
    gh_params.translation_point[ 2 ] -= cp_structure_aux[ (int) gh_params.translation_point[ 0 ] ][ 1 ];
    gh_params.translation_point[ 3 ] -= cp_structure_aux[ (int) gh_params.translation_point[ 0 ] ][ 2 ];
  }
  
  for (int i = 0; i < num_points; i++) {
    cp_structure[ i*3 + 0 ] = cp_structure_aux[ i ][ 0 ] + gh_params.translation_point[ 1 ];
    cp_structure[ i*3 + 1 ] = cp_structure_aux[ i ][ 1 ] + gh_params.translation_point[ 2 ];
    cp_structure[ i*3 + 2 ] = cp_structure_aux[ i ][ 2 ] + gh_params.translation_point[ 3 ];
  }
  
  delete [] cp_structure_aux;
  delete [] right_points;
  delete [] out_points;
}//populate_structure

void
LogicVariables::clear_variables () {
  delete [] cp_structure;
  cp_variables.clear();
}//clear_variables

void
LogicVariables::set_point_variables ( real* pt_vars ) {
  if ( !cp_structure ) {
    assert ( cp_variables.size() * 15 == gh_params.n_points );
    cp_structure = (real*) calloc ( cp_variables.size() * 15, sizeof(real) );
  }
  memcpy ( cp_structure, pt_vars, cp_variables.size() * 15 * sizeof(real) );
}//set_point_variables

void
LogicVariables::set_interval_point_variables ( real* pt_vars, int bb_start, int bb_end, real* dest ) {
  assert ( bb_start > 0 && (bb_start % 5 == 0) );
  
  real * list_of_points = &cp_structure[ 0 ];
  if ( dest ) list_of_points = &dest[ 0 ];
  
  bool cover_whole_str = ( bb_end == (5 * cp_variables.size() - 1) );
  int len = bb_end - bb_start + 1 + 3; /// Preceding C O, and H atoms
  int len_aux = (5 * cp_variables.size()) - (bb_end - 2);
  if ( !cover_whole_str ) len++; /// Add Following N atom
  
  point pa, pb, pc;
  point * out_points     = new point[ len ];
  point * out_points_aux = new point[ len_aux ];
  
  for (int j = 0; j < 3; j++ ) {
    pa[ j ] = list_of_points [ (bb_start - 3)*3 + j ]; /// (N - 3)*3 = C
    pb[ j ] = list_of_points [ (bb_start - 2)*3 + j ]; /// (N - 2)*3 = O
    pc[ j ] = list_of_points [ bb_start*3 + j       ]; /// (N - 0)*3 = N
  }
  
  for ( int i = 0; i < len; i++ ) {
    out_points[ i ][ 0 ] = pt_vars[ (bb_start - 3)*3 + i*3     ];
    out_points[ i ][ 1 ] = pt_vars[ (bb_start - 3)*3 + i*3 + 1 ];
    out_points[ i ][ 2 ] = pt_vars[ (bb_start - 3)*3 + i*3 + 2 ];
  }
  
  Utilities::overlap_structures ( pa, pb, pc, out_points, len );
  
  if ( !cover_whole_str ) { /// # points = 5 * cp_variables.size()
    for ( int j = 0; j < 3; j++ ) {
      pa[ j ] = out_points [ len-4 ][ j ];
      pb[ j ] = out_points [ len-3 ][ j ];
      pc[ j ] = out_points [ len-1 ][ j ];
    }

    for ( int i = 0; i < len_aux; i++ ) {
      out_points_aux[ i ][ 0 ] = list_of_points[ (bb_end - 2)*3 + i*3     ];
      out_points_aux[ i ][ 1 ] = list_of_points[ (bb_end - 2)*3 + i*3 + 1 ];
      out_points_aux[ i ][ 2 ] = list_of_points[ (bb_end - 2)*3 + i*3 + 2 ];
    }
    
    Utilities::overlap_structures ( pa, pb, pc, out_points_aux, len_aux );
  }
  
  for ( int i = 0; i < len; i++ ) {
    list_of_points [ (bb_start - 3)*3 + i*3     ] = out_points[ i ][ 0 ];
    list_of_points [ (bb_start - 3)*3 + i*3 + 1 ] = out_points[ i ][ 1 ];
    list_of_points [ (bb_start - 3)*3 + i*3 + 2 ] = out_points[ i ][ 2 ];
  }
 
  if ( !cover_whole_str) {
    for ( int i = 0; i < len_aux; i++ ) {
      if (i == 2) continue; /// Skip first H atom on the right
      list_of_points [ (bb_end - 2)*3 + i*3     ] = out_points_aux[ i ][ 0 ];
      list_of_points [ (bb_end - 2)*3 + i*3 + 1 ] = out_points_aux[ i ][ 1 ];
      list_of_points [ (bb_end - 2)*3 + i*3 + 2 ] = out_points_aux[ i ][ 2 ];
    }
  }
  
  delete [] out_points;
  delete [] out_points_aux;
}//set_interval_point_variables

void
LogicVariables::print_point_variables () {
  print_point_variables ( 0, (int)cp_variables.size() );
}//print_point_variables

void
LogicVariables::print_point_variables ( int start_aa, int end_aa ) {
  int len = (end_aa - start_aa) * 5;
  point* cp_structure_aux = new point[ len ];
  
  real translation_vector[ 3 ];
  translation_vector[ 0 ] = cp_structure[ 1*3 + 0 ] * gh_params.translate_str_fnl;
  translation_vector[ 1 ] = cp_structure[ 1*3 + 1 ] * gh_params.translate_str_fnl;
  translation_vector[ 2 ] = cp_structure[ 1*3 + 2 ] * gh_params.translate_str_fnl;
  
  for (int i = start_aa; i < len; i++) {
    cp_structure_aux[ i ][ 0 ] = cp_structure[ i*3 + 0 ] - translation_vector[ 0 ];
    cp_structure_aux[ i ][ 1 ] = cp_structure[ i*3 + 1 ] - translation_vector[ 1 ];
    cp_structure_aux[ i ][ 2 ] = cp_structure[ i*3 + 2 ] - translation_vector[ 2 ];
  }
  string out_string = Utilities::output_pdb_format ( cp_structure_aux, len );
  /// Print on std ouput
  if ( gh_params.output_file.compare( "" ) == 0 ) {
    cout << out_string << endl;
  }
  else {
    FILE *fid;
    /// Open an output file
    fid = fopen ( gh_params.output_file.c_str(), "a" );
    if (fid < 0){
      printf( "Cannot open %s to write!\n", gh_params.output_file.c_str() );
      return;
    }
    fprintf( fid, "%s", out_string.c_str() );
    ///Close file
    fclose(fid);
  }
  
  delete [] cp_structure_aux; 
}//print_point_variables

void
LogicVariables::dump() {
  cout << "LOGIC VARIABLES:" << endl;
  for ( uint i = 0; i < cp_variables.size(); i++ )
    cp_variables[i]->dump();
}//dump
