#include "output.h"
#include "typedefs.h"
#include "globals.h"
#include "logic_variables.h"
#include "variable_point.h"
#include "variable_fragment.h"
#include "protein.h"
#include "utilities.h"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <stdlib.h> // for atoi
using namespace std;


Output::Output(int argc, char* argv[]) 
  : model_ct(0) {

  string outfile;
  for (int narg = 0 ; narg < argc  ; narg++) {
    if (!strcmp ("--outfile", argv[narg]) || 
	!strcmp ("-o",argv[narg])) {
      outfile = argv[narg + 1];
      break;
    }
  }
  if (!outfile.empty())
    set_out_file (outfile);
  else
    set_out_file (g_target.get_name()+".out.pdb");


  buffer.reserve(1000000);


};

string get_format_spaces(real x) {
  if (x < 0 && x > -10)  return "  ";
  if (x >= 0 && x < 10)  return "   ";
  if (x > 10 && x < 100) return "  ";   
  return " ";
}

int get_format_digits(real x) {
  if (x < 0 && x > -10)  return 4;
  if (x >= 0 && x < 10)  return 4;
  if (x > 10 && x < 100) return 5;   
  if (x > 100 || x < -100) return 6;
  return 5;
}
 
void 
Output::store_results( real energy ) {
  store_results( 0, g_logicvars->var_point_list.size() - 1, energy );
}//store_results

void 
Output::store_results( real rmsd, real energy ) {
  store_results( 0, g_logicvars->var_point_list.size()-1, rmsd, energy ) ;
}//store_results

void
Output::store_best_results( real energy ) {
  store_results( 0, g_logicvars->var_point_list.size() - 1, energy, -1, true );
}//store_best_results

void
Output::store_best_results( real rmsd, real energy ) {
  store_results( 0, g_logicvars->var_point_list.size()-1, energy, rmsd, true ) ;
}//store_best_results

void 
Output::store_results ( uint atom_s, uint atom_e, real energy, real rmsd, bool best_result ){
  stringstream s;
  real x,y,z;
  int aa_idx=-1;
  s << "REMARK \t ENERGY: " << energy << endl;
  if (rmsd >= 0)
    s << "REMARK \t RMSD: " << rmsd << endl;
  for (uint i = atom_s; i <= atom_e; i++) {
    x = g_logicvars->var_point_list.at(i)[0];
    y = g_logicvars->var_point_list.at(i)[1];
    z = g_logicvars->var_point_list.at(i)[2];
    
    s <<"ATOM   "<< setw(4)<<i+1<<"  ";
    if (i%4 == NITROGEN_ATOM){
      s<<"N   ";
      aa_idx = Utilities::get_aaidx_from_bbidx(i, N);
    }
    if (i%4 == CALPHA_ATOM){
      s<<"CA  ";
      aa_idx = Utilities::get_aaidx_from_bbidx(i, CA);
    }
    if (i%4 == CBETA_ATOM) {
      s<<"C   ";
      aa_idx = Utilities::get_aaidx_from_bbidx(i, CB);
    }
    if (i%4 == OXYGEN_ATOM) {
      s<<"O   ";
      aa_idx = Utilities::get_aaidx_from_bbidx(i, O);
    }
    
    s<<Utilities::cv_aa1_to_aa3(g_target.sequence[aa_idx])
     <<" A "
     <<setw(3)<<Utilities::get_aaidx_from_bbidx(i, atom_type(i%4))
     <<"    "
     <<fixed
     <<get_format_spaces(x)<<setprecision(3)<<x
     <<get_format_spaces(y)<<setprecision(3)<<y
     <<get_format_spaces(z)<<setprecision(3)<<z
     <<"  1.00  1.00\n";
  }


  // Side Chains
  for (int i = Utilities::get_aaidx_from_bbidx (atom_s, atom_type(atom_s%4)); 
       i < Utilities::get_aaidx_from_bbidx (atom_e, atom_type(atom_e%4))-1; 
       i++) {

    // if (!g_target.side_chains[i].empty()) {
    //   for (int j=0; j<g_target.side_chains[i].size(); j++) {
    // 	x = g_target.side_chains[i][j][0];
    // 	y = g_target.side_chains[i][j][1];
    // 	z = g_target.side_chains[i][j][2];
    // 	// Translate these points first -- but it may not belong to 
    // 	// this fragment specifically -- i.e. block -- 
    // 	// Math::rotate (x, g_logicvars->var_point_list[])

    // 	s << "ATOM   " << setw(4) << i+g_logicvars->var_point_list.size()
    // 	  << "  CG  "  << Utilities::cv_aa1_to_aa3(g_target.sequence[i+1])
    // 	  << " A " << setw(3) << i+1 << "    " << fixed
    // 	  << get_format_spaces(x)<<setprecision(3) << x
    // 	  << get_format_spaces(y)<<setprecision(3) << y
    // 	  << get_format_spaces(z)<<setprecision(3) << z
    // 	  << "  1.00  1.00\n";	
    //   }
    // }
    // else {
      if(g_logicvars->var_cg_list.size() >= i) {
	x = g_logicvars->var_cg_list.at(i)[0];
	y = g_logicvars->var_cg_list.at(i)[1];
	z = g_logicvars->var_cg_list.at(i)[2];
	s<<"ATOM   "
	 <<setw(4)<<i+g_logicvars->var_point_list.size()
	 <<"  CG  "
	 <<Utilities::cv_aa1_to_aa3(g_target.sequence[i+1])
	 <<" A "<<setw(3)<<i+1<<"    "
	 <<fixed
	 <<get_format_spaces(x)<<setprecision(3)<<x
	 <<get_format_spaces(y)<<setprecision(3)<<y
	 <<get_format_spaces(z)<<setprecision(3)<<z
	 <<"  1.00  1.00\n";
      }
      //    }
  }
  
  if ( best_result ) {
    buffer.clear();
    buffer.push_back( s.str() );
  }
  else {
    /// Save everything
    buffer.push_back( s.str() );
  }
  
}//store_results


void 
Output::store_points ( std::vector<real> points ){
  stringstream s;
  real x,y,z;
  for (uint i=0; i < points.size(); i+=3) {
    x = points[i];
    y = points[i+1];
    z = points[i+2];
    
    s <<"ATOM   "<< setw(4)<<"0  ";
    if (i==0) s << "C   XXX";
    if (i==3) s << "O   XXX";
    if (i==6) s << "N   XXX";
    s <<" A "
      <<setw(3)<<i
      <<"    "
      <<fixed
      <<get_format_spaces(x)<<setprecision(3)<<x
      <<get_format_spaces(y)<<setprecision(3)<<y
      <<get_format_spaces(z)<<setprecision(3)<<z
      <<"  1.00  1.00\n";
  }
  buffer.push_back(s.str());
}//
  
void 
Output::dump() {
  ofstream os ( file_out.c_str(), ios::out | ios::app );
  
  if ( os.is_open() ){
    for ( uint i=0; i<buffer.size(); i++ ) {
      os << "MODEL     " << model_ct++ << endl;
      os << buffer.at(i); 
      os << "ENDMDL\n";
    }
    os.close();  
    buffer.clear();
  }
  else cout << "Error: cannot open " << file_out << " file " << endl;
}//-

