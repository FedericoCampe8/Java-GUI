#include "globals.h"
#include "output.h"
#include "statistics.h"
#include "logic_variables.h"
#include "variable_fragment.h"
#include "variable_point.h"

#include "anchor.h"
#include "input_data.h"
#include "dfs_search.h"
#include "flexseq_first_search.h"
#include "utilities.h"


#include "constraint.h"
#include "distance_leq_constraint.h"
#include "distance_geq_constraint.h"
#include "alldistant_constraint.h"
#include "centroid_constraint.h"
#include "fragment_constraint.h"
#include "bundle_constraint.h"
#include "ellipsoid_constraint.h"
#include "jm_constraint.h"
#include "table_constraint.h"
#include "uniform_constraint.h"
#include "uniqueseq_constraint.h"
#include "unique_source_sinks_constraint.h"
#include "end_anchor_different_constraint.h"


using namespace std;


int main (int argc, char* argv[]) {

  string dbg = "DBG_MAIN() - ";    

  // Process inputs
  Input_data In (argc, argv);
  // Load Known and Target Proteins
  cout << dbg << "Read input\n";
  g_known_prot.load_protein ( In.get_known_prot_file(), In.get_known_prot_chain() );
  g_target.load_protein ( In.get_target_prot_file(), In.get_known_prot_chain() );
  cout << dbg << "Load input and target proteins\n";
  // Process output preferences
  g_output = new Output (argc, argv);
  // Create Statistics object
  g_statistics = new Statistics (argc, argv);
  // Load Fragment Assembly DB
  Utilities::populate_fragment_assembly_db ( g_assembly_db, 1, In.get_fragmentdb() );
  cout << dbg << "Populate fragments\n";
  // ---------------------------------------------------------------
  // Set Problem Variables 
  // ---------------------------------------------------------------
  g_logicvars = new LogicVariables (argc, argv);
  // Set Reference System
  Fragment *first_fragment = &g_logicvars->var_fragment_list[0].domain[0];
  R_MAT &rot_mat_ori_system = first_fragment->rot_m;
  vec3  &sh_vec_ori_system = first_fragment->shift_v;
  g_reference_system = 
    Math::set_reference_system (first_fragment->backbone[0].position, 
                                first_fragment->backbone[1].position,
                                first_fragment->backbone[2].position);

  // Reduce domain for first Frag
  //g_logicvars->var_fragment_list[0].set_domain_singleton( 0 );
  cout << dbg << "Set variables\n";
  
  
  
  // Loop modelling ------------------------------------------------
  vector<pair<uint, uint> > aas_and_aae_of_rigid_blocks;
  vector<VariableFragment*> bundle_fragments;
  
  // FIX THIS PART BELOW
  // set AA start and end positions of RIGID BLOCKS
  for (uint i=0; i<g_logicvars->var_fragment_list.size(); i++)  {
    VariableFragment *VF = &g_logicvars->var_fragment_list[i];
    for (uint ii=0; ii < VF->domain_size(); ii++)
      if (VF->domain[ii].get_type() == special) {
        bundle_fragments.push_back(VF);
        uint aa_s = VF->domain[0].get_aa_s();
        uint aa_e = VF->domain[0].get_aa_e();
        aas_and_aae_of_rigid_blocks.push_back (make_pair (aa_s, aa_e));
        break;
      }
  }
  //-
  
  // Setting the FLEXIBLE PROTIEN SEQUENCE
  uint flexible_sequence_aas = aas_and_aae_of_rigid_blocks[0].second + 1;
  uint flexible_sequence_aae = aas_and_aae_of_rigid_blocks[1].first  - 1;
  uint flexible_sequence_bbs =
  Utilities::get_bbidx_from_aaidx (flexible_sequence_aas, N);
  uint flexible_sequence_bbe =
  Utilities::get_bbidx_from_aaidx (flexible_sequence_aae, O);
  
  g_target.set_loop ( flexible_sequence_bbs, flexible_sequence_bbe, "jm_loop" );
  
  // Set this other loop for our convenience. This will be only used to print the
  // flexible part of the protein, untill the end
  g_target.set_loop ( flexible_sequence_bbs-8, flexible_sequence_bbe+8, "flexible_chain_to_print" );
  g_statistics->new_loop_search_space (flexible_sequence_aas, flexible_sequence_aae);
  
  
  // ---------------------------------------------------------------
  // Set Constraints 
  // ---------------------------------------------------------------  
  int parse_pos = 0;
  // Fragment Constraints
  for (uint i=0; i < g_target.get_nres(); i++) {
    ConstraintFragment *c = 
      new ConstraintFragment (&g_logicvars->var_fragment_list[i]);
  }


   // Ditance Constraints 
  parse_pos = 0;
  while (parse_pos >= 0)
    DistanceLEQConstraint *leq = 
      new DistanceLEQConstraint(argc, argv, parse_pos);


  parse_pos = 0;
  while (parse_pos >= 0)
    DistanceGEQConstraint *geq = 
      new DistanceGEQConstraint(argc, argv, parse_pos);

  // AllDist Constraint
  AlldistantConstraint *alldist = 
    new AlldistantConstraint();


//  Centroid Constraints
  for (uint i=1; i < g_target.get_nres()-1; i++) {
    int bb1 = Utilities::get_bbidx_from_aaidx (i-1, CA);
    int bb2 = Utilities::get_bbidx_from_aaidx (i,   CA);
    int bb3 = Utilities::get_bbidx_from_aaidx (i+1, CA);
    CentroidConstraint *cg = 
      new CentroidConstraint (&g_logicvars->var_point_list[bb1],
			      &g_logicvars->var_point_list[bb2],
			      &g_logicvars->var_point_list[bb3]);
  }
#ifdef FALSE

  // Ditance Constraints 
  parse_pos = 0;
  while (parse_pos >= 0)
    DistanceLEQConstraint *leq = 
      new DistanceLEQConstraint(argc, argv, parse_pos);

  

  
#endif
  
  // Ellipsoid Constraint
  parse_pos = 0;
  while (parse_pos >= 0)
    EllipsoidConstraint *ce = 
      new EllipsoidConstraint (argc, argv, parse_pos, 
  			       rot_mat_ori_system, sh_vec_ori_system);

  // Uniform Constraints
  parse_pos = 0;
  while (parse_pos >= 0)
    UniformConstraint *uniform = 
      new UniformConstraint (argc, argv, parse_pos, 
			     rot_mat_ori_system, sh_vec_ori_system);

  // UniqueSeqence Constraints
  parse_pos = 0;
  while (parse_pos >= 0)
    UniqueSeqConstraint *unique_seq = 
      new UniqueSeqConstraint (argc, argv, parse_pos);

  // UniqueSourceSinks Constraints
  parse_pos = 0;
  while (parse_pos >= 0)
    UniqueSourceSinksConstraint *unique_source_sinks = 
      new UniqueSourceSinksConstraint (argc, argv, parse_pos);

  // End-Anchor-Different Constraint
  parse_pos = 0;
  while (parse_pos >= 0)
    EndAnchorDifferentConstraint *eadiff = 
      new EndAnchorDifferentConstraint (argc, argv, parse_pos);

  // JM Constraints
  parse_pos = 0;
  int scanner = 0;
  while (parse_pos >=0 || scanner >= 0) {
    JMConstraint * jm = 
      new JMConstraint (argc, argv, parse_pos, scanner);
  }
  cout << dbg << "Post constraints\n";
  // ---------------------------------------------------------------
  // Start the Search Engine 
  // ---------------------------------------------------------------
  DepthFirstSearchEngine *search_engine = new DepthFirstSearchEngine (argc, argv);
  //FlexSeqFirstSearch *search_engine = new FlexSeqFirstSearch (argc, argv);
  
  cout << dbg << "Start search...\n";
  g_statistics->set_timer (t_search);
  search_engine->search();
  g_statistics->stopwatch (t_search);

  search_engine->dump_statistics();
  g_output->dump();
  g_statistics->dump();


  // Flush Memory
  if (g_output)      delete g_output;
  if (g_logicvars)   delete g_logicvars;
  //if (g_statistics)  delete g_statistics;
  if (search_engine) delete search_engine;
  for (uint i = 0; i < g_constraints.size(); i++)
    if (g_constraints[i]) 
      delete g_constraints[i];
  cout << dbg << "Flush memory\n";
  cout << dbg << "Exit from FIASCO\n";
  return 0;
}//-

