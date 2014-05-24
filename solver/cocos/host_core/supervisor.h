#ifndef COCOS_SUPERVISOR_AGENT__
#define COCOS_SUPERVISOR_AGENT__

#include "globals.h"
#include "mas_agent.h"
#include "worker_agent.h"

class Supervisor {
private:
  int _n_mas_agents;
  std::string _dbg;
  agent_type _agt_type;
  real * _current_solution;
  std::vector< std::pair < uint , MasAgent* > > _mas_agents;          // Priority, MasAgent
  std::map< int, std::pair < MasAgent*, WorkerAgent* > > _wrk_agents; // V_id, <Parent, Wrk>

  void create_workers();
  void set_agents ();
public:
  Supervisor ();
  ~Supervisor ();
  
  void search();
  void clear_agents();
  
  void dump();
  
  
  /*
  void set_current_structure();
  void create_agents(std::vector<int> agts);
  void set_weights(std::vector<real>& weights_val);
  void clear_agents();
  
  void print_results(int str_agt_idx);
  void print_to_file(int str_agt_idx, real rmsd=1000);
  void print_to_file(std::vector< std::vector<real> > structure, real energy,
                       int start_aa, int num_wa, real rmsd);
   */
};

#endif

