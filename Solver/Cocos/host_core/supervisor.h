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
};

#endif

