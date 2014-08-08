#ifndef COCOS_COORDINATOR_AGENT__
#define COCOS_COORDINATOR_AGENT__

#include "globals.h"
#include "mas_agent.h"

class CoordinatorAgent : public MasAgent {
public:
  CoordinatorAgent ( MasAgentDes description, int prot_len );
  CoordinatorAgent ( const CoordinatorAgent& other );
  ~CoordinatorAgent ();
  
  virtual void search ();
  virtual void dump();
};

#endif

