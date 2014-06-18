#ifndef COCOS_STRUCTURE_AGENT__
#define COCOS_STRUCTURE_AGENT__

#include "globals.h"
#include "mas_agent.h"

class StructureAgent : public MasAgent {
public:
  StructureAgent ( MasAgentDes description, int prot_len );
  StructureAgent ( const StructureAgent& other );
  ~StructureAgent ();
  
  virtual void search ();
  virtual void dump();
};

#endif

