/******************************************
 *           ENERGY FACTORY               *
 * Interface for creating an object       *
 * that represents an energy function.    *
 ******************************************/

#ifndef COCOS_ENERGY_FACTORY__
#define COCOS_ENERGY_FACTORY__

#include "globals.h"
#include "energy.h"
#include "potential_energy.h"
#include "rmsd_energy.h"
#include "contact_energy.h"
#include "contact_decay_energy.h"

  enum class EnergyType {
    Potential_Energy_t,
    Rmsd_Energy_t,
    Contact_Energy_t,
    Contact_Decay_Energy_t
  };

class EnergyFactory {
public:
  
  static Energy*
  getEnergyFunction( EnergyType energyType, agent_type agentType=structure ) {
    Energy * energy_class = nullptr;
    switch ( energyType ) {
      case EnergyType::Potential_Energy_t:
      {
        real w_a = gh_params.str_weights[ 0 ];
        real w_b = gh_params.str_weights[ 1 ];
        real w_c = gh_params.str_weights[ 2 ];
        if ( agentType == coordinator ) {
          w_a = gh_params.crd_weights[ 0 ];
          w_b = gh_params.crd_weights[ 1 ];
          w_c = gh_params.crd_weights[ 2 ];
        }
	energy_class = new PotentialEnergy ();
	static_cast<PotentialEnergy*> ( energy_class )->set_parameters ( gd_params.secondary_s_info,
                                                                            gd_params.h_distances,
                                                                            gd_params.h_angles,
                                                                            gd_params.contact_params,
                                                                            gd_params.aa_seq,
                                                                            gd_params.tors, gd_params.tors_corr,
                                                                            w_a, w_b, w_c );
      }
      break;
      case EnergyType::Rmsd_Energy_t:
	energy_class = new RmsdEnergy ();
	static_cast<RmsdEnergy*> ( energy_class )->set_parameters ( gd_params.known_prot );
	break;
      case EnergyType::Contact_Energy_t:
	energy_class = new ContactEnergy ();
	static_cast<ContactEnergy*> ( energy_class )->set_parameters ( gd_params.aa_seq );
	break;
      case EnergyType::Contact_Decay_Energy_t:
      {
        point atom  = { 0, 0, 0 };
	energy_class = new ContactDecayEnergy ();
	static_cast<ContactDecayEnergy*> ( energy_class )->set_parameters ( gd_params.aa_seq, atom, 0 );
      }
	break;
    }
    return energy_class;
  }//getEnergyFunction
};


#endif
