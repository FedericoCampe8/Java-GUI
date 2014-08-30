# ---------------------------------------------------------------------
# Objects
# ---------------------------------------------------------------------
SOLVER_PATH=.
MAIN=cocos
H_FOLDER=host_core

HEADERS_CORE=atom.h\
    logic_variables.h\
	atom_grid.h\
	aminoacid.h\
	domain.h\
	protein.h\
	globals.h\
	input_data.h\
	energy.h\
	energy_factory.h\
	potential_energy.h\
	rmsd_energy.h\
	contact_energy.h\
	contact_decay_energy.h\
	mathematics.h\
	utilities.h\
	supervisor.h\
	mas_agent.h\
	structure_agent.h\
	coordinator_agent.h\
	worker_agent.h\
	search_engine.h\
	icm.h\
 	docking.h\
	montecarlo.h\
	constraint_store.h\
	constraint.h\
	propagator.h\
	sang.h\
	mang.h\
	rand_move.h\
	atm_grd.h\
	all_distant.h\
	centroid.h\
	distance.h\
	gibbs.h\
	k_angle_shuffle.h\
	k_rang.h\
	rmsd_fast.h\

HEADERS=$(HEADERS_CORE:%.h=${H_FOLDER}/%.h)

SOURCES=${MAIN}.cpp $(HEADERS:%.h=%.cpp) 
OBJECTS = $(SOURCES:%.cpp=%.o)

# ---------------------------------------------------------------------
# Compiler options 
# ---------------------------------------------------------------------
CC = g++
#CC = clang++
NVCC = /usr/local/cuda/bin/nvcc
DEPEND = g++ -MM
CCOPT =    # -Wall -W -Wno-sign-compare -ansi -pedantic -Werror -Wno-deprecated
LINKOPT =

## debug, don't optimize 
CCOPT   += -m64 -DIL_STD

## debug, optimize
## use --ptxas-options=-v for cuda (verbose for each kernel)
## use -ftz=false -prec-div=true -prec-sqrt=true
## -L /usr/local/cuda/lib -lcurand
## –use_fast_math
CCOPT += -O3 -std=c++0x 
LINKOPT += -gx

PROFOPT = -pg -O2 -DNDEBUG

vpath %.o obj

## lib dirs -L...
CCLNDIRS= 
## include dirs -I...
INCLDIRS = -I$(SOLVER_PATH) -I$(SOLVER_PATH)/$(H_FOLDER) 

## Compiler Flags
CCFLAGS = $(CCOPT) $(INCLDIRS)
#NVCCFLAGS = -arch=sm_21 

all:	cocos

cocos: $(OBJECTS) 
	$(CC) $(CCLNDIRS) $(DFLAGS) -o $(MAIN) \
		$(OBJECTS:%=$(SOLVER_PATH)/obj/%)

$(OBJECTS): %.o: %.cpp
	$(CC) -c $(CCFLAGS) $< -o $(SOLVER_PATH)/obj/$@

clean:
	rm -f obj/host_core/*.o
	rm -f obj/cocos.o
	rm  $(MAIN)
