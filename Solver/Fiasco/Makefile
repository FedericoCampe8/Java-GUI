# ---------------------------------------------------------------------
# Objects
# ---------------------------------------------------------------------
SOLVER_PATH=.
MAIN=fiasco
H_FOLDER=host_core

HEADERS_CORE=alldistant_constraint.h\
	anchor.h\
	atom_grid.h\
	atom.h\
	b_cluster.h\
	bitset.h\
	bundle_constraint.h\
	centroid_constraint.h\
	constraint_store.h\
	constraint.h\
	cubic_lattice.h\
	dfs_search.h\
	distance_geq_constraint.h\
	distance_leq_constraint.h\
	ellipsoid_constraint.h\
	end_anchor_different_constraint.h\
	flexseq_first_search.h\
	fragment_constraint.h\
	fragment.h\
	globals.h\
	input_data.h\
	jm_constraint.h\
	k_medoids.h\
	logic_variables.h\
	mathematics.h\
	output.h\
	protein.h\
	r_cluster.h\
	rb_cluster.h\
	rmsd.h\
	search_engine.h\
	statistics.h\
	table_constraint.h\
	trailstack.h\
	uniform_constraint.h\
	unique_source_sinks_constraint.h\
	uniqueseq_constraint.h\
	utilities.h\
	variable_fragment.h\
	variable_point.h\

HEADERS=$(HEADERS_CORE:%.h=${H_FOLDER}/%.h)

SOURCES=${MAIN}.cc $(HEADERS:%.h=%.cc) 
OBJECTS = $(SOURCES:%.cc=%.o)

# ---------------------------------------------------------------------
# Compiler options 
# ---------------------------------------------------------------------
CC = g++
#NVCC = /usr/local/cuda/bin/nvcc
DEPEND = g++ -MM
CCOPT =    # -Wall -W -Wno-sign-compare -ansi -pedantic -Werror -Wno-deprecated
LINKOPT =

## debug, don't optimize 
# -std=gnu++0x -std=c++0x -Wstatic-float-init
CCOPT   += -m64 -DIL_STD

## debug, optimize
## use --ptxas-options=-v for cuda (verbose for each kernel)
## use -ftz=false -prec-div=true -prec-sqrt=true
## -L /usr/local/cuda/lib -lcurand
## –use_fast_math
CCOPT += -O3 
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

all:	fiasco

fiasco: $(OBJECTS) 
	$(CC) $(CCLNDIRS) $(DFLAGS) -o $(MAIN) \
		$(OBJECTS:%=$(SOLVER_PATH)/obj/%)

$(OBJECTS): %.o: %.cc
	$(CC) -c $(CCFLAGS) $< -o $(SOLVER_PATH)/obj/$@

clean:
	rm -f obj/host_core/*.o
