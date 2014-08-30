Do the following steps to set your compiler (if needed):
- cd jnetsrc/src
- edit CC and CFLAGS lines in the Makefile to reflect your 
  compiler and optimiser
- cd ..
- edit CC and CFLAGS lines in the Makefile to reflect your 
  compiler and optimiser
To compile:
- run the script “CompileAndRun.sh”

Use:
	 ./cocos -h
to print a help message.

In proteins/ there are some examples. 
In particular, you may want to try:
	./cocos -i proteins/1zdd.fa -a -v
that takes just a FASTA file as input.
Otherwise you can try:
	./cocos -i proteins/1ZDD.in.cocos -v
that uses a complete input file.

News (6/11/14): Ongoing implementation of docking analysis algorithms!
Try out an example:
	 ./cocos -i Docking/PeptideDocking.in.cocos -k 4


%%%%%%%%%%%%%%%%%%%%%%%%%
%%%		      %%%
%%%    INPUT FILES    %%%
%%%		      %%%
%%%%%%%%%%%%%%%%%%%%%%%%%

Input files for COCOS set the options for either Ab_initio prediction or Docking.
In what follows we describe how such files must be formatted.

- The first line must identify the type of job the tool has to perform.
  In particular, the user should specify “AB_INITIO” prediction or “DOCKING” analysis.
  Note: if the line is missing, the system will ask the user which kind of computation the
	user wants the tool to perform.
- The next lines identify the paths for the databases (i.e., energy tables and angles/domains).
  The user should specify the following paths:
  COULOMBPAR  <path>  % Coulomb tables for contacts
  LJPARAMETER <path>  % Lenard-Jones values
  HDPARAMETER <path>  % Hydrogen distances
  HAPARAMETER <path>  % Hydrogen angles
  CONTACT     <path>  % Contact energies
  TORSPAR     <path>  % Torsional energies
  ANGLES      <path>  % Domains of the FD variables
  If the user does not specify these paths, the system will set “config/” as default path.
- The next line identifies the target protein:
  a) For AB_INITIO the user can set 
	TARGET_PROT <path> % path of the pdb containing the target (used as known structure)
	KNOWN_PROT  <path> % path to the pdb containing the known protein
	<fasta>            % Fasta sequence of the target
			   % For example,
 			   % >SEQUENCE_1
			   % MTEITAAMVKELR…
  b) For DOCKING the user can set
	PEPTIDE <path> % path of the pdb containing the peptide
- Next lines identify either the agents or the Constraints depending on AB_INITIO or DOCKING analysis. 

%%%%%%%%%%%%%%%%%%%%%%%%% 
%%%     AB_INITIO     %%%
%%%%%%%%%%%%%%%%%%%%%%%%%
The Supervisor agent creates a Structure agent for each highly constrained secondary structure element and a single Coordinator agent that folds the entire structure by moving turns or loops. 
Nevertheless, the MA system can handle several structure and coordinator agents. 
Moreover, it is possible to associate a priority among these agent in order to impose a preference on the order of the sub-structures to fold. This options can be chosen by the user through a specific syntax and these setting can be given to the MAS either by command line or written in an input file.Here we present the syntax of the input file in standard BNF.

input       ::= [structure;*][coordinator;*]
structure   ::= secondary int_const int_const priority search
coordinator ::= loop int_const int_const scope priority search
secondary   ::= H | P | S
loop        ::= T | C | epsilon
priority    ::= p int_const
scope       ::= s [int_const, int_const]
search      ::= icm | gibbs | montecarlo
int_const   ::= [[0-9]][[0-9]]*

Structure agents are associated with alpha-helices (H), polyproline II (P), and beta-sheets (S), while coordinator agents can model coils (C), turns (T), or both (epsilon).
Priorities are use to invoke MAS agents with the specific order given by the user.
By default, agents are invoked with the order defined in the input file (i.e., following their definition from top to bottom).
Structure agents use icm as default search algorithm, while coordinator agents use Gibbs. The scope for the Coordinator agents defines the sub-part of the structure considered when energy values are calculated. Note that by default a Coordinator agent is always set on the sets of amino acids that are not associated to any MAS agent. In particular, this coordinator agent calculates the energy considering the whole structure. 

For some examples, see the folder “proteins/”.

%%%%%%%%%%%%%%%%%%%%%%%%% 
%%%      DOCKING      %%%
%%%%%%%%%%%%%%%%%%%%%%%%%
The user must specify 4 constraints:
1) TRANSLATE atom_type AA x y z:
   This constraint translate the peptide considering the atom “atom_type” (i.e., N, C, CA, O, and H) of the
   AAth amino-acid into the position x, y, z.
2) ATOM_GRID path:
   This constraints enables the “atom_grid” constraint. 
   Docking is performed checking clashes between the atoms of the peptide and the atoms specified in the pdb file indicated in path.
3) DOCK_GRID path [dist]: 
   Docking is performed evaluating the number of contacts between the peptide and the dock specified in the pdb file indicated in path. Dist is optional and set the minimum distance in Angstroms for a contact (default: 3 A).
4) SEED x y z r h:
   The user can specify one or more seed where an oc_tree will be placed to sample the search space, where:
   x (real): x seed's coordinate, 
   y (real): y seed's coordinate,
   z (real): z seed's coordinate,
   r (real): half diagonal of the cube centered in (x, y, z),
   h (int) : height of the octree (i.e., number of partitions).
  If the user does not specify any seed, the system will ask the user to insert at least one seed by command line.

For some examples, see the folder “docking/”.

For any questions, feel free to write at: campe8@nmsu.edu.