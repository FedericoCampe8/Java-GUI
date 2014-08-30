FIASCO description and usage

====================
0. Compile and Run
$> make clean
$> make


The Fiasco Fragment DBs are stored in fragment-assembly-db/
To run the solver see Section 5.


====================
1. Brief Description

Following is a brief description of the general structure of FIASCO.
FIASCO (Fragment-based Interactive Assembly for protein Structure 
prediction with COnstraints) is a C++ constraint solver optimized to 
address studies related to the analysis and prediction of the 
three-dimensional structure of proteins.
Its conceptual declarative nature allows an easy manipulation of 
constraints targeted at describing protein sequence and structural 
properties, interaction forces fields, spatial local, global constraints, etc. 
More information about FIASCO can be found in [citation needed].

===================
2. Variables

FIASCO manages two type of variabels:
     - Finite Domain Variables (FDVs): These variables will be instantiated	
     with a unique  non negative integer value. The domain of a ﬁnite domain
     variable is therefore a finite set of non negative integer numbers. In
     this version of the solver these variables are identified as Fragment Variables.
     - Point Variables (PVs): These variables will assume the coordinates of a
     3D point in R3  (with the precision of C++). Their domains are, initially,
     3D boxes identiﬁed by two  opposite vertices ⟨min, max⟩.

All the variables of a problem are stored in the "LogicVariables"  data
structure. 

===================
3. Constraints

The constraints are the fundamental components of FIASCO solver. The Constraint
class is the base class for all constraints. Each constraint is identified by a
univoque id, assigned at its creation. Its weight characterizes the order of its
extraction from the Constraint Store. Is contains the following:
   - point_variables (vpt): A list of references to PVs identifying those
   involved in the constraint.
   - finite_domain_variables (vfrag): A list of references to FDVs
   (FragmentVariables) identifying those involved in the constraint.
   - caused_by_vpt: The list of PVs waking up the constraint in the current arc
   consistency step.
   - caused_by_vfrag: The list of FDVs waking up the constraint in the current
   arc consistency step. 


3.1 Constraints currently implemented:
The type of constraints implemented in the current versions are:

DEFAULT CONSTRAINTS:
  The default constraints specify properties of the amino acid chain, they
  cannot be deactivated by command line for they underlying biological
  relevance.
  1. Alldistant
  It models the non-overlapping chain property: each two amino acid are
  separated by a distance equal to the sum of their covalent radii. 

  2. Centroid
  It models the side chain of an amino acid - computed incrementally.

  3. Fragment
  Implements the Fragment Assembly concept. The Fragment constraint links each
  Fragment Variable with a set of macro-blocks (fragments).
 
OPTIONAL CONSTRAINTS:
  4. Ellipsoid
  Given a list of amino acid A_1, ..., A_k, (k>=1) two focus f1, f2, and the sum
  of the radii it specify an area in which the amino acid 'A_i' should be bound
  in.

  INPUT SYNTAX: 
  --ellipsoid aa_1 .. aa_n : f1= X Y Z f2= X Y Z sum-radii= K
  where:
  - aa_1 .. aa_n, is a list of amino acids (starting from 0) 
  for which CAs will be involved in the constraint
  - f1 and f2 X Y Z, are the coordinates for the two focus
  - K, is the radius sum
  
  5. Bundle
  When two or more rigid-bodies (e.g., secondary structures) can be specified, a
  Bundle constraint ensure that in every solution the bodies positions and space
  orientation will be preserved, as given in input.

  6. JM constraint
  The Joined Multibody constraint, is set between two anchor points: front- and
  end-anchor (included) to generate non-redundant paths between them. It uses an
  approximated propagation algorithm (JMf) that heavily relies on some
  clustering procedure.
  INPUT SYNTAX:
  --jm aa_s {-> | <- | <->} aa_f : numof-clusters= K_min K_max 
                                                     sim-params= R B 
                                                     tollerances= eps_R eps_B
   where:
   - aa_s: is the amino acid (starting from 0) corresponding to the front-anchor
  of the  flexible chain. 
  - aa_f: is the amino acid relative to the end-anchor of the flexible
  chain.
  - '->' | '<-' | '<->', defines the propagation direction, either MONO- or
  BI-directional 
  - K_min, K_max: are the minimum / maximum number of cluster to generate
  - R, B: are the sim-clustering parameters eps_R, eps_B: are the maximum
  tolerances radius (eps_R) and orientation (eps_B) within which an
  end-anchor can be placed.  
 
   DEFAULT VALUES:
   - K_min = max {50, dom-size}
   - K_max = 500
   - R = 0.5 A
   - B = 15 deg
   - eps_R = 1.5 A
   - eps_R = 60 deg
 
   EXAMPLE:
   --jm 88 '->' 96 : numof-clusters= 60 200 \
                               sim-param= 0.5 15 \
                               tolerances= 1.5 30


 7. Unique Source-Sink constraint
 The unique-source-sink constraint between the varaibles V_i and V_j is a
 cardinaliry constraints which given an assignment for V_i, ensures that in the
 solutions pool there is at most one assignment for V_j, for each grid voxel of
 G_j. This constraint associates a grid to the variable V_j. 
 
 SEMANTICS:
 Fixed V_i, \not\exists V_j^m, V_j^n s.t. find_voxel(V_j^m) = find_voxel(V_j^n)
 where:
 - V_j^{k} represents the assignment for variable V_j in the k-th solution
 generated.
 - find_voxel : V -> \mathbb{N} is a function which associate each variable
 assignemnt with its grid voxel index (if any). 
 
 INPUT SYNTAX:
 --unique-source-sinks a_i '->' a_j : voxel-side= K
 where:
 - a_i, is the source atom (the CA_i, with i >= 0)  
 - a_j, is the sink atom (the CA_j, with i >= 0)
 - K, is the side of a voxel in Angstroms.
 
 NOTE:
 The lattice side is now fixed to 200 Angstroms. If you experience any problem
 with the lattice size, please enlarge the lattice side.
 
 To exploit the synergy of the JM and the unique_source_sink constraints use the
 following parameters: 
 if JM is set on: AA_i -> AA_j
 set unique_source_sink on: AA_{i-1} -> AA_j

  
 8. Unique Constraint
 The unique constraint is a spatial constraint which guarantees that a given
 discretization for a chain of adjacent points is visited in at most one
 solution.

 SEMANTICS:
 Consider a 3D lattice approximating the 3D space. We say that \gamma(aa_i)
 gives the lattice position for the amino acid aa_i. An assignment (x_1, ...,
 x_n) of variables P_1, .., P_n satisfy the  unique constraint over a sequence
 of amino acids aa_1, .., aa_n,  if there is no other solution SOL_k
 s.t. (\gamma_k(aa_1) = x_1 \and ... \and \gamma_k(aa_n) = x_n.
  
 INPUT SYNTAX:
 --uniform aa_1 .. aa_n : voxel-side= K [center= X Y Z ]
 where:
 - aa_1 .. aa_n, is a list of amino acids (starting from 0) 
 for which CAs will be involved in the uniform constraint (every a_i will be
 placed using one grid) 
 - K, is the side of a voxel in Angstroms.

NOTE:
The lattice side is now fixed to 200 Angstroms. If you experience
any problem with the lattice size, please enlarge the lattice side.


3.2 Creating New Constraints
   
Creating a new constraint is done by deriving the new constraint from the
"Constraint" class. The following methods need to be declared in the derived
class:
  - bool propagate (int trailtop);
  - bool consistency;
  - bool check_cardinality (size_t& backjump);
  - bool synergic_consistency (const point& p, atom_type t, int aa_idx) ;
  - void reset_synergy ();
  - void dump (bool all);

  The 'propagate' method propagates the effect of the application of the
     constraint over a set of variable.
   The 'consistency' ensures the consistency of the applied constraint.
   The 'check_cardinality' ensures that at least L and at most Y constraints
    will be given when the solver terminates.
 

4. SEARCH ENGINE


=================
5. APPLICATIONS
  FIASCO requires the following input files:
  1. A Fragment data-base.
  In this version the database adopted is the FREAD DB for flexible sequence
  sampling stored in FIASCO_PATH/fragment-assembly/db/FREAD/fread_loop_db.dat
  2. The FIASCO input file ".in.fiasco":
  The input file which specifies the following information:
  % Database information
  FRAGMENTDB  path_to_fragment_db/fread_loop_db.dat
  % Proteins Information
  TARGET_PROT path_to_target_protein/protein-id
  KNOWN_PROT path_to_known_protein/protein-id
  CONSTRAINTS path_to_protein_blocks/'file'.in.con
  FRAG_SEC_FL   path_to_protein_blocks/'file'.in.pdb
  
  where protein-id is the PDB protein id (e.g. 1xpc)
  

  5.1 Ab-initio prediction
  ...

  5.2 Protein Flexibility (1xpc)

  An example of protein flexibility study is done by using the UNIFORM
  constraint applied to a given input. 
  For the 1xpc study the script test-1xpc.sh is set so that the UNIFORM
  constraint is applied to the CA of the amino acids No. 528, 532, 536 (222,
  226, 230 in the Fiasco representation -- where the first amino acid of the
  sequence is AA_0) with a voxel side of 1 Angstrom. And the CA 540 (AA_233) is
  constrained in an ellipsoid with focus:f1=28.15, -2.27, 9.97 f2=38.51, -17.84,
  28.35 and radii sum of 33 Angstrom. 
  To run the test with this setting execute:
     
  ./test-1xpc.sh proteins/1xpc

  An output file (1xpc.out.pdb) will be saved under /proteins/

  To generate approximatively 1ML solutions for the flexibility study use the
  following parameters:
  
  --jm 216 '->' 222  : numof-clusters= $domain_size 100 \
                                    sim-params= 1.0 60 \
  --jm 223 '->' 227  : numof-clusters= $domain_size 100 \
                                   sim-params= 1.0 60 \
  --jm 228 '->' 230  : numof-clusters= $domain_size 100 \
                                   sim-params= 1.0 60

  In general the number of solution generable by the solver is bounded from the
  above by \prod_i k_max_i (for all i = 1...#jm)


  5.3 Loop Sampling
   ...
  
