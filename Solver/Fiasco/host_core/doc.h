/*! \mainpage FIASCO Index Page
 *
 * \section brief_sec Brief Description
 *
 * Following is a brief description of the general structure of FIASCO:
 *
 * FIASCO (Fragment-based Interactive Assembly for protein Structure 
 * prediction with COnstraints) is a C++ constraint solver optimized to 
 * address studies related to the analysis and prediction of the 
 * three-dimensional structure of proteins.
 * Its conceptual declarative nature allows an easy manipulation of 
 * constraints targeted at describing protein sequence and structural 
 * properties, interaction forces fields, spatial local and global 
 * constraints, etc. 
 *
 *
 * \subsection constraint_sec Constraint 
 *
 * The Constraint class is the base class for all constraints.
 * The type of constraints implemented in the current versions are:
 *
 * DEFAULT CONSTRAINTS:
 * The default constraints specify properties of the amino acid chain, they cannot 
 * be deactivated by command line for they underlying biological relevance.
 * - Alldistant - models the non-overlapping chain property: each two amino acid 
 *   are separated by a distance equal to the sum of their covalent radii. 
 * - Centroid   - models the side chain of an amino acid - computed incrementally.
 * - Fragment   - implements the Fragment Assembly concept. The Fragment constraint
 *   links each Fragment Variable with a set of macro-blocks (fragments). 
 *
 * OPTIONAL CONSTRAINTS:
 * - Ellipsoid  - Given a list of amino acid A_1, ..., A_k, (k>=1) two focus f1, f2, and 
 *   the sum of the radii it specify an area in which the amino acid 'A_i' should be 
 *   bound in.
 *   SYNTAX: 
 *             --ellipsoid aa_1 .. aa_n : f1= X Y Z f2= X Y Z sum-radii= K
 *   where:
 *   - aa_1 .. aa_n, is a list of amino acids (starting from 0) 
 *     for which CAs will be involved in the constraint
 *   - f1 and f2 X Y Z, are the coordinates for the two focus
 *   - K, is the radius sum
 *
 * - Bundle     - When two or more rigid-bodies (e.g., secondary structures) can be 
 *   specified, a Bundle constraint ensure that in every solution the bodies positions
 *   and space orientation will be preserved, as given in input.
 * - JM constraint - The Joined Multibody constraint, is set between two anchor points 
 *   (start- and end-anchor) in order to (attempt) to find non-redundant paths between 
 *   them. It uses a sophisticated form of propagation that hevly relies on a clustering
 *   procedure. 
 *   SYNTAX:
 *
 *             --jmf front-Anchor-AA end-anchor-AA : min_no_clusters max_no_clusters
 *             cluster-tolerance beta_angle (steps to ignore).
 *
 *   Uniform - This constraint builds a grid where a voxel can be occupied at most one 
 *   time and my at most one of the amino acids in A_1, ..., A_k (for all the ensembles 
 *   generated).
 *   SYNTAX:
 *             --uniform aa_1 .. aa_n : voxel-side= K [center= X Y Z ]
 *   where:
 *   - aa_1 .. aa_n, is a list of amino acids (starting from 0) 
 *     for which CAs will be involved in the uniform constraint
 *     (every a_i will be placed using one grid)
 *   - K, is the side of a voxel in Angstroms.
 *
 *   NOTE: The lattice side is now fixed to 200 Angstroms. If you experience
 *   any problem with the lattice size, please enlarge the lattice side.
 *
 *
 *   Each of the above is implemented as a derived class from 'Constraints'.
 *   Moreover each constraint has tree fundamental methods:
 *   1. propagate()   - it propagates the effect of the application of the constraint over 
 *   a set of variable.
 *   2. consistency() - it ensures the consistency of the applied constraint.
 *   3. check_cardinality() it ensures that at least L and at most Y constraints will be 
 *   given when the solver terminates.
 *
 *
 * * \subsection search_engine_sec Search Engine
 * TODO
 *
 * \subsection trailstack_sec Trail Stack and Constraint Store 
 * (TODO)
 */
