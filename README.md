Java-GUI
========

GUI Interface for Protein Analysis 
This is the GUI interface to coordinate both CoCoS and FIASCO.
It is a Java user interface which allows the user to load proteins in pdb/fasta format, select protein fragments, run CoCoS to determine the folding and then run FIASCO to compute/modify loops.

The interface is composed by three main panels:
1) The "Extraction" panel: this is the panel used to load and visualize a known protein structure It allows the user to select protein fragments. If needed, the user can run CoCoS from a fasta sequence to predict a structure.
2) The "Assembling" panel: this is the panel where the fragments are loaded into. It allows the user to impose some geometrical constraints (see the FIASCO manual).
3) The "Output" panel: it visualizes the results computed by FIASCO, and it allows the user to transfer a structure on the Extraction panel to repeat the process again.

How to use it:
- cd solver
- cd solver/cocos
- ./CompileAndRun.sh
- cd ..
- make 
- jafatt.jar

Note: This is an ongoing project. Some bugs may be present!


