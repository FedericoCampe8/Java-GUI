Java-GUI
========================================

The tool is based on the FIASCO (1) and CoCoS (2) solvers. Please cite:

1. Federico Campeotto, Alessandro Dal Palù, Agostino Dovier, Ferdinando Fioretto, Enrico Pontelli.
A Constraint Solver for Flexible Protein Model. J. Artif. Intell. Res. (JAIR) 48: 953-1000 (2013)

2. Federico Campeotto, Agostino Dovier, Enrico Pontelli: Protein Structure Prediction on GPU.
A Declarative Approach in a Multi-agent Framework. ICPP 2013: 474-479

For further information please visit http://www.cs.nmsu.edu/fiasco/

========================================

GUI Interface for Protein Analysis 
This is the GUI interface to coordinate both CoCoS and FIASCO.
It is a Java user interface which allows the user to load proteins in pdb/fasta format, select protein fragments, run CoCoS to determine the folding and then run FIASCO to compute/modify loops.

The interface is composed by three main panels:

1) The "Extraction" panel: this is the panel used to load and visualize a known protein structure It allows the user to select protein fragments. If needed, the user can run CoCoS from a fasta sequence to predict a structure.

2) The "Assembling" panel: this is the panel where the fragments are loaded into. It allows the user to impose some geometrical constraints (see the FIASCO manual).

3) The "Output" panel: it visualizes the results computed by FIASCO, and it allows the user to transfer a structure on the Extraction panel to repeat the process again.

Getting Started:
- Download the Java-Gui tool
- Unpackage the zip file and then go to the Java-Gui directory
- Launch the jafatt.jar file (usually to perform this action you need to open the shell prompt and then from the path to the Java-Gui directory launch java -jar jafatt.jar)
- The first time you launch the Java-Gui tool you need to compile the solvers. You can now use the Java-Gui built-in tool.
  - From the main window select Tools->Compile. A new window will appear. From the Compile window select the solvers you need to compile (CoCos and FIASCO are already selected, CoCosGPGU
    isn't. This is because the CoCosGPGU solver needs the CUDA framework to work, so make sure you have CUDA on your device before even compiling).
- If the compile process ended up successfully you can now use the Java-Gui tool.
- Select a protein (in Fasta or Pdb format) by clicking the folder icon in the top-left corner (or select File->Load Target). You can even download a protein form the rcsb.com protein data bank (select
  the world icon and then type in the name of the protein you want to download, or File->Download Protein). Or you can build your own protein (File->Build Target) by typing the amino acid sequence you may
  want to analyze.
- Press now the gear icon to run the CoCos solver on the uploaded protein (see the CoCos reference manual for further informations).
- Now you have your 3D protein prediction on the Extraction Panel. You can now select some fragments from the protein by clicking the protein itself (one click to amino acid i and one to amino acid j forms the fragment F[i,j], |i-j| amino acids long) or using the select button (for a correct selection make sure you have selected at least two fragments, each of which at least two amino acids long). You can now transfer them to the Assembling Panel, with the Assemble button.
- Now you have the fragments on the Assembling Panel, press the arrow button upon the Solve button to open the Tool Box. Select one or more fragments, by clicking on it(them) (if the selection is successfull the fragment(s) will change color) and then press the circular arrow button, or the four arrows button. Now if you drag the mouse on the Assembling panel you will see the fragment(s) move. If you selected the previous button you will see the fragment(s) spin, if you selected the latter you will see the fragment(s) shift. (Hint: press the ruler button to activate measurements for a better adjustment).
- You can now run the FIASCO Solver. Press the Solve Button, set the constraints to the model (click the question-mark button for further informations about the constraints or see the FIASCO reference manual) and then press Run.
- You now have the 3D protein on the Output Panel, press the Extract button to transfer the protein on the Extraction panel and start over again. If FIASCO produced more than one acceptable model a new window may appear where you can select the model to transfer.

Note: If the compile process doesn't work try to compile the solver on your own. Go to the Java-Gui path and then, from shell type in
- cd Solver
- cd Solver/Cocos
- ./CompileAndRun.sh
- cd ..
- cd Solver/Fiasco
- make clean
- make
- cd ..

If you want to compile the CoCosGPU Sover too, then type in 

- cd Solver
- cd Solver/CocosGPU
- ./CompileAndRun.sh
- cd ..

Note: the "src" folder contains the java files, while the "Solver" folder contains the folders of the Solvers.
Note: This is an ongoing project. Some bugs may be present!

Have Fun!

