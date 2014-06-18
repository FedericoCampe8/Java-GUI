package jafatt;

import java.io.File;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.PrintWriter;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.Scanner;
import java.util.ArrayList;
import java.util.NoSuchElementException;
import org.biojava.bio.BioException;
import org.biojava.bio.seq.Sequence;
import org.biojava.bio.seq.SequenceIterator;
import org.biojava.bio.seq.io.SeqIOTools;
import org.biojava.bio.structure.Chain;
import org.biojava.bio.structure.Structure;
import org.biojava.bio.structure.io.PDBFileReader;
import org.biojava.bio.symbol.*;

public class Controller{
    
    public int currentPanel = Defs.EXTRACTION;
    
    public String[] infoPanel;
    
    private UserFrame view;
    private ProteinModel model;
    private String loadPath;
    
    /* Extraction */
    private boolean isFirstPickExtraction;
    private String[] lastPickEInfo;
    
    /* Assembling */
    private int fragmentNumber;
    private boolean isPDB;
    
    public Controller(){
        setup();
    }
    
    /* Setup variables */
    private void setup(){
        isFirstPickExtraction = true;
        fragmentNumber = 0;
    }//setup
    
    /***************************************** 
     *****************************************
     *                                       *   
     *           GENERAL METHODS             *
     *                                       *
     *****************************************   
     *****************************************/
    
    /* Print a string */
    public void printStringLn(String str){
        view.printStringLn(str);
    }//printStringLn
    
    /* Print a string */
    public void printStringLn(String str, boolean b){
        view.printStringLn(str, b);
    }//printStringLn
    
    /* Set mvc components */
    public void setViewMod(UserFrame view, ProteinModel model){
        this.view = view;
        this.model = model;
    }//setWorkPanel
    
    /* Set the current panel selected */
    public void setCurrentPanel(int currentPanel){
        this.currentPanel = currentPanel;
    }//setCurrentPanel
    
    /* Load a structure from either Target panel or Extraction panel */
    public boolean loadStructure(String path, int panel, boolean pdbOrNot){
        boolean ok;
        
        /* Set selected path */
        loadPath = path;
        if(panel == Defs.TARGET){
            ok = loadTarget( pdbOrNot ); 
        }else{
            ok = loadProtein();
        }
        
        /* Return */
        return ok;
    }//loadStructure
    
    /* Load the target structure */
    private boolean loadTarget( boolean isPDB ){
        String seq;
        Sequence sequence = null;
        
        this.isPDB = isPDB;
        
        /* Target Structure */
        Structure targetStructure;
        if(isPDB){
            
            /* PDB file reader */
            PDBFileReader pdbReader = new PDBFileReader();
            
            /* Chain made by residues */
            Chain chain;
            
            /* Load structure from PDB file */
            try{
                targetStructure = pdbReader.getStructure(loadPath);
                chain = targetStructure.getChain(0);
            }catch(Exception e){
                 printStringLn("Error on loading Target "
                         + "(pdbReader or Chain): " + e);
                return false;
            }
            seq = chain.getSeqResSequence();
            try{
                sequence = chain.getBJSequence();
            }catch(IllegalSymbolException ex){
                printStringLn("Error with target's Symbols: " + ex);
                return false;
            }
        }else{
            try{
                
                /* Read target from Fasta format, initialize the buffer reader */
                BufferedReader br;
                try{
                    br = new BufferedReader( new FileReader(loadPath) );
                }catch(FileNotFoundException fnfe){
                    printStringLn("FASTA file not found: " + fnfe);
                    return false;
                }
                
                /* Set the iterator for reading the FASTA sequence */
                SequenceIterator iterator = SeqIOTools.readFastaProtein(br);
                
                /* Read every single sequence from FASTA format. It must be one.
                 Add multiple sequences in future developments */
                while(iterator.hasNext()){
                    sequence = iterator.nextSequence();
                }
                 
                /* Stringify this symbol list */
                seq = sequence.seqString();
                
                /* Close the bufered reader */
                try{
                    br.close();
                }catch(IOException ioe){
                    printStringLn("Error to close FASTA file: " + ioe);
                }
            }catch(BioException be){
                printStringLn("Error BioException (not FASTA format or"
                        + " wrong alphabet: " + be);
                return false;
            }catch(NoSuchElementException nse){
                printStringLn("No FASTA sequences in the file: " + nse);
                return false;
            }     
        }
        
        if(!seq.equals("")){
            model.setTargetString(seq);
            model.setTarget(sequence);
            model.setTargetPath(loadPath, isPDB);
            
            /* Debug */
            /* printStringLn(seq);*/
            
            /* Return */
            return true;
        }else{
            printStringLn("Failed to load the target");
            return false;
        }
    }//loadTarget
    
    /* Actions made by user when he picks on panels:
     * [Num.Protein][Res_Name][Res_num][Atom][Ax][Ay][Az]
     */
    public void usePickedInfo(String[] info){
        String pnl;
        
        infoPanel = info;
        
        /* Print info */
        if(currentPanel == Defs.EXTRACTION)
            pnl = "EXTRACTION";
        else
            pnl = "ASSEMBLING";
        
        /* Print info */
        printStringLn("Atom " + info[3] + " selected from residue " + info[2] 
                       + " (" + info[1] + ")");
        
        /* Differentiate between panels */
        if(currentPanel == Defs.EXTRACTION)
            pickOnExtraction(info);
        else
            pickOnAssembling(info);
    }//usePickedInfo
    
    /* Check on panel past in input */
    public void usePickedInfo(String[] info, int panel){
        String pnl;
        
        /* Print info */
        if(panel == Defs.EXTRACTION)
           pnl = "EXTRACTION";
        else
           pnl = "ASSEMBLING";
        
        printStringLn("Atom " + info[3] + " selected from residue " + info[2] 
                       + " (" + info[1] + ") on " + pnl);
        
        /* Differentiate between panels */
        if(panel == Defs.EXTRACTION)
            pickOnExtraction(info);
        else
            pickOnAssembling(info);
    }//usePickedInfo
    
    public String[] getInfo(){
        return infoPanel;
    }
    
    public String getPath(){
        return loadPath;
    }
    
    /* Color a specific fragment */
    public void colorFragment(Fragment frg, String color, int panel){
        view.upView.colorFragment(frg, color, panel); 
    }//colorFragment
    
    
    
    /***************************************** 
     *****************************************
     *                                       *   
     *           EXTRACTION PANEL            *
     *                                       *
     *****************************************   
     *****************************************/
    
    
    
    /* It means that no operations are allowed if a user has done only the first
     * pick on the structure
     */
    public boolean getPermissionExtraction(){
        return isFirstPickExtraction;
    }//getPermissionExtraction
    
    /* Set different views on Extraction panel or Jmol Panel */
    public void setViewOptions(String viewOption){
        String switchViewString;
        switchViewString = Utilities.switchViewString(viewOption);
        ((ExtractionPanel)view.getPanel(Defs.EXTRACTION)).switchView(switchViewString);    
    }//setViewOptions
    
    /* Load the protein structure from witch extract fragments */
    private boolean loadProtein(){
        boolean ok = false;
        
        /* Protein Structure and Sequence */
        Structure proteinStructure = null;
        
        /* PDB file reader */
        PDBFileReader pdbReader = new PDBFileReader();
        
        /* Read the structure */
        try{
            proteinStructure = pdbReader.getStructure(loadPath);
            
            /* Print some infos */
            printStringLn("");
            printStringLn("" + proteinStructure, false);
        }catch(IOException ioe){
            
            /* Usually if this happens something is wrong with the PDB header
             * e.g., 2bdr - there is no Chain A, although it is specified in 
             * the header
             */
            printStringLn("Error on loading the PDB file: " + ioe);
            return ok;
        }
        /* Set the current protein structure */
        ok = model.setCurrentProtein(proteinStructure, loadPath);

        /* Print some infos (debug) */
        /* System.out.println("Structure:");
        System.out.println(proteinStructure); */
        
        /* Return */
        return ok;
    }//loadProtein
    
    public void loadProteinRmsd(){
        view.rmsd.loadProtein(loadPath);        
    }
    
    /* Check if a residue is already present in the Extraction panel */
    public boolean isResidueAlreadyPresent(String info[]){
        boolean isAlreadyPresent;
        int i = -1;
        
        /* Check if the residue contained in info is already present in a 
         * selected fragment */
        i = model.isFragmentPresent(info);
        isAlreadyPresent = i >= 0;
        
        /* Return */
        return isAlreadyPresent;    
    }//isFragmentInfoPresent
    
    /* Work with infos from Extraction panel */
    private void pickOnExtraction(String[] info){
        boolean isAlreadyPresent;
        
        /* Verify if the users wants to delete a fragment from the previous
         selection*/
        int i = model.isFragmentPresent(info);
        isAlreadyPresent = i >= 0;
        
        /* If present, delete it, else select it*/
        if(isAlreadyPresent){
            
            /* Check if it is the first or the second selection */
            if(isFirstPickExtraction)
                model.deleteFragmentE(i);
            else{
                printStringLn("AA already selected");
                
                /* Back to the initial state */
                view.upView.colorAtom(lastPickEInfo, Defs.COLOR_DESELECT_FRAGMENT,
                        Defs.EXTRACTION);
                isFirstPickExtraction = true;
                return;
            }
        }else{
            
            /* Need two picks for selecting a fragment */
            if(isFirstPickExtraction){
                
                /* Saving infos for the first pick */
                lastPickEInfo = info;
                view.upView.colorAtom(lastPickEInfo, Defs.COLOR_SELECT_AN_ATOM, 
                        Defs.EXTRACTION);
                isFirstPickExtraction = false;
            }else{
                /* Set a new fragment */
                view.upView.colorAtom(info, Defs.COLOR_SELECT_AN_ATOM, 
                        Defs.EXTRACTION);
                model.addFragmentE(lastPickEInfo, info);
                isFirstPickExtraction = true;
            }
        }
    }//pickOnExtraction
    
    /* Set the offset for the fragments */
    public void setOffsetsOnFragments(String[] offsets){
        int numFragments = model.getAllFragmentsE().size();
        
        /* Set offsets and Write the mapping of the offsets on the Target panel */
        for(int i = 0; i < numFragments; i++){
            model.setCurrentFragmentE(model.getAllFragmentsE().get(i));
            model.getAllFragmentsE().get(i).setOffset(offsets[i]);
        }
    
        /* Info string */
        printStringLn("Offset set on fragments");
    }//setOffsetsOnFragments
    
    /* Transfer the selected fragments from Extraction panel to 
     * Assembling panel. First it saves the fragments on a file and then it
     * loads the saved file on Assembling panel.
     */
    public void transferFragments(){
        
        /* Picks up the fragments */
        ArrayList<Fragment> fragments = model.getAllFragmentsE();
        int numFragments = fragments.size();
        
        /* For each fragment selected do as follows */
        for(int i = 0; i < numFragments; i++){
            
            /* Set the unique number for the fragment */
            fragmentNumber++;
            fragments.get(i).setFragmentNumber("" + fragmentNumber);
            
            /* Create a new fragment to add to the fragments present on the
             * Assembling panel */
            String[] info1 = new String[7];
            String[] info2 = new String[7];
            
            for(int j = 0; j < info1.length; j++){
                info1[j] = Defs.EMPTY;
                info2[j] = Defs.EMPTY;
            }
            
            Fragment newFragment = new Fragment(model);
            
            /* Clone the old fragment into the new one */
            newFragment.clone(fragments.get(i));
            
            
            /* Add the fragment on the array representing the fragments
             on the Assembling panel */
            model.addFragmentA(newFragment);   
        }
        
        view.upView.execute("load " + Defs.PROTEINS_PATH+model.idProteinCode+".in.pdb ; ", Defs.ASSEMBLING);
        view.upView.prepareStructure(Defs.ASSEMBLING);
        
        /* Connect the fragments on the Assembling panel */
        view.upView.connectFragments(true);
        
        /* Delete the fragments from Extraction panel */
        for(int i = 0; i < numFragments; i++)
            model.deleteFragmentE(0);
        
        ((TargetPanel)view.getPanel(Defs.TARGET)).reset();
        
        setup();
        
    }//transferFragments
    
    
    
    /***************************************** 
     *****************************************
     *                                       *   
     *           ASSEMBLING PANEL            *
     *                                       *
     *****************************************   
     *****************************************/
    
   
    
    /* Work with infos from Assembling panel */
    private void pickOnAssembling(String[] info){
        /* Set the fragment corresponding to info */
        //model.setCurrentFragmentA(info[Defs.INFO_NUMPROT]);  // DEPRECATED
        //if(model.isFragmentSelected() && model.getCurrentFragmentA()!=null)
        /*if(Integer.parseInt(info[Defs.FRAGMENT_NUM]) >=
                Integer.parseInt(model.getCurrentFragmentA().getParameters()[Defs.FRAGMENT_START_AA]) &&
                Integer.parseInt(info[Defs.FRAGMENT_NUM]) <=
                Integer.parseInt(model.getCurrentFragmentA().getParameters()[Defs.FRAGMENT_END_AA]))                
            deselect();
        else*/
            model.selectFragmentA(info[Defs.FRAGMENT_NUM]);
        
    }//pickOnAssembling
    
    /* Deselect all */
    public void deselect(){
        //model.setCurrentFragmentA((Fragment) null);
    }//deselect
    
    public void saveToPdb(){
        
        String text = "";
        String line;
        int aa, startFrg, endFrg;
        ArrayList<Fragment> fragments = model.getAllFragments(Defs.ASSEMBLING);
        
        try {
            Scanner scanner = new Scanner(new File(Defs.TEMP + "cache.pdb"));
          
            for (int k = 0; k < fragments.size(); k++) {

                startFrg = Integer.parseInt(fragments.get(k).getParameters()[Defs.FRAGMENT_START_AA]);
                endFrg = Integer.parseInt(fragments.get(k).getParameters()[Defs.FRAGMENT_END_AA]);

                while (scanner.hasNextLine()) {
                    line = scanner.nextLine();
                    String[] parsedLine = line.split("\\s+");

                    if (parsedLine[0].equals("ATOM")) {
                        aa = Integer.parseInt(parsedLine[5]);
                        if ((parsedLine[2].equals("N")) && (aa == startFrg)) {
                            line = scanner.nextLine();
                            line = scanner.nextLine();
                            parsedLine = line.split("\\s+");
                        }

                        if ((parsedLine[2].equals("N")) && (aa == endFrg)) {
                            text = text + line + "\n";
                            while (!parsedLine[0].equals("ENDMDL")) {
                                line = scanner.nextLine();
                                parsedLine = line.split("\\s+");
                            }
                            text = text + line + "\n";
                            break;
                        }
                    }
                    //if (!parsedLine[0].equals("ENDMDL"))
                        text = text + line + "\n";
                }

            }
            scanner.close();
          
        } catch (IOException e) {
            System.out.print(e);
        }

        try{
            FileOutputStream inpdb = new FileOutputStream(Defs.PROTEINS_PATH +model.idProteinCode+".in.pdb");
            inpdb.close();
            PrintWriter out = new PrintWriter(new BufferedWriter(
                    new FileWriter(Defs.PROTEINS_PATH + model.idProteinCode +".in.pdb", true)));
            out.print(text);
            out.close();
        }catch (IOException e) {
           view.printStringLn("Error: " + e);
        }
    }
    
    /* Impose to move all binded fragments */
    public void moveAllBindedFragments(Boolean move){
        
    }//moveAllBindedFragments
    
    /* Set a constraint on a fragment */
    public void setConstraint(int constraint){
        Fragment currentFragment = model.getCurrentFragmentA();
        
        printStringLn(Utilities.setConstraintString(constraint, currentFragment));
        switch(constraint){
            case Defs.CONSTRAINT_BLOCK:
                String[] infoBlock = getGroupBlock(currentFragment);
                if(infoBlock == null){
                    printStringLn("Constraint didn't set");
                    return;
                }else
                    model.setConstraintOnFragment(Defs.CONSTRAINT_BLOCK, infoBlock);
                break;
            case Defs.CONSTRAINT_VOLUME:
                break;
            case Defs.CONSTRAINT_EXACTD:
                break;
            case Defs.CONSTRAINT_WITHIND:
                break;
            case Defs.CONSTRAINT_THESE_COORDINATES:
                break;
        }
        
    }//setConstraint
    
    /* Request the block from the View */
    private String[] getGroupBlock(Fragment currentFragment){
        printStringLn("Select a block group for the fragment");
        
        /* Return the resulting block */
        return view.upView.requestBlock(currentFragment);
    }//requestBlock
   
}//Controller
