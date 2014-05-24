package jafatt;

import java.io.File;
import java.io.BufferedWriter;
import java.io.PrintWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Scanner;
import javax.swing.JOptionPane;

public class UpdateView{
    
    private UserFrame view;
    private ExtractionPanel extractionPanel;
    private AssemblingPanel assemblingPanel;
    private OutputPanel outputPanel;
    private ProteinModel model;
    private int i = 0;
    
    public UpdateView(UserFrame view, ProteinModel model){
        setup(view, model);
    }
    
    /* Setup parameters */
    private void setup(UserFrame view, ProteinModel model){
        this.view = view;
        this.model = model;
        extractionPanel = (ExtractionPanel)view.getPanel(Defs.EXTRACTION);
        assemblingPanel = (AssemblingPanel)view.getPanel(Defs.ASSEMBLING);
        outputPanel = (OutputPanel)view.getPanel(Defs.OUTPUT);
    }//setup
    
    /* Set a new Structure on Extraction panel */
    public void newStructure(){
        if(model.totalLoadedProteins >= 1)
                execute("zap; ", Defs.EXTRACTION);
            
        /* Visualize the new protein on the Extraction panel */
        execute("load " + model.getCurrentPath() + "; ", Defs.EXTRACTION);
        
        /* Set the current structure on Extraction Panel */
        extractionPanel.setProtein(model.getCurrentProtein());
        assemblingPanel.setProtein(model.getCurrentProtein());
        outputPanel.setProtein(model.getCurrentProtein());
        
        /* Prepare the view */
        prepareStructure(0, Defs.EXTRACTION);
    }//newStructure
    
    /* Prepare the view on Jmol panels */
    public void prepareStructure(int numStructure, int panel){
        if(panel == Defs.EXTRACTION)
            extractionPanel.executeCmd(Defs.COMMAND_PREPARE_STRUCURE);
        else{
            assemblingPanel.executeCmd(
                    Utilities.prepareStructureString(numStructure)
                    );
            view.upView.execute("set allowMoveAtoms TRUE; ", Defs.ASSEMBLING);
            view.upView.execute("set allowModelKit TRUE; ", Defs.ASSEMBLING);
        }
    }//prepareStructure
    
    /* Color a specific fragment */
    public void colorFragment(Fragment frg, String color, int panel){
        if(panel == Defs.EXTRACTION)
            extractionPanel.colorFragment(frg, color);
        else
            assemblingPanel.colorFragment(frg, color);
    }//colorFragment
    
    /* Color all fragments */
    public void colorAllFragments(String color, int panel){
        ArrayList<Fragment> fragments = model.getAllFragments(panel);
        Fragment frg;
        int z = fragments.size();
        int source = -1;
        if(panel == Defs.ASSEMBLING_S){
            source = panel;
            panel = Defs.ASSEMBLING;
        }        
        
        if(z == 0 && panel == Defs.ASSEMBLING)
            assemblingPanel.executeCmd("select 0; ");
        
        /* Color each single fragment */
        for(int i = 0; i < z; i++){
            frg = fragments.get(i);
            if(color.equals(Defs.COLOR_DESELECT_FRAGMENT))
                colorFragment(frg, color, panel);
            else
                colorFragment(frg, color, panel);
        }
        if (source == Defs.ASSEMBLING_S)
            assemblingPanel.executeCmd(Utilities.selectFragmetString(fragments));
    }//colorAllFragments
    
    /* Color a specific AA */
    public void colorAtom(String[] info, String color, int panel){
        if(panel == Defs.EXTRACTION)
            extractionPanel.colorAtom(info[3], info[0], color);
    }//colorAA
    
    /* Execute a command on panel "pnl" */
    public void execute(String cmd, int pnl){
        if(pnl == Defs.EXTRACTION)
            extractionPanel.executeCmd(cmd);
        else
            assemblingPanel.executeCmd(cmd);
    }//execute
    
    /* Get the target sequence string */
    public String getTargetSequence(){
        return model.getTargetSequenceString();
    }//getTargetSequence
    
    /* Get all the fragments present on Extraction panel */
    public ArrayList<Fragment> getAllFragmentsExtraction(){
        return model.getAllFragments(Defs.EXTRACTION);
    }//getAllFragments
    
    /* Get all the fragments present on Extraction panel */
    public ArrayList<Fragment> getAllFragmentsAssembling(){
        return model.getAllFragments(Defs.ASSEMBLING);
    }//getAllFragments
    
    /* Create a new fragment */
    public Fragment createFragment(String[] infoAA1, String[] infoAA2){
        return new Fragment(infoAA1, infoAA2, model);
    }//createFragment
    
    public void deleteAllFragmentsAssembling(){
        model.deleteFragmentA();
        i = 0;
    }
    
    /* Set an offset string on Target panel */
    public void setOffsetOnTarget(Fragment frg){
        /* Get the offset of the fragment */
        int offset = -1;
        try{
            offset = Integer.parseInt(frg.getParameters()[Defs.FRAGMENT_OFFSET]);
        }catch(NumberFormatException nfe){}
        
        /* Debug */
        /* System.out.println("Offset to set: " + offset + " offset of protein: " 
                + model.getCurrentProteinOffset() + " String fragment: " 
                + frg.getParameters()[Defs.FRAGMENT_SEQUENCE_STRING]); */
        
        ((TargetPanel) view.getPanel(Defs.TARGET)).setOffsetString(
                frg.getParameters()[Defs.FRAGMENT_SEQUENCE_STRING],
                offset);
    }//setOffsetOnTarget
    
    /* Chech if an old fragment is transfering again */
    public boolean checkRightTransfer(){
        return model.fragmentsAlreadyPresent();
    }//checkRightTransfer
    
    /* Add a fragment on the Assembling panel */
    public void addFragment(Fragment frg){
        
        /* Select the fragment from Extraction panel */
        extractionPanel.executeCmd(Utilities.selectFragmetString(frg));
        
        /* Save the fragment on a PDB file */
        extractionPanel.executeCmd(Defs.COMMAND_SAVE_ON_PDB);
        String cmd;
        int start = Integer.parseInt(frg.getParameters()[Defs.FRAGMENT_START_AA]);
        int end = Integer.parseInt(frg.getParameters()[Defs.FRAGMENT_END_AA]);
        System.out.println(start + " " + end);
        cmd = "select resno >=" + start + " and resno <=" + end;
        extractionPanel.executeCmd(cmd);
        //script = viewer.scriptWait("load append pdb::" + Defs.path_prot + structure.getPDBCode() + ".in.pdb" + "\"; ");
        
        extractionPanel.executeCmd("write pdb \"" + "fragment.pdb" + "\"; ");
        
        //extractionPanel.executeCmd("load pdb \"" + Defs.path_prot + model.getCurrentProtein().getPDBCode() + ".in.pdb" + "\"; ");
        //extractionPanel.executeCmd("append pdb \"" + Defs.path_prot + model.getCurrentProtein().getPDBCode() + ".in.pdb" + "\"; ");
        //extractionPanel.executeCmd("load append pdb::" + Defs.path_prot + model.getCurrentProtein().getPDBCode() + ".in.pdb" + "; ");
        //extractionPanel.executeCmd("write pdb \"" + Defs.path_prot + model.getCurrentProtein().getPDBCode()+ "[" + (start -2) + "," + end +"]" + ".in.pdb" + "\"; ");
        //cmd = "select resno >=" + (end + 1) + " and resno <=" + (end + 1);
        //extractionPanel.executeCmd(cmd);
        //extractionPanel.executeCmd("write pdb \"" + "next.pdb" + "\"; ");
        
        String text = "";
        String line;
        
        try {
        //text = new Scanner(new File(Defs.DUMMY_PDB_FILE)).useDelimiter("\\A").next();
 
          Scanner scanner = new Scanner(new File("fragment.pdb"));
          //Scanner scr = new Scanner(new File("next.pdb"));
          //ignore = scanner.nextLine();
          //ignore = scanner.nextLine();
          
          while (scanner.hasNextLine()) {
              line = scanner.nextLine();/*
              String[] parsedLine = line.split("\\s+");

              if ((parsedLine[2].equals("N")) && (Integer.parseInt(parsedLine[5]) == start)) 
                  line = scanner.nextLine();

              if ((parsedLine[2].equals("N")) && (Integer.parseInt(parsedLine[5]) == end)) {
                  text = text + line + "\n";
                  break;
              }*/
              text = text + line + "\n";
          }
          //text = text + scr.nextLine() + "\n";
          scanner.close();
          //text = new Scanner(new File(Defs.DUMMY_PDB_FILE)).useDelimiter("\n").next();
        } catch (IOException e) {
        }
        
        try {
            PrintWriter out = new PrintWriter(new BufferedWriter(
                    new FileWriter(Defs.path_prot + model.idProteinCode +".in.pdb", true)));
            out.println("MODEL " + ++i);
            out.print(text);
            out.println("ENDMDL");
            out.close();
        } catch (IOException e) {
        }
        
        /* Load that PDB file on Assembling panel. Check if the fragment is
         * the first fragment on Assembling panel or it can be appended to the 
         * others fragments */
        int fragmentNum;
        try{
            fragmentNum = Integer.parseInt(frg.getParameters()[Defs.FRAGMENT_NUM]);
        }catch(NumberFormatException nfe){
            view.printStringLn("Error in adding a fragment on Assembling "
                    + "panel: " + nfe);
            return;
        }
        /*
        if(fragmentNum < 2)
            assemblingPanel.executeCmd(Defs.COMMAND_LOAD_FROM_PDB);
        else
            assemblingPanel.executeCmd(Defs.COMMAND_LOAD_APPEND_FROM_PDB);
        
        /* Prepare the structure */
        prepareStructure(fragmentNum, Defs.ASSEMBLING);
    }//addFragment
    
    public void prepareFragments(){
        String text = "";
        String line;
        try {
 
          Scanner scanner = new Scanner(new File("fragment.pdb"));
          
          while (scanner.hasNextLine()) {
              line = scanner.nextLine();/*
              String[] parsedLine = line.split("\\s+");

              if ((parsedLine[2].equals("N")) && (Integer.parseInt(parsedLine[5]) == start)) 
                  line = scanner.nextLine();

              if ((parsedLine[2].equals("N")) && (Integer.parseInt(parsedLine[5]) == end)) {
                  text = text + line + "\n";
                  break;
              }*/
              text = text + line + "\n";
          }
          //text = text + scr.nextLine() + "\n";
          scanner.close();
          //text = new Scanner(new File(Defs.DUMMY_PDB_FILE)).useDelimiter("\n").next();
        } catch (IOException e) {
        }
    }

    
    /* Connect fragments on Assembling panel */
    public void connectFragments(Boolean print){
        int numFragments = model.getAllFragmentsA().size();
        String cmd;
        if(print)
            view.preparePanel(Defs.ASSEMBLING);
        Fragment currentFragment;
        Fragment nextFragment;
        /* Connect each fragment to its successor */
        for(int i = 0; i < numFragments; i++){
            currentFragment = model.getAllFragmentsA().get(i);
            nextFragment = model.getNextFragment(currentFragment);   //----- problem
            
            /* Draw a line between the two fragments */
            if(nextFragment != null){
                cmd = Utilities.drawConnectionString(currentFragment, nextFragment);
                assemblingPanel.executeCmd(cmd);
                
                if(print){
                    /* Print infos */
                    view.printStringLn(
                        Utilities.connectionInfoString(currentFragment, nextFragment)
                        );
                }
            }
        }
    }//connectFragments
    
    /* Deselect all fragments on Assembling panel */
    public void deselectAll(){
        colorAllFragments(Defs.COLOR_DESELECT_FRAGMENT , Defs.ASSEMBLING);
    }//deselectAll
    
    /* Open a panel and get the group block for the fragment */
    public String[] requestBlock(Fragment fragment){
        
        /* Open a new block panel with all the infos about other groups */
        BlockPanel blockPanel = new BlockPanel(fragment, model.getBlockGroups());
        
        /* Get the resulting group */
        int result = JOptionPane.showConfirmDialog(view, blockPanel, 
                "Set Block Group", JOptionPane.OK_CANCEL_OPTION, 
                JOptionPane.PLAIN_MESSAGE);
        
        /* Check the correctness of the input and the whole process */
        if(result == JOptionPane.OK_OPTION && blockPanel.constraintToSet()){
            return blockPanel.getBlockGroup();
        }else
            return null;
    }//requestBlock
    
    /* Color the constraints set on the input fragment */
    public void colorConstraints(Fragment frg){
        boolean constraints[] = frg.getConstraints();
        int numCon = constraints.length;
        
        /* Color all the constraints imposed on the fragment */
        for(int i = 0; i < numCon; i++)
            if(constraints[i])
                colorSpecificConstraint(frg, i);
    }//colorConstraints
    
    /* Color a specific constraint */
    public void colorSpecificConstraint(Fragment frg, int i){
        switch(i){
            case Defs.CONSTRAINT_BLOCK:
                colorConstraintBlock(frg);
                break;
        }
    }//colorSpecificConstraint
    
    /* Color the block constraint */
    private void colorConstraintBlock(Fragment frg){
        String fragmentGroup = frg.getGroup();
        ArrayList<Fragment> frgs = model.getAllFragmentsA();
        Fragment fragment;
        int size = frgs.size();
        
        /* Color all the fragments belonging to the frg's group */
        for(int i = 0; i < size; i++){
            fragment = frgs.get(i);
            if(fragment.getConstraints()[Defs.CONSTRAINT_BLOCK] &&
               fragment.getGroup().equals(fragmentGroup)){
                String color = "color " + Utilities.getGroupColor(fragmentGroup);
                colorFragment(fragment, color, Defs.ASSEMBLING);
            }      
        }
    }//colorConstraintBlock
    
}//UpdateView
