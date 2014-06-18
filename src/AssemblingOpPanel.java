package jafatt;

import java.io.*;
import java.util.ArrayList;
import java.awt.BorderLayout;
import java.awt.GridLayout;
import java.awt.GridBagConstraints;
import java.awt.Dimension;
import java.awt.Color;
import java.awt.event.ActionEvent;
import java.util.List;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.ImageIcon;
import javax.swing.BorderFactory;
import org.biojava.bio.structure.Chain;
import org.biojava.bio.structure.Structure;
import org.biojava.bio.structure.AminoAcid;
import java.io.BufferedReader;


public class AssemblingOpPanel extends OpPanel{
    
    /* Buttons on the panel */
    private OpButton constraintsButton;
    private OpButton solveButton;
    private OpButton move;
    private OpButton rotate;
    private OpButton undoButton;
    private OpButton redoButton;
    private OpButton saveButton;
    private OpButton rulerButton;
    private OpButton expandButton;
    
    /* Items */
    private UserFrame view;
    private Structure structure;
    private ProteinModel model;
    private ArrayList<Fragment> fragments;
    private ArrayList<Fragment> frgSelected;
    private ArrayList<Integer> jm;
    private ArrayList<Double> sp;
    private ArrayList<Integer> uss;
    private ArrayList<Integer> dgeq;
    private ArrayList<Integer> dleq;
    private ArrayList<Double> uniList;
    private ArrayList<Double> ellList;
    private ArrayList<String> choice;
    private ConstraintPanel cp;
    private int[] values;
    private int firstPosition;
    private Chain chain;
    private boolean[] bool;
    private String structureSequence;
    private String proteinId;
    private boolean ruleOn;
    
    private String currentDir = System.getProperty("user.dir");
    private String separator = Utilities.getSeparator();
    private String imSuffix = separator + "images" + separator;
    private String moveOnString = currentDir + imSuffix + "moveON.png";
    private String moveOffString = currentDir + imSuffix + "moveOFF.png";
    private String rotateOnString = currentDir + imSuffix + "rotateON.png";
    private String rotateOffString = currentDir + imSuffix + "rotateOFF.png";
    private String undoString = currentDir + imSuffix + "undo.png";
    private String redoString = currentDir + imSuffix + "redo.png";
    private String saveString = currentDir + imSuffix + "save.png";
    private String rulerOnString = currentDir + imSuffix + "rulerON.png";
    private String rulerOffString = currentDir + imSuffix + "rulerOFF.png";
    private String expandImage = currentDir + imSuffix + "expandWhite.png";
    private String reduceImage = currentDir + imSuffix + "reduceWhite.png";
    
    ImageIcon iconExpand = new ImageIcon(expandImage);
    ImageIcon iconReduce = new ImageIcon(reduceImage);
    
    ImageIcon moveOn = new ImageIcon(moveOnString);
    ImageIcon moveOff = new ImageIcon(moveOffString);
    
    ImageIcon rotateOn = new ImageIcon(rotateOnString);
    ImageIcon rotateOff = new ImageIcon(rotateOffString);
    
    ImageIcon undo = new ImageIcon(undoString);
    ImageIcon redo = new ImageIcon(redoString);
    
    ImageIcon save = new ImageIcon(saveString);
    
    ImageIcon rulerOn = new ImageIcon(rulerOnString);
    ImageIcon rulerOff = new ImageIcon(rulerOffString);
    
    private boolean expanded = false;   
    
    double widthPanel = this.getWidth();
    double heightPanel = this.getHeight();  //540

    
    public AssemblingOpPanel(UserFrame view){
        super(true);
        initComponents(view);
    }
    
    /* Set the components of the panel */
    private void initComponents(UserFrame view){
        
        this.view = view;
        structureSequence = "";
        
        /*
    
        constraintsButton = new OpButton("Constraints", "Set the Constraints to the model") {
            @Override
            public void buttonEvent(ActionEvent evt){
                constraintEvent();
            }
        };
         
         */
        

        solveButton = new OpButton("Solve", "Run Fiasco!") {
            @Override
            public void buttonEvent(ActionEvent evt){
                //solveEvent();
                constraintEvent();
            }
        };
         
        
        move = new OpButton(moveOff, "Move Selected Fragment") {
            @Override
            public void buttonEvent(ActionEvent evt){
                moveEvent();
            }
        };
        move.setPreferredSize(new Dimension(30, 30));
        
        rotate = new OpButton(rotateOff, "Rotate Selected Fragment") {
            @Override
            public void buttonEvent(ActionEvent evt){
                rotateEvent();
            }
        };
        rotate.setPreferredSize(new Dimension(30, 30));
        
        undoButton = new OpButton(undo, "Undo Move") {
            @Override
            public void buttonEvent(ActionEvent evt){
                undoEvent();
            }
        };
        undoButton.setPreferredSize(new Dimension(30, 30));
        
        redoButton = new OpButton(redo, "Redo Move") {
            @Override
            public void buttonEvent(ActionEvent evt){
                redoEvent();
            }
        };
        redoButton.setPreferredSize(new Dimension(30, 30));
        
        saveButton = new OpButton(save, "Save Changes") {
            @Override
            public void buttonEvent(ActionEvent evt){
                saveCacheEvent();
            }
        };
        saveButton.setPreferredSize(new Dimension(30, 30));
        
        ruleOn = false;
        rulerButton = new OpButton(rulerOff, "View Distances") {
            @Override
            public void buttonEvent(ActionEvent evt){
                distanceEvent();
            }
        };
        rulerButton.setPreferredSize(new Dimension(30, 30));
        
        expandButton = new OpButton(iconExpand, "Show more options") {

            @Override
            public void buttonEvent(ActionEvent evt) {
                expandEvent();
            }
        };
        expandButton.setPreferredSize(new Dimension(140, 6));
        expandButton.setBackground(Color.BLACK);
        expandButton.setFocusPainted(false);
        expandButton.setBorderPainted(false);
        expandButton.setContentAreaFilled(false);
        
        //super.setLayout(new GridLayout(2,1));
        expand.add(expandButton, BorderLayout.CENTER);
         //Add buttons
        //main.add(constraintsButton);
        main.add(solveButton);
        tool.add(rotate);
        tool.add(move);
        tool.add(undoButton);
        tool.add(redoButton);
        tool.add(rulerButton);
        tool.setBorder(BorderFactory.createTitledBorder("Tool Box"));
        
        
    }//initComponents
    
    
    private void expandEvent(){
        if(expanded){
            expanded = false;
            expandButton.setIcon(iconExpand);
            reducePanel();
        }else{
            expanded = true;
            expandButton.setIcon(iconReduce);
            expandPanel();
        }
    }
    
    private void expandPanel(){
        c.fill = GridBagConstraints.VERTICAL;
        c.gridx = 0; c.gridy = 0;
        add(expand,c);
        c.gridy = 1;
        add(tool,c);
        c.gridy = 2;
        add(main,c);
        getRootPane().revalidate();
    }
    
    private void reducePanel(){
        remove(tool);
        c.fill = GridBagConstraints.VERTICAL;
        c.gridx = 0; c.gridy = 0;
        add(expand,c);
        c.gridy = 1;
        add(main,c);
        getRootPane().revalidate();
    }
    
    private void moveEvent(){        
        
        if(structureSequence.equals("")){
            view.printStringLn("Load a protein first!");
            return;
        }
        
        if(!view.fragmentLoaded()){
            view.printStringLn("Transfer fragments first!");
            return;
        }
        if(view.move){
            //view.upView.execute("set allowMoveAtoms TRUE; ", Defs.ASSEMBLING);
            view.move = false;
            move.setIcon(moveOff);
        }else{
            view.move = true;
            view.rotate = false;
            move.setIcon(moveOn);
            rotate.setIcon(rotateOff);
        }

    }
    
    private void rotateEvent(){
        
        if(structureSequence.equals("")){
            view.printStringLn("Load a protein first!");
            return;
        }
        
        if(!view.fragmentLoaded()){
            view.printStringLn("Transfer fragments first!");
            return;
        }
        
        if(view.rotate){
           //view.upView.execute("set allowMoveAtoms TRUE; ", Defs.ASSEMBLING);
           view.rotate = false;
           rotate.setIcon(rotateOff);
       }else{
           view.rotate = true;
           view.move = false;
           rotate.setIcon(rotateOn);
           move.setIcon(moveOff);
       }
    }
    
    private void undoEvent(){
        
        if(structureSequence.equals("")){
            view.printStringLn("Load a protein first!");
            return;
        }
        
        if(!view.fragmentLoaded()){
            view.printStringLn("Transfer fragments first!");
            return;
        }
        
        move(true);
               
    }
    
    private void redoEvent(){
        
        if(structureSequence.equals("")){
            view.printStringLn("Load a protein first!");
            return;
        }
        
        if(!view.fragmentLoaded()){
            view.printStringLn("Transfer fragments first!");
            return;
        }        
        move(false);        
    }
    
    public void move(Boolean isUndo){
        
        view.upView.execute(Utilities.deleteConnectionString(
                        model.getDimensionFragmentsA()),
                        Defs.ASSEMBLING);
        if(isUndo){
            view.upView.execute("undoMove 1; ", Defs.ASSEMBLING);
        }else{
            view.upView.execute("redoMove 1; ", Defs.ASSEMBLING);
        }
        
        view.upView.connectFragments(false);
        
    }
    
    private void saveCacheEvent(){
        
        if(structureSequence.equals("")){
            view.printStringLn("Load a protein first!");
            return;
        }
        
        if(!view.fragmentLoaded()){
            view.printStringLn("Transfer fragments first!");
            return;
        }
        //to do
        ((AssemblingPanel)view.getPanel(Defs.ASSEMBLING)).executeCmd("select * ;");
        ((AssemblingPanel)view.getPanel(Defs.ASSEMBLING)).executeCmd("write pdb \"" 
                + Defs.TEMP + "cache.pdb" + "\"; ");
        
        view.cacheToPdb();
                
    }
    
    private void distanceEvent(){

        if(structureSequence.equals("")){
            view.printStringLn("Load a protein first!");
            return;
        }
        
        if(!view.fragmentLoaded()){
            view.printStringLn("Transfer fragments first!");
            return;
        }

        /*
        String origin = ((AssemblingPanel) view.getPanel(Defs.ASSEMBLING)).molViewPanel.getAxesCoordinates();
        origin = origin.replaceAll("[()]", "");
        */
        
        ArrayList<Fragment> frgsSelected = (ArrayList<Fragment>)view.getModel().getAllSelectedFragmentsA().clone();
        int size = frgsSelected.size();
        
        if(ruleOn){
            
            if((frgsSelected.size() == size) ||
                    (frgsSelected.isEmpty()) ||
                    (size == 0)) {
                rulerButton.setIcon(rulerOff);
                ruleOn = false;
                deleteMeasures();
            }else{
                deleteMeasures();
                measure(frgsSelected);
                
            }
        }else{
            
            if(frgsSelected.size()< 2){
                view.printStringLn("Select two or more fragments!");
                return;
            }
            size = frgsSelected.size();
            measure(frgsSelected);                    
            rulerButton.setIcon(rulerOn);
            ruleOn = true;
        }
    }
    
    private void measure(ArrayList<Fragment> frgs){
        
        for (int i = 0; i < frgs.size(); i++) {
                String startFirstAA = frgs.get(i).getParameters()[Defs.FRAGMENT_START_ATOM];
                String endFirstAA = frgs.get(i).getParameters()[Defs.FRAGMENT_END_ATOM];
                frgs.remove(i);
                for (int j = 0; j < frgs.size(); j++) {
                    String startSecondAA = frgs.get(j).getParameters()[Defs.FRAGMENT_START_ATOM];
                    String endSecondAA = frgs.get(j).getParameters()[Defs.FRAGMENT_END_ATOM];
                    ((AssemblingPanel) view.getPanel(Defs.ASSEMBLING)).executeCmd(
                            "monitor (atomno=" + startFirstAA + ") (atomno=" + endSecondAA + "); " + 
                            "monitor (atomno=" + endFirstAA + ") (atomno=" + startSecondAA + "); ");
                }
            }     
        
    }
    
    private void deleteMeasures(){
        ((AssemblingPanel) view.getPanel(Defs.ASSEMBLING)).executeCmd(
                        "measure DELETE;");
    }
    
    private void saveEvent(){
        
        chain = structure.getChain(0);
        List groups = chain.getAtomGroups("amino");
        
        proteinId = model.idProteinCode;
        
        /* First AA */
        AminoAcid firstAA = (AminoAcid) groups.get(0);
        firstPosition = Utilities.getAAPosition(firstAA);
        
        
        String cms = "BASEDIR=" + Defs.SOLVER_PATH + "\n\n$BASEDIR/Fiasco/fiasco \\\n"
                     +  Defs.tab1 + "--input ../proteins/"+proteinId+".in.fiasco \\\n"
                     +  Defs.tab1 + "--outfile ../proteins/"+proteinId+".out.pdb \\\n";
            
        bool = cp.getFilled();
        values = cp.getValues();

        if ((bool[Defs.DOMAIN]) && (values[Defs.DOMAIN] != -1)) {
            cms = cms + Defs.tab1 + "--domain-size " + values[Defs.DOMAIN] + " \\\n";
        }

        if ((bool[Defs.SOLUTIONS]) && (values[Defs.SOLUTIONS] != -1)) {
            cms = cms + Defs.tab1 + "--ensembles " + values[Defs.SOLUTIONS] + " \\\n";
        }

        if ((bool[Defs.TIMEOUT_SEARCH]) && (values[Defs.TIMEOUT_SEARCH] != -1)) {
            cms = cms + Defs.tab1 + "--timeout-search " + values[Defs.TIMEOUT_SEARCH] + " \\\n";
        }

        if ((bool[Defs.TIMEOUT_TOTAL]) && (values[Defs.TIMEOUT_TOTAL] != -1)) {
            cms = cms + Defs.tab1 + "--timeout-total " + values[Defs.TIMEOUT_TOTAL] + " \\\n";
        }

        if (bool[Defs.DGEQ]) {
            dgeq = cp.getDGEQList();
            int j = 0;

            while (dgeq.size() > j) {
                cms = cms + Defs.tab1 + "--distance-geq "
                        + dgeq.get(j).intValue() + " "
                        + dgeq.get(++j).intValue() + " "
                        + dgeq.get(++j).intValue() + " \\\n";
                j++;
            }
        }

        if (bool[Defs.DLEQ]) {
            dleq = cp.getDLEQList();
            int j = 0;

            while (dleq.size() > j) {
                cms = cms + Defs.tab1 + "--distance-leq "
                        + dleq.get(j).intValue() + " "
                        + dleq.get(++j).intValue() + " "
                        + dleq.get(++j).intValue() + " \\\n";
                j++;
            }
        }

        if (cp.measures()) {
            String measurements = ((AssemblingPanel) view.getPanel(Defs.ASSEMBLING)).molViewPanel.getMeasurements();
            cms = cms + measurements;
        }


        if (bool[Defs.UNIFORM]) {
            int k = 0;
            uniList = cp.getUniformList();

            while (uniList.size() - 1 > k) {
                cms = cms + Defs.tab1 + "--uniform ";
                while (uniList.get(k) != -1.0) {
                    cms = cms + uniList.get(k++).intValue() + " ";
                }
                cms = cms + ": voxel-side= " + uniList.get(++k) + " \\\n";
                k++;
            }

        }

        if (bool[Defs.ELLIPSOID]) {
            int k = 0;
            ellList = cp.getEllList();

            while (ellList.size() > k) {
                cms = cms + Defs.tab1 + "--ellipsoid ";
                while (ellList.get(k) != -1.0) {
                    cms = cms + ellList.get(k++).intValue() + " ";
                }
                cms = cms + ": f1= " + ellList.get(++k) + " "
                        + ellList.get(++k) + " " + ellList.get(++k) + " "
                        + "f2= " + ellList.get(++k) + " "
                        + ellList.get(++k) + " " + ellList.get(++k) + " "
                        + "sum-radii= " + ellList.get(++k).intValue() + " \\\n";
                k++;
            }

        }

        if (bool[Defs.JM]) {
            jm = cp.getJmList();
            sp = cp.getSpList();
            choice = cp.getChoice();
            int j = 0;
            int k = 0;

            while (jm.size() > j) {
                cms = cms + Defs.tab1 + "--jm " + (jm.get(j).intValue() - firstPosition)
                        + " '" + choice.get(k) + "' " + (jm.get(++j).intValue() - firstPosition)
                        + "  : numof-clusters= " + jm.get(++j).intValue() + " " + jm.get(++j).intValue()
                        + " \\\n" + Defs.tab2 + " sim-params= " + sp.get(k).doubleValue() + " " + jm.get(++j).intValue() + " \\\n";
                j++;
                k++;
            }
        }

        if (bool[Defs.USS]) {

            uss = cp.getUSSList();
            int j = 0;
            while (uss.size() > j) {
                cms = cms + Defs.tab1 + "--unique-source-sinks " + uss.get(j).intValue() + " '->' " + uss.get(++j).intValue()
                        + "  : voxel-side= " + uss.get(++j).intValue() + " \\\n";
                j++;
            }
        }
        
        try{
            FileOutputStream filesh = new FileOutputStream(Defs.FIASCO_PATH + "solve.sh");
            filesh.write(cms.getBytes());
            filesh.flush();
            filesh.close();
            view.printStringLn("Script solve.sh saved");
        }catch (IOException e) {
            view.printStringLn("Error: " + e);
        }        
        
        String path = view.getController().getPath();
        String protein = path.substring(path.lastIndexOf("/") + 1).split("\\.")[0];
        if(!path.substring(path.lastIndexOf("/") + 1).split("\\.")[1].equals("pdb"))
            protein += "." + path.substring(path.lastIndexOf("/") + 1).split("\\.")[1];
        
        String str = "% Database information\n"
                /*
                + "FRAGMENTDB  ../fragment-assembly-db/FREAD/fread_loop_db.dat\n"
                + "ENERGYDB    ../../config/energy_table.csv\n"
                + "CLASSDB     ../../config/defclass.txt\n"
                + "TORSDBPATH  ../../pmf/\n"
                + "CORRDB      t1_t2_new.pmf\n"*/
                + "FRAGMENTDB  DB/loopdb.dat\n"
                + "COULOMBPAR  config/coulomb.csv\n"
                + "LJPARAMETER config/lenard_jones.csv\n"
                + "HDPARAMETER config/h_distances.csv\n"
                + "HAPARAMETER config/h_angles.csv\n"
                + "CONTACT     config/contact.csv\n"
                + "TORSPAR     config/table_corr.pot\n"
                + "% Proteins Information\n"
                + "TARGET_PROT ../proteins/"+protein+"\n"
                + "KNOWN_PROT  ../proteins/"+protein+"\n"
                + "CONSTRAINTS ../proteins/"+proteinId+".in.con\n"
                + "FRAG_SEC_FL ../proteins/"+proteinId+".in.pdb";
        
         try{
            FileOutputStream infiasco = new FileOutputStream(Defs.PROTEINS_PATH +proteinId+".in.fiasco");
            infiasco.write(str.getBytes());
            infiasco.flush();
            infiasco.close();
            view.printStringLn("File "+proteinId+".in.fiasco saved");
       }catch (IOException e) {
           view.printStringLn("Error: " + e);
       }

         String prot;
         try{
            prot = chain.getAtomSequence()+"\n";
        }catch(Exception e){
            view.printStringLn("Error in SelectFragments Panel: " + e);
            return;
        }
         
        model = view.getModel();
        fragments = model.getAllFragmentsA();
         
         String[] param;
         for(int i=0;i<fragments.size();i++){
             param = fragments.get(i).getParameters();     
             prot = prot + "Protein ID: "+proteinId+"\n"
                     +"Fragment n. : "+(i+1)+"\n"
                     //+param[Defs.FRAGMENT_OFFSET]
                     + "1"
                     +"\nOffset on target:\n"
                     +(Integer.parseInt(param[Defs.FRAGMENT_START_AA]) + 1 - firstPosition + 1)+"\n"
                     +(Integer.parseInt(param[Defs.FRAGMENT_END_AA]) -1 - firstPosition + 1)+"\n"
                     +"CONSTRAINTS:\n"+"NO CONSTRAINTS"+"\n\n";
         }
         try{
            FileOutputStream incon = new FileOutputStream(Defs.PROTEINS_PATH +proteinId+".in.con");
            incon.write(prot.getBytes());
            incon.flush();
            incon.close();
            view.printStringLn("File "+proteinId+".in.con saved");
         }catch (IOException e) {
           view.printStringLn("Error: " + e);
         }

         //if(cpBool)
         //    cp.cancel();
         
    }//saveEvent
        
    private void constraintEvent(){
        
        if(structureSequence.equals("")){
            view.printStringLn("Load a protein first!");
            return;
        }
        
        if(!view.fragmentLoaded()){
            view.printStringLn("Transfer fragments first!");
            return;
        }
        
        model = view.getModel();        
        fragments = model.getAllFragmentsA();


        /* Open a new Constraint panel with a new thread */
        cp = new ConstraintPanel(view, structure, fragments, ruleOn);
                
        /* Create the thread */
        Thread threadSelectFragments;
        threadSelectFragments = new Thread(cp);
        threadSelectFragments.start();
        
    }//constraintEvent
    
    public void solveEvent(){


        if(structureSequence.equals("")){
            view.printStringLn("Load a protein first!");
            return;
        }
        
         if(!view.fragmentLoaded()){
            view.printStringLn("Transfer fragments first!");
            return;
        }
                
        
        saveEvent();
        saveCacheEvent();
        
        //Change permission of the script to be executable
        try{
            Process permission = new ProcessBuilder("chmod", "+x", Defs.FIASCO_PATH + "solve.sh").start();
            permission.waitFor();
        }catch(Exception e){
            e.printStackTrace(System.out);
        }
        
        //Delete any out.pdb file existing
        try{
            FileOutputStream outpdb = new FileOutputStream(Defs.PROTEINS_PATH +proteinId+".out.pdb");
            outpdb.close();
         }catch (IOException e) {
           view.printStringLn("Error: " + e);
         }


        view.printStringLn("Solver running...");
        
        //Run a new thread that controls the solver
        FiascoSolver solver = new FiascoSolver(view);
        Thread fiasco = new Thread(solver);
        fiasco.start();
        
    }//selectEvent
    
    public void outputEvent(){
        
        if(structureSequence.equals("")){
            view.printStringLn("Load a protein first!");
            return;
        }
        //check whether the out.pdb file is empty. 
        //If so, the solver finished improperly (e.g. seg-fault).
        BufferedReader br;
        String protein = Defs.PROTEINS_PATH
                +proteinId+ ".out.pdb";
        try{
            br = new BufferedReader(new FileReader(protein));     
            if (br.readLine() == null) {
                view.printStringLn("Something went wrong with the solver.");
                view.printStringLn("Check Fragments and Constraints and try to call the solver again.");
                return;
            }
        }catch (IOException e){
            return;
        }
        //If the out.pdb file is not empty,
        //we're ready to see the output in the output panel.
        view.loadOutput(structure);
     
        
    }
    
    /* Set the current protein structure */
    public void setProtein(Structure structure){
        Chain chain = null;
        this.structure = structure;
        
        /* Set the sequence string */
        try{
            chain = structure.getChain(0);
            //structureSequence = chain.getSeqResSequence();
            structureSequence = HeaderPdb.getTargetSequence();
        }catch(Exception e){
            view.printStringLn("Unable to load the sequence of the protein");
        }
    }//setProtein
    
    private class ConfirmPanel extends JPanel{
    
        public ConfirmPanel() {
            setLayout(new GridLayout(1, 2));
            add(new JLabel("Do you really want to run Fiasco without any constraints?"));
        }
    }//ConfirmtPanel
    
}//AssemblingOpPanel 
