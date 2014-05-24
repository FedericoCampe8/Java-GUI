package jafatt;

import java.io.*;
import java.util.ArrayList;
import java.awt.GridLayout;
import java.awt.Dimension;
import java.awt.event.ActionEvent;
import java.util.List;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JOptionPane;
import javax.swing.ImageIcon;
//import org.jmol.api.JmolViewer;
//import org.jmol.api.JmolAdapter;
import org.biojava.bio.structure.Chain;
import org.biojava.bio.structure.Structure;
import org.biojava.bio.structure.AminoAcid;
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.awt.event.WindowListener;

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
    private boolean cpBool, solverBool;
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
    
    ImageIcon moveOn = new ImageIcon(moveOnString);
    ImageIcon moveOff = new ImageIcon(moveOffString);
    
    ImageIcon rotateOn = new ImageIcon(rotateOnString);
    ImageIcon rotateOff = new ImageIcon(rotateOffString);
    
    ImageIcon undo = new ImageIcon(undoString);
    ImageIcon redo = new ImageIcon(redoString);
    
    ImageIcon save = new ImageIcon(saveString);
    
    ImageIcon rulerOn = new ImageIcon(rulerOnString);
    ImageIcon rulerOff = new ImageIcon(rulerOffString);
    
    
    public AssemblingOpPanel(UserFrame view){
        initComponents(view);
    }
    
    /* Set the components of the panel */
    private void initComponents(UserFrame view){
        
        this.view = view;
        structureSequence = "";
        
        constraintsButton = new OpButton("Constraints", "Set the Constraints to the model") {
            @Override
            public void buttonEvent(ActionEvent evt){
                constraintEvent();
            }
        };

        solveButton = new OpButton("Solve", "Run Fiasco!") {
            @Override
            public void buttonEvent(ActionEvent evt){
                solveEvent();
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
        
         //Add buttons
        topA.add(constraintsButton);
        topA.add(solveButton);
        topB.add(rotate);
        topB.add(move);
        topB.add(undoButton);
        topB.add(redoButton);
        //topB.add(saveButton);
        topB.add(rulerButton);
        
    }//initComponents
       
    
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
        view.move(true);
               
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
        view.move(false);        
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
        //((AssemblingPanel)view.getPanel(Defs.ASSEMBLING)).executeCmd("write pdb \""  + Defs.path_prot+model.idProteinCode+ ".in.pdb" + "\"; ");
        ((AssemblingPanel)view.getPanel(Defs.ASSEMBLING)).executeCmd("write pdb \"" + "cache.pdb" + "\"; ");
        
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
        
        ((AssemblingPanel) view.getPanel(Defs.ASSEMBLING)).executeCmd(
                    "set showMeasurements TRUE; measure ON;");
        
        if(ruleOn){
            rulerButton.setIcon(rulerOff);
            ruleOn = false;
            ((AssemblingPanel) view.getPanel(Defs.ASSEMBLING)).executeCmd(
                    "measure DELETE;");
        }else{
            ArrayList<Fragment> frgsSelected = view.getModel().getAllSelectedFragmentsA();
            if(frgsSelected.size()< 2){
                //view.printStringLn("Select two or more fragments first!");
                return;
            }
            for (int i = 0; i < frgsSelected.size(); i++) {
                String startFirstAA = frgsSelected.get(i).getParameters()[Defs.FRAGMENT_START_AA];
                String endFirstAA = frgsSelected.get(i).getParameters()[Defs.FRAGMENT_END_AA];
                for (int j = 0; j < frgsSelected.size(); j++) {
                    if (j != i) {
                        String startSecondAA = frgsSelected.get(j).getParameters()[Defs.FRAGMENT_START_AA];
                        String endSecondAA = frgsSelected.get(j).getParameters()[Defs.FRAGMENT_END_AA];
                        ((AssemblingPanel) view.getPanel(Defs.ASSEMBLING)).executeCmd(
                                "measure (resno=" + startFirstAA + ") (resno=" + endSecondAA + "); ");
                        ((AssemblingPanel) view.getPanel(Defs.ASSEMBLING)).executeCmd(
                                "measure (resno=" + endFirstAA + ") (resno=" + startSecondAA + "); ");
                        System.out.println("measure (" + endFirstAA + ") (" + startSecondAA + "); ");

                    }
                }
            }
            rulerButton.setIcon(rulerOn);
            ruleOn = true;
        }
    }
    
    private void saveEvent(){
        
        chain = structure.getChain(0);
        List groups = chain.getAtomGroups("amino");
        
        proteinId = model.idProteinCode;
        
        /* First AA */
        AminoAcid firstAA = (AminoAcid) groups.get(0);
        firstPosition = Utilities.getAAPosition(firstAA);
        
        
        String cms = "BASEDIR=" + Defs.path_scrpt + "\n\n$BASEDIR/fiasco \\\n"
                     +  Defs.tab1 + "--input proteins/"+proteinId+".in.fiasco \\\n"
                     +  Defs.tab1 + "--outfile proteins/"+proteinId+".out.pdb \\\n";
                
        if (cpBool){
            
            bool = cp.getFilled();
            values = cp.getValues();
        
            if ((bool[Defs.Domain_Size]) && (values[Defs.Domain_Size] != -1)) {
                
                cms = cms + Defs.tab1 + "--domain-size " + values[Defs.Domain_Size] + " \\\n";
            }

            if ((bool[Defs.Maximum_Solutions]) && (values[Defs.Maximum_Solutions] != -1)) {
                
                cms = cms + Defs.tab1 + "--ensembles " + values[Defs.Maximum_Solutions] + " \\\n";
            }

            if ((bool[Defs.Timeout_Search]) && (values[Defs.Timeout_Search] != -1)) {
                
                cms = cms + Defs.tab1 + "--timeout-search " + values[Defs.Timeout_Search]  + " \\\n";
            }

            if ((bool[Defs.Timeout_Total]) && (values[Defs.Timeout_Total] != -1)) {
                
                cms = cms + Defs.tab1 + "--timeout-total " + values[Defs.Timeout_Total] + " \\\n";
            }
            
            if(bool[Defs.DGEQ]){
                dgeq = cp.getDGEQList();
                int j = 0;            
                
                while (dgeq.size() > j){
                    cms = cms + Defs.tab1 + "--distance-geq " 
                            + dgeq.get(j).intValue() +" "
                            + dgeq.get(++j).intValue() + " "
                            + dgeq.get(++j).intValue() + " \\\n";
                    j++;
                }
            }
            
            if(bool[Defs.DLEQ]){
                dleq = cp.getDLEQList();
                int j = 0;            
                
                while (dleq.size() > j){
                    cms = cms + Defs.tab1 + "--distance-leq " 
                            + dleq.get(j).intValue() +" "
                            + dleq.get(++j).intValue() + " "
                            + dleq.get(++j).intValue() + " \\\n";
                    j++;
                }
            }
            
            if (bool[Defs.Uniform]){
                
                int k = 0;
                uniList = cp.getUniformList();
                
                while (uniList.size()-1 > k){
                    cms = cms + Defs.tab1 + "--uniform ";
                    while (uniList.get(k) != -1.0) {
                        cms = cms + uniList.get(k++).intValue() + " ";
                    }
                    cms = cms + ": voxel-side= " + uniList.get(++k) + " \\\n";
                    k++;
                }
                
            }

            if (bool[Defs.Ellipsoid]){
                
                int k = 0;
                ellList = cp.getEllList();
                
                while (ellList.size() > k){
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
            
                while (jm.size() > j){
                    cms = cms + Defs.tab1 + "--jm " + (jm.get(j).intValue() - firstPosition + 1)  +
                            " '" + choice.get(k) +"' " + (jm.get(++j).intValue() - firstPosition + 1) + 
                            "  : numof-clusters= " + jm.get(++j).intValue() + " " + jm.get(++j).intValue() +
                            " \\\n" + Defs.tab2 + " sim-params= " + sp.get(k).doubleValue() + " " + jm.get(++j).intValue() + " \\\n";
                    j++;
                    k++;
                }
            }
        
            if (bool[Defs.USS]){
            
                uss = cp.getUSSList();        
                int j = 0;            
                while (uss.size() > j){
                    cms = cms + Defs.tab1 + "--unique-source-sinks " + uss.get(j).intValue() + " '->' " + uss.get(++j).intValue() + 
                            "  : voxel-side= " + uss.get(++j).intValue() + " \\\n";
                    j++;
                }
            }
        }
        
        try{
            FileOutputStream filesh = new FileOutputStream(Defs.path_script + "solve.sh");
            filesh.write(cms.getBytes());
            filesh.flush();
            filesh.close();
            view.printStringLn("Script solve.sh saved");
        }catch (IOException e) {
            view.printStringLn("Error: " + e);
        }        
        
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
                + "TARGET_PROT proteins/"+proteinId+"\n"
                + "KNOWN_PROT  proteins/"+proteinId+"\n"
                + "CONSTRAINTS proteins/"+proteinId+".in.con\n"
                + "FRAG_SEC_FL proteins/"+proteinId+".in.pdb";
        
         try{
            FileOutputStream infiasco = new FileOutputStream(Defs.path_prot +proteinId+".in.fiasco");
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
         
        if(!cpBool){
           model = view.getModel();
           fragments = model.getAllFragmentsA();
        }
         
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
            FileOutputStream incon = new FileOutputStream(Defs.path_prot +proteinId+".in.con");
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

        cpBool = true;
        
        model = view.getModel();        
        fragments = model.getAllFragmentsA();

        /* Open a new Constraint panel with a new thread */
        cp = new ConstraintPanel(view, structure, fragments);
                
        /* Create the thread */
        Thread threadSelectFragments;
        threadSelectFragments = new Thread(cp);
        threadSelectFragments.start();
        
        /* Print infos */
        //view.printStringLn("Output loaded on Output Panel");
    }//constraintEvent
    
    private void solveEvent(){


        if(structureSequence.equals("")){
            view.printStringLn("Load a protein first!");
            return;
        }
        
         if(!view.fragmentLoaded()){
            view.printStringLn("Transfer fragments first!");
            return;
        }
        
        if (!cpBool){
        
            ConfirmPanel confirmPanel = new ConfirmPanel();

            //Open a confirm dialog
            int result = JOptionPane.showConfirmDialog(view, confirmPanel, "Fiasco",
                    JOptionPane.OK_CANCEL_OPTION, JOptionPane.PLAIN_MESSAGE);

             if (result == JOptionPane.CANCEL_OPTION) {
                return;
            }
        }        
        
        saveEvent();
        saveCacheEvent();
        
        //Change permission of the script to be executable
        try{
            Process permission = new ProcessBuilder("chmod", "+x", Defs.path_script + "solve.sh").start();
            permission.waitFor();
        }catch(Exception e){
            e.printStackTrace(System.out);
        }
        
        //Delete any out.pdb file existing
        try{
            FileOutputStream outpdb = new FileOutputStream(Defs.path_prot +proteinId+".out.pdb");
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
        String protein = Defs.path_prot
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
