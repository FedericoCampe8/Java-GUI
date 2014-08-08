package jafatt;

import java.awt.event.ActionEvent;
import org.biojava.bio.structure.Chain;
import org.biojava.bio.structure.Structure;

public class OutputOpPanel extends OpPanel {

    /* Buttons on the panel */
    private OpButton sendButton;
    /* Items */
    private UserFrame view;
    private Structure structure;
    public String structureSequence;

    public OutputOpPanel(UserFrame view) {
        super(false);
        initComponents(view);
    }

    /* Set the components of the panel */
    private void initComponents(UserFrame view) {

        /* Set the view component */
        this.view = view;
        structureSequence = "";

        sendButton = new OpButton("Extract", "Transfer fragments to Extraction panel") {

            @Override
            public void buttonEvent(ActionEvent evt) {
                transferEvent();
            }
        };
        main.add(sendButton);
    }

    public void transferEvent() {

        if (structureSequence.equals("")) {
            view.printStringLn("Load a protein first!");
            return;
        }

        int modelNumber = ((OutputPanel) view.getPanel(Defs.OUTPUT)).molViewPanel.getModelNumber();
        int modelDisplayed = ((OutputPanel) view.getPanel(Defs.OUTPUT)).molViewPanel.getDisplayedModel();

        if (modelDisplayed == 0) {
            ((OutputPanel) view.getPanel(Defs.OUTPUT)).executeCmd("model 0; ");
            modelDisplayed = 1;
        }

        if (modelNumber > 1) {
            SelectModelPanel smp = new SelectModelPanel(view, modelNumber, modelDisplayed);

            /* Create the thread */
            Thread threadSubSeq;
            threadSubSeq = new Thread(smp);
            threadSubSeq.start();
            return;
        }

        boolean ok;
        String proteinPath = Defs.PROTEINS_PATH + view.getModel().idProteinCode + ".out.pdb";
        ok = view.getController().loadStructure(proteinPath, Defs.EXTRACTION, true);

    }
    /*
    public void viewPdb(){
        
        if(structureSequence.equals("")){
            view.printStringLn("Load a protein first!");
            return;
        }
        
        if(!view.fragmentLoaded()){
            view.printStringLn("Transfer fragments first!");
            return;
        }
        
        String protein = Defs.PROTEINS_PATH + view.getModel().idProteinCode + ".out.pdb";
        ViewPdbPanel vpp = new ViewPdbPanel(view,protein);
        Thread vp = new Thread(vpp);
        vp.start();
    }
     
    public void changeViewProtein() {
   
        if(structureSequence.equals("")){
            view.printStringLn("Load a protein first!");
            return;
        }
        
        if(!view.fragmentLoaded()){
            view.printStringLn("Transfer fragments first!");
            return;
        }
        
        Thread threadViewOptions;
        ViewOptionsPanel vop = new ViewOptionsPanel(view, Defs.OUTPUT);

        // Create the thread
        threadViewOptions = new Thread(vop);
        threadViewOptions.start();
    }
     */

    /* Set the current protein structure */
    public void setProtein(Structure structure) {
        Chain chain = null;
        this.structure = structure;

        /* Set the sequence string */
        try {
            //chain = structure.getChain(0);
            structureSequence = HeaderPdb.getTargetSequence();
            //structureSequence = chain.getSeqResSequence();
        } catch (Exception e) {
            view.printStringLn("Unable to load the sequence of the protein");
        }
    }//setProtein
}
