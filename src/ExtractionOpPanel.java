package jafatt;

import org.biojava.bio.structure.Chain;
import org.biojava.bio.structure.Structure;
import javax.swing.JOptionPane;
import javax.swing.JLabel;
import javax.swing.JPanel;
import java.awt.event.ActionEvent;
import java.awt.GridLayout;
import java.util.ArrayList;
import java.io.*;

public class ExtractionOpPanel extends OpPanel{
    
    /* Buttons on the panel */
    //private OpButton viewButton;
    private OpButton selectButton;
    private OpButton sendButton;
    //private OpButton subSequenceButton;
    
    /* Items */
    private UserFrame view;
    private Structure structure;
    private String structureSequence;
    
    public ExtractionOpPanel(UserFrame view){
        super(false);
        initComponents(view);
    }
    
    /* Set the components of the panel */
    private void initComponents(UserFrame view){
        
        /* Set the view component */
        this.view = view;
        structureSequence = "";
        
        /*
        viewButton = new OpButton("View", "Setting for the protein's view"){
            @Override
            public void buttonEvent(ActionEvent evt){
                viewEvent();
            }
        };
         * 
         */
        selectButton = new OpButton("Select", "Select fragments by offset"){
            @Override
            public void buttonEvent(ActionEvent evt){
                selectEvent();
            }
        };
        sendButton = new OpButton("Assemble", "Transfer fragments to Assembling panel"){
            @Override
            public void buttonEvent(ActionEvent evt){
                transferEvent();
            }
        };
        /*
        subSequenceButton = new OpButton("Subseq", "Get the common subsequence between protein and target"){
            @Override
            public void buttonEvent(ActionEvent evt){
                subSequenceEvent();
            }
        };
         * 
         */
        
        /* Add buttons */
        //main.add(viewButton);
        main.add(selectButton);
        //topA.add(subSequenceButton);
        main.add(sendButton);
    }//initComponents
    
    /* 
     * Load a known protein 
    private void viewEvent(){
        if(structureSequence.equals("")){
            view.printStringLn("Load a protein first!");
            return;
        }
        
        /* Open a new ViewOptions panel with a new thread 
        Thread threadViewOptions;
        ViewOptionsPanel vop = new ViewOptionsPanel(view);
        
        /* Create the thread
        threadViewOptions = new Thread(vop);
        threadViewOptions.start();
        
        /* Print infos
        view.printStringLn("Open ViewOptions panel");
    }//loadEvent
     * 
     */
    
    /* Select the fragments by their offset */
    private void selectEvent(){
        if(structureSequence.equals("")){
            view.printStringLn("Load a protein first!");
            return;
        }
        
        if(!view.getController().getPermissionExtraction())
            return; 
        
        /* Open a new SelectFragments panel with a new thread */
        SelectFragmentsPanel sfp = new SelectFragmentsPanel(view, structure);
        
        /* Create the thread */
        Thread threadSelectFragments;
        threadSelectFragments = new Thread(sfp);
        threadSelectFragments.start();
        
        /* Print infos */
        view.printStringLn("Open SelectFragments panel");
    }//selectEvent
    
    /* Set the current protein structure */
    public void setProtein(Structure structure){
        Chain chain = null;
        this.structure = structure;
        
        /* Set the sequence string */
        try{
            //chain = structure.getChain(0);
            structureSequence = HeaderPdb.getTargetSequence();
            //structureSequence = chain.getSeqResSequence();
        }catch(Exception e){
            view.printStringLn("Unable to load the sequence of the protein");
        }
    }//setProtein
    
    /* Transfer fragments to Assembling panel */
    private void transferEvent(){
        String[] offsets;
        
        if(structureSequence.equals("")){
            view.printStringLn("Load a protein first!");
            return;
        }
        
        if(view.upView.getAllFragmentsExtraction().isEmpty()){
            view.printStringLn("Select fragments first!");
            return;
        }
        
        if(!view.getController().getPermissionExtraction())
            return;
        
        //view.upView.execute("zap;", Defs.ASSEMBLING);
        
        /* Check if there is an "old" framgent again */
        //boolean ok = view.upView.checkRightTransfer();
        //if(!ok){
        
        if (!view.upView.getAllFragmentsAssembling().isEmpty()){
            
           ConfirmPanel confirmPanel = new ConfirmPanel();

            /*Open a confirm dialog */
           int result = JOptionPane.showConfirmDialog(view, confirmPanel, "Fiasco",
                   JOptionPane.OK_CANCEL_OPTION, JOptionPane.PLAIN_MESSAGE);

           if (result == JOptionPane.CANCEL_OPTION) {
               return;
           }else{

               view.clearAssemblingPanel();

               //((TargetPanel)view.getPanel(Defs.TARGET)).reset();
               //((TargetPanel)view.getPanel(Defs.TARGET)).setOffsetString("",0);
               

           }

        }
        //Delete any in.pdb file existing
        try{
            FileOutputStream inpdb = new FileOutputStream(Defs.PROTEINS_PATH +
                    view.getModel().idProteinCode+".in.pdb");
            inpdb.close();
         }catch (IOException e) {
           view.printStringLn("Error: " + e);
         }
        
        //OffsetPanel
        // Set the offset
        /*
        OffsetPanel offPanel = new OffsetPanel(
                view.upView.getAllFragmentsExtraction()
                );
        
        // Open a confirm dialog
        int result = JOptionPane.showConfirmDialog(view, offPanel, "Set offsets",
                JOptionPane.OK_CANCEL_OPTION, JOptionPane.PLAIN_MESSAGE);
        
        // Check the result
        if(result == JOptionPane.OK_OPTION){
            if(offPanel.checkData(view.upView.getTargetSequence()))
                offsets = offPanel.getOffsets();
            else{
                view.printStringLn("Set offsets properly");
                return;
            }
            
            // Set the offsets of the fragments
            view.getController().setOffsetsOnFragments(offsets);   
        }else{
            view.printStringLn("Please, set the offset before transfer the fragments "
                    + "on Assembling panel");
            return;
        }
        
        ArrayList<Fragment> fragments = view.upView.getAllFragmentsExtraction();
        int numFragments = fragments.size();
        Boolean correct = false;
        for(int i = 0; i< numFragments; i++){
            if(Integer.parseInt(fragments.get(i).getParameters()[Defs.FRAGMENT_START_AA]) == 
                    Integer.parseInt(fragments.get(i).getParameters()[Defs.FRAGMENT_END_AA])){
                for(int j = 0; j< numFragments; j++){
                    if((Integer.parseInt(fragments.get(i).getParameters()[Defs.FRAGMENT_END_AA]) == 
                    Integer.parseInt(fragments.get(j).getParameters()[Defs.FRAGMENT_START_AA]) - 1) ||
                            (Integer.parseInt(fragments.get(i).getParameters()[Defs.FRAGMENT_START_AA]) == 
                    Integer.parseInt(fragments.get(j).getParameters()[Defs.FRAGMENT_END_AA]) + 1)){
                        correct = true;
                    }
                }
            }
        }
        if(!correct){
            view.printStringLn("Seems");
            return;
        }*/
        
        ArrayList<Fragment> fragments = view.upView.getAllFragmentsExtraction();
        int numFragments = fragments.size();

        for(int i = 0; i< numFragments; i++){
            if(Integer.parseInt(fragments.get(i).getParameters()[Defs.FRAGMENT_START_AA]) == 
                    Integer.parseInt(fragments.get(i).getParameters()[Defs.FRAGMENT_END_AA])){
                view.printStringLn("Invalid Fragment Selection: ["
                        + fragments.get(i).getParameters()[Defs.FRAGMENT_START_AA] + ","
                        + fragments.get(i).getParameters()[Defs.FRAGMENT_END_AA] + "]"
                        + " Fragment size must be greater than one!");
                return;
            }
        }
        
        offsets = new String[numFragments];
        for(int i = 0; i < numFragments; i++){
            offsets[i] = fragments.get(i).getParameters()[Defs.FRAGMENT_START_AA];
        }
        //view.printStringLn(view.upView.getAllFragmentsExtraction().get(0).getParameters()[Defs.FRAGMENT_SEQUENCE_STRING] + "");
        view.getController().setOffsetsOnFragments(offsets);   
        /*
        /* Transfer the fragments on Assembling panel */
        //view.initTransfer();
        /* The real transfer */
        view.getController().transferFragments();
        
        /* Inform the complete transfer */
        view.fragmentsLoaded();
        
        ((AssemblingPanel)view.getPanel(Defs.ASSEMBLING)).executeCmd("select 0;");
        
        //view.preparePanel(Defs.ASSEMBLING);
        //view.clearExtractionPanel();
    }//sendEvent
    
    private class ConfirmPanel extends JPanel{
    
        public ConfirmPanel() {
            setLayout(new GridLayout(1, 2));
            add(new JLabel("Overwrite the previous fragments?"));
        }
    }//ConfirmtPanel
    
    /* Get the common subsequence between protein and target */
    private void subSequenceEvent(){
        if(structureSequence.equals("")){
            view.printStringLn("Load a protein first!");
            return;
        }
        
        if(!view.getController().getPermissionExtraction())
            return;
        
        /* Open a new panel with a new thread */
        CommonSubSequencePanel cssp = new CommonSubSequencePanel(view, 
                structure);
        
        /* Create the thread */
        Thread threadSubSeq;
        threadSubSeq = new Thread(cssp);
        threadSubSeq.start();
        
        /* Print infos */
        view.printStringLn("Open SubSequence panel");
        
    }//subSequenceEvent
    
}//ExtractionOpPanel
