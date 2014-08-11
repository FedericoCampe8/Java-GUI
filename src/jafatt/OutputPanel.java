package jafatt;

import java.awt.BorderLayout;
import org.biojava.bio.structure.Structure;

public class OutputPanel extends MolPanel{
    
    private OutputOpPanel outOP;
    private UserFrame view;
    private String str;
    
    public OutputPanel (UserFrame view, String str){
        super(str);
        this.str = str;
        setup(view);
        buildPanel(false);
    }
    
    /* Setup layout */
    private void setup(UserFrame view){
        
        this.view = view;
        
        outOP = new OutputOpPanel(view);
        add(outOP, BorderLayout.SOUTH);
        expanded = false;
        
    }//setup
    
    public void updateRmsd(String rmsd){
        setLabel(str + "  (RMSD " + rmsd + " \u212B)");
        
    }
    
     public void setProtein(Structure structure){
        outOP.setProtein(structure);
    }//setProtein
    
     @Override
    public void switchView(String newView){
        
        /* Change the view */
        molViewPanel.executeCmd(newView);
        
    }//switchView
    
    public void loadOutput(Structure structure){
        view.loadOutput(structure);
    } 
    
    @Override
    public void viewPdb() {
        viewPdb(view, outOP.structureSequence, Defs.OUTPUT);
    }
    
    @Override
    public void deselectFragments(){
        //only for extractionPanel
    }
    
    @Override
    public void changeView(){
       changeViewProtein(view, outOP.structureSequence, Defs.OUTPUT);
    }
    
    @Override
    public void saveProtein(){
        saveEvent(view, outOP.structureSequence, Defs.OUTPUT);
    }
    
    @Override
    public void event(){
        view.expand(Defs.OUTPUT);
        expanded = expanded ? expand(true) : expand(false);
        ((ExtractionPanel) view.getPanel(Defs.EXTRACTION)).expand(true);
        ((ExtractionPanel) view.getPanel(Defs.EXTRACTION)).expanded = false;
        ((AssemblingPanel) view.getPanel(Defs.ASSEMBLING)).expand(true);
        ((AssemblingPanel) view.getPanel(Defs.ASSEMBLING)).expanded = false;
    }
    
}//OutputPanel
