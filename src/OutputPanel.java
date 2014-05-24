package jafatt;

import java.awt.BorderLayout;
import org.biojava.bio.structure.Structure;

public class OutputPanel extends MolPanel{
    
    private OutputOpPanel outOP;
    private UserFrame view;
    boolean expanded;
    
    public OutputPanel (UserFrame view, String str){
        super(str);
        setup(view);        
    }
    
    /* Setup layout */
    private void setup(UserFrame view){
        
        this.view = view;
        
        outOP = new OutputOpPanel(view);
        add(outOP, BorderLayout.SOUTH);
        expanded = false;
        
    }//setup
    
     public void setProtein(Structure structure){
        outOP.setProtein(structure);
    }//setProtein
    
    public void loadOutput(Structure structure){
        view.loadOutput(structure);
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
