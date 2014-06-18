package jafatt;

import java.awt.BorderLayout;
import org.biojava.bio.structure.Structure;

public class ExtractionPanel extends MolPanel{
    
    private ExtractionOpPanel exOP;
    private UserFrame view;
    
    public ExtractionPanel(UserFrame view, String str){
        super(str);
        setup(view);
    }
    
    /* Setup layout */
    private void setup(UserFrame view){
        this.view = view;
        exOP = new ExtractionOpPanel(view);
        add(exOP, BorderLayout.SOUTH);
        expanded = false;
    }//setup
    
    /* Set the current protein structure */
    public void setProtein(Structure structure){
        exOP.setProtein(structure);
    }//setProtein
    
    /* Change the view of the protein */
    @Override
    public void switchView(String newView){
        
        /* Change the view */
        molViewPanel.executeCmd(newView);
        
        /* Color the selected fragments */
        view.upView.colorAllFragments(Defs.COLOR_ADD_FRAGMENT, Defs.EXTRACTION);
    }//switchView
    
    
    @Override
    public void event(){
        view.expand(Defs.EXTRACTION);
        expanded = expanded ? expand(true) : expand(false);
        ((AssemblingPanel) view.getPanel(Defs.ASSEMBLING)).expand(true);
        ((AssemblingPanel) view.getPanel(Defs.ASSEMBLING)).expanded = false;
        ((OutputPanel) view.getPanel(Defs.OUTPUT)).expand(true);
        ((OutputPanel) view.getPanel(Defs.OUTPUT)).expanded = false;
    }
    
}//ExtractionPanel
