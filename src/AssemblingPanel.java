package jafatt;

import java.awt.BorderLayout;
import org.biojava.bio.structure.Structure;

public class AssemblingPanel extends MolPanel{
    
    private AssemblingOpPanel asOP;
    private UserFrame view;
    boolean expanded;
    
    public AssemblingPanel(UserFrame view, String str){
        super(str);
        setup(view);
    }
    
    /* Setup layout */
    private void setup(UserFrame view){
        this.view = view;
        asOP = new AssemblingOpPanel(view);
        add(asOP, BorderLayout.SOUTH);
        expanded = false;
    }//setup
    
    /* Set the current protein structure */
    public void setProtein(Structure structure){
        asOP.setProtein(structure);

    }//setProtein
    
    public void loadOutput(){
        asOP.outputEvent();

    }//setProtein
    
    @Override
    public void event(){
        view.expand(Defs.ASSEMBLING);
        expanded = expanded ? expand(true) : expand(false);
        ((ExtractionPanel) view.getPanel(Defs.EXTRACTION)).expand(true);
        ((ExtractionPanel) view.getPanel(Defs.EXTRACTION)).expanded = false;
        ((OutputPanel) view.getPanel(Defs.OUTPUT)).expand(true);
        ((OutputPanel) view.getPanel(Defs.OUTPUT)).expanded = false;
        
    }
    
}//AssemblingPanel
