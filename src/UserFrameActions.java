package jafatt;

import java.awt.Component;
import java.awt.event.ActionEvent;
import javax.swing.JFileChooser;
import javax.swing.JOptionPane;
import org.biojava.bio.gui.sequence.tracklayout.SimpleTrackLayout;

public class UserFrameActions{
   private boolean loadATarget;
   private boolean loadAProtein;
   private Controller ctr;
   private UserFrame uf;
   
   public UserFrameActions(Controller ctr, UserFrame uf){
       setup(ctr, uf);
   }
   
   /* Setup */
   private void setup(Controller ctr, UserFrame uf){
       loadATarget = false;
       loadAProtein = false;
       this.ctr = ctr;
       this.uf = uf;
   }//setup
    
   /* Measure fragments */
   public void measureEvent(){
       
   }//measureEvent
   
   /* Center on a fragment */
   public void centerEvent(){
       
   }//measureEvent
   
   /* Reset the view */
   public void resetEvent(){
       
   }//resetEvent
   
   /* Show all the constraits */
   public void showEvent(){
       
   }//showEven
   
   /* Wrap the target sequence on an input number */
   public void wrapTarget(ActionEvent evt){
       if(loadATarget){
           int i = 10000;
           /* Open a new input dialog */
           String result = JOptionPane.showInputDialog((Component)evt.getSource(), 
                   "Enter a single value to wrap");
           
           /* Wrap with target panel method */
           try{
               i = Integer.parseInt(result);
           }catch(NumberFormatException nfe){
               uf.printStringLn("Please, enter a number");
               return;
           }
           
           /* Set a new layout for the target sequence */
           ((TargetPanel)uf.getPanel(Defs.TARGET)).getTargetSequence().setTrackLayout(
                   new SimpleTrackLayout(
                           ( (TargetPanel)uf.getPanel(Defs.TARGET)).getTargetSequence().getSequence(),
                           i )
                   );
           uf.printStringLn("Target wrapped on " +  i);
       }else
           uf.printStringLn("Load a target first");
   }//wrapTarget
   
   /* Load target protein */
   public void loadTargetEvent(){
       String targetPath = "";
       String[] options = new String[]{ "From FASTA", "From PDB" };
       JFileChooser choice;
       boolean fromPDB = true;
       
       /* Select option panel */
       String result = (String)JOptionPane.showInputDialog(
               uf, "Select the source", "Load Target",
               JOptionPane.QUESTION_MESSAGE, null, options, "From FASTA" );
       try{
           fromPDB = result.equals("From PDB");
       }catch(Exception e){}
       
       /* Open the file chooser to select the file target */
       if(result != null){
           //choice = new JFileChooser();
           choice = new JFileChooser(Defs.path_prot);
           try{
               int option = choice.showOpenDialog(uf);
               targetPath = choice.getSelectedFile().getAbsolutePath();
           }catch(Exception eopen){
               uf.printString("Error on loading target from file: " + eopen);
           }
           
           /* Load the target */
           boolean ok;
           ok = ctr.loadStructure(targetPath, Defs.TARGET, fromPDB);
           if(ok){
               loadATarget = true;
               uf.targetLoaded();
           }else{
               uf.printStringLn("Error on loading the target");
           }
       }
   }//loadTargetEvent
   
   /* Load protein where extract fragments */
   public void loadProteinEvent(){
       String proteinPath = "";
       if(loadATarget){
           
           /* Create a file chooser for the protein file */
           JFileChooser proteinFile = new JFileChooser(Defs.path_prot);
           try{
               proteinFile.showOpenDialog(uf);
               proteinPath = proteinFile.getSelectedFile().getAbsolutePath();
           }catch(Exception e){
               return;
           }
           
           /* Load the structure */
           boolean ok;
           
           /* Init loading */
           uf.initProtinLoaded();
           
           //HeaderPdb.info(proteinPath);
           
           /* Real loading */
           ok = ctr.loadStructure(proteinPath, Defs.EXTRACTION, true);           
           if(ok){
               loadAProtein = true;
               uf.proteinLoaded();
           }else{
               
               /* Stop the progress bar */
               uf.barPanel.stop();
               uf.printStringLn("Error on loading the protein");
           }
       } else 
           uf.printStringLn("Load a target first");
   }//loadProteinEvent
   
}//UserFrameActions
