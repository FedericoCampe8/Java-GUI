package jafatt;

import java.awt.Component;
import java.awt.event.ActionEvent;
import javax.swing.ImageIcon;
import javax.swing.JFileChooser;
import javax.swing.filechooser.FileNameExtensionFilter;
import javax.swing.JOptionPane;
import org.biojava.bio.gui.sequence.tracklayout.SimpleTrackLayout;

public class UserFrameActions{
   public boolean loadATarget;
   public boolean loadAProtein;
   private Controller ctr;
   private UserFrame view;
   
   public UserFrameActions(Controller ctr, UserFrame view){
       setup(ctr, view);
   }
   
   /* Setup */
   private void setup(Controller ctr, UserFrame view){
       loadATarget = false;
       loadAProtein = false;
       this.ctr = ctr;
       this.view = view;
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
               view.printStringLn("Please, enter a number");
               return;
           }
           
           /* Set a new layout for the target sequence */
           ((TargetPanel)view.getPanel(Defs.TARGET)).getTargetSequence().setTrackLayout(
                   new SimpleTrackLayout(
                           ( (TargetPanel)view.getPanel(Defs.TARGET)).getTargetSequence().getSequence(),
                           i )
                   );
           view.printStringLn("Target wrapped on " +  i);
       }else
           view.printStringLn("Load a target first");
   }//wrapTarget
   
   /* Load target protein */
   public void loadTargetEvent(){
       String targetPath = "";
       String[] options = new String[]{ "From FASTA", "From PDB" };
       JFileChooser choice;
       boolean fromPDB = true;
       
       /* Select option panel */
       
       String result = (String)JOptionPane.showInputDialog(
               view, "Select the source", "Load Target",
               JOptionPane.QUESTION_MESSAGE,
               new ImageIcon(new ImageIcon(view.frameIcon).getImage().getScaledInstance(70, 70,
                java.awt.Image.SCALE_SMOOTH)), options, "From FASTA" );
       try{
           fromPDB = result.equals("From PDB");
       }catch(Exception e){}
       
       /* Open the file chooser to select the file target */
       if(result != null){
           //choice = new JFileChooser();
           choice = new JFileChooser(Defs.PROTEINS_PATH);
           FileNameExtensionFilter fastaFilter = new FileNameExtensionFilter("Fasta Files", "fa", "fasta", "fasta.txt");
           FileNameExtensionFilter pdbFilter = new FileNameExtensionFilter("Pdb Files", "pdb");
           if(fromPDB)
               choice.setFileFilter(pdbFilter);
           else
               choice.setFileFilter(fastaFilter);
           try{
               int option = choice.showOpenDialog(view);
               if (option == JFileChooser.APPROVE_OPTION) {
                   targetPath = choice.getSelectedFile().getAbsolutePath();
               } else if (option == JFileChooser.CANCEL_OPTION) {
                   return;
               }
           }catch(Exception eopen){
               view.printString("Error on loading target from file: " + eopen);
           }
           
           /* Load the target */
           boolean ok;
           ok = ctr.loadStructure(targetPath, Defs.TARGET, fromPDB);
           if(ok){
               loadATarget = true;
               view.targetLoaded();
           }else{
               view.printStringLn("Error on loading the target");
           }
       }
   }//loadTargetEvent
   
   public void buildTargetEvent(){
       BuildFasta bf = new BuildFasta(view);
       Thread threadBuildFasta;
       threadBuildFasta = new Thread(bf);
       threadBuildFasta.start();
   }
   
   public void downloadProtein(){       
       DownloadProteinPanel dpp = new DownloadProteinPanel(view);
       Thread threadFasta;
       threadFasta = new Thread(dpp);
       threadFasta.start();
   }
   
   public void runCocos(){
        OptionsPanel opPanel = new OptionsPanel(view,view.getModel().getTargetPath(),false);
        Thread panel = new Thread(opPanel);
        panel.start();
    }
   
   public void runCocosGPU(){
        OptionsPanel opPanel = new OptionsPanel(view,view.getModel().getTargetPath(),true);
        Thread panel = new Thread(opPanel);
        panel.start();
   }

   /* Load protein where extract fragments */
   public void loadProteinEvent(){
       String proteinPath = "";
       if(loadATarget){
           
           /* Create a file chooser for the protein file */
           JFileChooser proteinFile = new JFileChooser(Defs.PROTEINS_PATH);
           try{
               proteinFile.showOpenDialog(view);
               proteinPath = proteinFile.getSelectedFile().getAbsolutePath();
           }catch(Exception e){
               return;
           }
           
           /* Load the structure */
           boolean ok;
           
           /* Init loading */
           view.initProteinLoaded(Defs.EXTRACTION);
           
           /* Real loading */
           ok = ctr.loadStructure(proteinPath, Defs.EXTRACTION, true);           
           if(ok){
               loadAProtein = true;
               view.proteinLoaded();
           }else{
               
               /* Stop the progress bar */
               view.proteinNotLoaded();
           }
       } else 
           view.printStringLn("Load a target first");
   }//loadProteinEvent
   
}//UserFrameActions
