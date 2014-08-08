package jafatt;

import java.awt.BorderLayout;
import java.awt.FlowLayout;
import java.awt.Color;
import java.awt.event.ActionEvent;
import java.awt.Dimension;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JFileChooser;
import javax.swing.filechooser.FileNameExtensionFilter;
import javax.swing.ImageIcon;
import javax.swing.JOptionPane;
import java.io.File;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.FileNotFoundException;
import java.util.Scanner;

public abstract class MolPanel extends JPanel{
    
    protected StructureViewPanel molViewPanel;
    protected Controller ctr;
    
    private JPanel jp,buttons;
    private JLabel panelName;
    private JLabel rmsd;
    private String cmdSetup = Defs.MOLMOUSESETUP;
    private OpButton expandButton,binButton,pdbButton, viewButton, saveButton;
    
    private String currentDir = System.getProperty("user.dir");
    private String separator = Utilities.getSeparator();
    private String imSuffix = separator + "images" + separator;
    private String expandString = currentDir + imSuffix + "expandPanel.png";
    private String reduceString = currentDir + imSuffix + "reducePanel.png";
    private String pdbString = currentDir + imSuffix + "pdb.png";
    private String binString = currentDir + imSuffix + "bin.png";
    private String viewString = currentDir + imSuffix + "view.png";
    private String saveString = currentDir + imSuffix + "saveit.png";
    
    boolean expanded;
    
    ImageIcon expandOn = new ImageIcon(expandString);
    ImageIcon reduceOn = new ImageIcon(reduceString);
    ImageIcon bin = new ImageIcon(binString);
    ImageIcon pdb = new ImageIcon(pdbString);
    ImageIcon view = new ImageIcon(viewString);
    ImageIcon save = new ImageIcon(saveString);
    
    public MolPanel(String str){
        setup(str);
    }
    
    /* Set the layout */
    private void setup(String str){
        try{
            molViewPanel = new StructureViewPanel();
            molViewPanel.executeCmd(cmdSetup);
        }catch (Exception e){
            System.out.println("Unable to load Jmol Panel");
            if(ctr != null){
                ctr.printStringLn("Unable to load Jmol Panel");
            }
        }
        
        /* Set the appearance */
        jp = new JPanel();
        jp.setBorder(new javax.swing.border.SoftBevelBorder(javax.swing.border.BevelBorder.RAISED));
        buttons = new JPanel(new FlowLayout());
        
        /* Alternative */
        /* jp.setBackground(UserFrame.INNERCOLOR); */
        jp.setBackground(Color.BLACK);
        buttons.setBackground(Color.BLACK);
        
        expandButton = new OpButton(expandOn, "Expand") {
            @Override
            public void buttonEvent(ActionEvent evt){
                event();
            }
        };
        expandButton.setPreferredSize(new Dimension(23, 23));
        expandButton.setBackground(Color.black);
        expandButton.setFocusPainted(false);
        expandButton.setBorderPainted(false);
        expandButton.setContentAreaFilled(false);
        
        
        binButton = new OpButton(bin, "Deselect Fragments") {
            @Override
            public void buttonEvent(ActionEvent evt){
                deselectFragments();
            }
        };
        binButton.setPreferredSize(new Dimension(23, 23));
        binButton.setBackground(Color.black);
        binButton.setFocusPainted(false);
        binButton.setBorderPainted(false);
        binButton.setContentAreaFilled(false);
        
        pdbButton = new OpButton(pdb, "View pdb") {
            @Override
            public void buttonEvent(ActionEvent evt){
                viewPdb();
            }
        };
        pdbButton.setPreferredSize(new Dimension(23, 23));
        pdbButton.setBackground(Color.black);
        pdbButton.setFocusPainted(false);
        pdbButton.setBorderPainted(false);
        pdbButton.setContentAreaFilled(false);
        
        
        saveButton = new OpButton(save, "Change View") {
            @Override
            public void buttonEvent(ActionEvent evt){
                saveProtein();
            }
        };
        
        saveButton.setPreferredSize(new Dimension(23, 23));
        saveButton.setBackground(Color.black);
        saveButton.setFocusPainted(false);
        saveButton.setBorderPainted(false);
        saveButton.setContentAreaFilled(false);
        
        
        viewButton = new OpButton(view, "Change View") {
            @Override
            public void buttonEvent(ActionEvent evt){
                changeView();
            }
        };
        
        viewButton.setPreferredSize(new Dimension(23, 23));
        viewButton.setBackground(Color.black);
        viewButton.setFocusPainted(false);
        viewButton.setBorderPainted(false);
        viewButton.setContentAreaFilled(false);

        
        /* Set the layout */
        setLayout(new BorderLayout());
        //jp.setLayout(new FlowLayout());
        jp.setLayout(new BorderLayout());
        panelName = new JLabel(str);
        panelName.setForeground(Color.green);
        rmsd = new JLabel("RMSD: ");
        rmsd.setForeground(Color.green);
        //jp.add(rmsd, BorderLayout.WEST);
        //jp.add(panelName, BorderLayout.CENTER);
        //jp.add(expandButton, BorderLayout.EAST);
        add(jp, BorderLayout.NORTH);
        add(molViewPanel, BorderLayout.CENTER);
    }//setup
    
    public void setLabel(String str){
        panelName.setText(str);
        panelName.setForeground(Color.green);
        validate();
    }
    
    public void buildPanel(boolean bin){
        jp.add(panelName, BorderLayout.CENTER);
        buttons.add(pdbButton);
        buttons.add(saveButton);
        //buttons.add(viewButton);
        if(bin)
            buttons.add(binButton);
        buttons.add(expandButton);
        jp.add(buttons, BorderLayout.EAST);
        validate();
        
    }
    
    public boolean expand(boolean reduced){
        //this.expanded = expanded;
        if(reduced){
            expandButton.setIcon(expandOn);
            expandButton.setToolTipText("Expand");
            reduced = false;
        }else{
            expandButton.setIcon(reduceOn);
            expandButton.setToolTipText("Reduce");
            reduced = true;
        }
        return reduced;
    }
    
    public void saveEvent(UserFrame view, String structureSequence, int panel){

        if(structureSequence.equals("")){
            view.printStringLn("Load a protein first!");
            return;
        }
        if(panel == Defs.ASSEMBLING){
            if (!view.fragmentLoaded()) {
                view.printStringLn("Transfer fragments first!");
                return;
            }
        }
        
        if(panel == Defs.OUTPUT){
            if (!view.isOutputLoaded()) {
                view.printStringLn("Run Fiasco first!");
                return;
            }
        }
        
        String file = view.getModel().idProteinCode;
        
        JFileChooser chooser = new JFileChooser(System.getProperty("user.dir"));
        FileNameExtensionFilter pdbFilter = new FileNameExtensionFilter("PDB Files", "pdb");
        chooser.setFileFilter(pdbFilter);
        chooser.setSelectedFile(new File(file));
        
        String pathFileToSave = molViewPanel.getLoadedFile();
        String fileToSave = "";
  
        int actionDialog = chooser.showSaveDialog(view);
        if (actionDialog == JFileChooser.APPROVE_OPTION) {
            File fileName = new File(chooser.getSelectedFile() + ".pdb");
            if (fileName == null) {
                return;
            }
            if (fileName.exists()) {
                actionDialog = JOptionPane.showConfirmDialog(this,
                        "Replace existing file?");
                // may need to check for cancel option as well
                if (actionDialog == JOptionPane.NO_OPTION) {
                    return;
                }
            }
            
            try {
                fileToSave = new Scanner(new File(pathFileToSave)).useDelimiter("\\Z").next();
            } catch (FileNotFoundException ex) {}
            
            try{
                BufferedWriter outFile = new BufferedWriter(new FileWriter(fileName));
                outFile.write(fileToSave); //put in textfile
                outFile.flush(); // redundant, done by close()
                outFile.close();
                //AttestDialog.getInstance( ).showErrorDialog(languageBundle.getString(
                //"LogFil eAlreadyExists"
            }catch (Exception e){}
        }

    }
    
    public void viewPdb(UserFrame view, String structureSequence, int panel){
        
        if(structureSequence.equals("")){
            view.printStringLn("Load a protein first!");
            return;
        }
        if(panel == Defs.ASSEMBLING){
            if (!view.fragmentLoaded()) {
                view.printStringLn("Transfer fragments first!");
                return;
            }
        }
        
        if(panel == Defs.OUTPUT){
            if (!view.isOutputLoaded()) {
                view.printStringLn("Run Fiasco first!");
                return;
            }
        }
        
        //String protein = Defs.PROTEINS_PATH + view.getModel().idProteinCode + ".out.pdb";
        String protein = molViewPanel.getLoadedFile();
        ViewPdbPanel vpp = new ViewPdbPanel(view,protein);
        Thread vp = new Thread(vpp);
        vp.start();
    }
    
    public void changeViewProtein(UserFrame view, String structureSequence, int panel){
        
        if(structureSequence.equals("")){
            view.printStringLn("Load a protein first!");
            return;
        }
        
        if(panel == Defs.ASSEMBLING){
            if (!view.fragmentLoaded()) {
                view.printStringLn("Transfer fragments first!");
                return;
            }
        }
        
        if(panel == Defs.OUTPUT){
            if (!view.isOutputLoaded()) {
                view.printStringLn("Run Fiasco first!");
                return;
            }
        }
        
        Thread threadViewOptions;
        ViewOptionsPanel vop = new ViewOptionsPanel(view, panel);

        /* Create the thread */
        threadViewOptions = new Thread(vop);
        threadViewOptions.start();
    }
    
    /* Get the JMol panel */
    public StructureViewPanel getJMolPanel(){
        return molViewPanel;
    }//getJMolPanel
    
    /* Set the controller instance */
    public void setController(Controller ctr){
        this.ctr = ctr;
    }//setController
    
    /* Execute a script */
    public void executeCmd(String cmd){
        molViewPanel.executeCmd(cmd);
    }//executeCmd
    
    /* Color a specific atom */
    public void colorAtom(String atom, String numFrg, String color){
        //System.out.println(Utilities.selectAtomString(atom, numFrg));
        molViewPanel.executeCmd(Utilities.selectAtomString(atom, numFrg));
        molViewPanel.executeCmd(color); 
    }//colorAA
    
    /* Color a specific AA */
    public void colorResidue(String aa, String numFrg, String color){
        molViewPanel.executeCmd(Utilities.selectResidueString(aa, numFrg));
        molViewPanel.executeCmd(color);
    }//colorAA
    
    /* Color a fragment on the panel */
    public void colorFragment(Fragment frg, String color){
        String selectFragment = Utilities.selectFragmetString(frg);
        molViewPanel.executeCmd(selectFragment);
        molViewPanel.executeCmd(color);
    }//colorFragment
    
    public abstract void event();
    public abstract void changeView();
    public abstract void deselectFragments();
    public abstract void viewPdb();
    public abstract void saveProtein();
    public abstract void switchView(String newView);
    
    
}//MolPanel
