package jafatt;

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.FlowLayout;
import java.awt.GridLayout;
import javax.swing.JLabel;
import javax.swing.JPanel;
import java.awt.event.ActionEvent;
import javax.swing.ImageIcon;
import java.awt.Dimension;

public abstract class MolPanel extends JPanel{
    
    protected StructureViewPanel molViewPanel;
    protected Controller ctr;
    
    private JPanel jp;
    private JLabel panelName;
    private JLabel rmsd;
    private String cmdSetup = Defs.MOLMOUSESETUP;
    private OpButton expandButton;
    
    private String currentDir = System.getProperty("user.dir");
    private String separator = Utilities.getSeparator();
    private String imSuffix = separator + "images" + separator;
    private String expandString = currentDir + imSuffix + "expandPanel.png";
    private String reduceString = currentDir + imSuffix + "reducePanel.png";
    
    ImageIcon expandOn = new ImageIcon(expandString);
    ImageIcon reduceOn = new ImageIcon(reduceString);
    
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
        
        /* Alternative */
        /* jp.setBackground(UserFrame.INNERCOLOR); */
        jp.setBackground(Color.BLACK);
        
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

        
        /* Set the layout */
        setLayout(new BorderLayout());
        //jp.setLayout(new FlowLayout());
        jp.setLayout(new BorderLayout());
        panelName = new JLabel(str);
        panelName.setForeground(Color.green);
        rmsd = new JLabel("RMSD: ");
        rmsd.setForeground(Color.green);
        //jp.add(rmsd, BorderLayout.WEST);
        jp.add(panelName, BorderLayout.CENTER);
        jp.add(expandButton, BorderLayout.EAST);
        add(jp, BorderLayout.NORTH);
        add(molViewPanel, BorderLayout.CENTER);
    }//setup
    
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
        //System.out.println("Before coloring...");
        System.out.println(Utilities.selectAtomString(atom, numFrg));
        molViewPanel.executeCmd(Utilities.selectAtomString(atom, numFrg));
        molViewPanel.executeCmd(color); 
        //System.out.println("Colored");
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
    
    
}//MolPanel
