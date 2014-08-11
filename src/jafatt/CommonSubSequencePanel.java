package jafatt;

import java.awt.BorderLayout;
import java.awt.Container;
import java.awt.Dimension;
import java.awt.Toolkit;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.AdjustmentEvent;
import java.awt.event.AdjustmentListener;
import java.util.List;
import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import org.biojava.bio.structure.AminoAcid;
import org.biojava.bio.structure.Chain;
import org.biojava.bio.structure.Structure;

public class CommonSubSequencePanel extends JFrame implements Runnable{
    
    private UserFrame view;
    private Controller ctr;
    private JPanel inPanel;
    private MessageArea msArea;
    private JScrollPane scroll;
    private JButton setButton;
    private Structure structure;
    private String firstSubAA;
    private String lastSubAA;
    
    public CommonSubSequencePanel(UserFrame view, Structure structure){
        super("Common Sub-sequence Panel");
        setup(view, structure);
    }
    
    /* Setup */
    private void setup(UserFrame view, Structure structure){
        Toolkit t = Toolkit.getDefaultToolkit();
        Dimension screensize = t.getScreenSize();
        double widthFrame = (screensize.getWidth() * 50.0) / 100.0;
        double heighFrame = (screensize.getHeight() * 35.0) / 100.0;
        
        String proteinSequence = "";
        this.view = view;
        this.structure = structure;
        ctr = view.getController();
        inPanel = new JPanel();
        
        /* Setup layout */
        setLocation((int)(view.getX() + (int)(view.getWidth()/2)),
                    (int)(view.getY() + (int)(view.getHeight()/2)));
        setPreferredSize(new Dimension((int)widthFrame, (int)heighFrame));
        setResizable(true);
        
        inPanel.setLayout(new BorderLayout());
        
        /* Central Panel */
        msArea = new MessageArea();
        
        /* Add the scroll bar and set auto-scroll */
        scroll = new JScrollPane(msArea);
        scroll.getVerticalScrollBar().addAdjustmentListener(new AdjustmentListener(){
            @Override
            public void adjustmentValueChanged(AdjustmentEvent e){
               msArea.select(msArea.getHeight() + 1000, 0);
            }
        });
        
        /* Button */
        setButton = new JButton("Set");
        String os = System.getProperty("os.name").toLowerCase();
            if((os.indexOf("win") >= 0) || (os.indexOf("nix") >= 0))
                setButton.setBorder(
                        new javax.swing.border.SoftBevelBorder(
                                javax.swing.border.BevelBorder.RAISED)
                        );
            setButton.setBackground(Defs.INNERCOLOR);
            setButton.setToolTipText("Click to set the fragment");
            setButton.addActionListener(new ActionListener(){
                @Override
                public void actionPerformed(ActionEvent evt) {
                    buttonSetEvent();
                }  
            });
            setButton.setEnabled(true);
        
        /* Add panels */
        inPanel.add(scroll, BorderLayout.CENTER);
        inPanel.add(setButton, BorderLayout.SOUTH);
        
        /* Get the protein string sequence */
        Chain chain = structure.getChain(0);
        proteinSequence = chain.getSeqResSequence();
        
        /* Print infos */
        printCommonSequence(proteinSequence);
    }//setup
    
    
    /* Print infos about common sub-sequence */
    private void printCommonSequence(String proteinSequence){
        String targetSequence = view.upView.getTargetSequence();
        int firstAAN;
        int lastAAN;
        
        /* Get the common subsequence */
        String[] commonSubSequence = Utilities.maxSubString(proteinSequence, 
                targetSequence);
        
        firstSubAA = commonSubSequence[1];
        lastSubAA = commonSubSequence[2];
        
        /* Get the offset of the protein structure */
        Chain chain = structure.getChain(0);
        
        List groups = chain.getAtomGroups("amino");
        
        /* First AA */
        AminoAcid firstAA = (AminoAcid) groups.get(0);
        int offsetPosition = Utilities.getAAPosition(firstAA);
        
        try{
            firstAAN = Integer.parseInt(firstSubAA);
            lastAAN = Integer.parseInt(lastSubAA);
        }catch(NumberFormatException nfe){
            view.printString("Unable to load common AAs");
            return;
        }
        
        firstAAN = firstAAN + offsetPosition;
        lastAAN = lastAAN + offsetPosition;
        firstSubAA = "" + firstAAN;
        lastSubAA = "" + lastAAN;
        
        /* Print infos */
        msArea.writeln("Target and Protein's sequence:");
        msArea.writeln(targetSequence, false);
        msArea.writeln();
        msArea.writeln();
        msArea.writeln(proteinSequence, false);
        msArea.writeln();
        msArea.writeln("Maximum common subsequence:");
        msArea.writeln(commonSubSequence[0], false);
        msArea.writeln("of length:");
        msArea.writeln("" + commonSubSequence[0].length(), false);
        msArea.writeln("From residue n. " + firstSubAA +
                " to residue n. " + lastSubAA + " in the protein's sequence");
        msArea.writeln("Note:");
        msArea.writeln("There is an offset of " + offsetPosition + " for the "
                + "first AA of the protein's sequence");
        
    }//printCommonSequence
    
    /* Action of setting the common sub sequence beetwen target and loaded
     * protein
     */
    private void buttonSetEvent(){
        
        /* Create a new fragment */
        String[] infoAA1 = new String[7];
        String[] infoAA2 = new String[7];
            
        for(int i = 0; i < 7; i++){
            infoAA1[i] = Defs.EMPTY;
            infoAA2[i] = Defs.EMPTY;
        }
            
        infoAA1[Defs.INFO_RESNUM] = firstSubAA;
        infoAA2[Defs.INFO_RESNUM] = lastSubAA;
        
        Fragment frg = view.upView.createFragment(infoAA1, infoAA2);
        
        /* Color the fragment
         * Note:
         The method doesn't pass through the Controller though the MVC pattern 
         */
        ctr.colorFragment(frg, Defs.COLOR_COMMON_SUBSEQUENCE, Defs.EXTRACTION);
            
        msArea.writeln("Note:");
        msArea.writeln("The maximum common subsequence is only colored.", false);
        msArea.writeln("To set it, please use the Select Fragments Panel.", false);
    }//buttonSetEvent

    @Override
    public void run(){
        pack();
        Container ct = getContentPane();
        ct.add(inPanel);
        setVisible(true);
        
    }//run
    
}//CommonSubSequencePanel
