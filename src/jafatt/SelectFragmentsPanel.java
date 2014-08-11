package jafatt;

import java.awt.BorderLayout;
import java.awt.Container;
import java.awt.Dimension;
import java.awt.GridLayout;
import java.awt.Toolkit;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.AdjustmentEvent;
import java.awt.event.AdjustmentListener;
import java.util.List;
import javax.swing.ImageIcon;
import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.JTextField;
import org.biojava.bio.structure.AminoAcid;
import org.biojava.bio.structure.Chain;
import org.biojava.bio.structure.Structure;
import org.biojava.bio.seq.Sequence;

public class SelectFragmentsPanel extends JFrame implements Runnable{
    
    private UserFrame view;
    private Controller ctr;
    private TargetPanel tp;
    private SetFragmentPanel setFragmentPanel;
    private MessageArea msArea;
    
    private JPanel inPanel;
    private JScrollPane scroll;
    
    public SelectFragmentsPanel(UserFrame view, Structure structure){
        super("Select Fragments Panel");
        setup(view, structure);
    }
    
    /* Setup */
    private void setup(UserFrame view, Structure structure){
        Toolkit t = Toolkit.getDefaultToolkit();
        Dimension screensize = t.getScreenSize();
        double widthFrame = (screensize.getWidth() * 30.0) / 100.0;
        double heighFrame = (screensize.getHeight() * 38.0) / 100.0;
        
        this.view = view;
        
        setIconImage(new ImageIcon(view.frameIcon).getImage());
        
        ctr = view.getController();
        inPanel = new JPanel();
        
        /* Setup layout */
        //setPreferredSize(new Dimension((int)widthFrame, (int)heighFrame));
        setResizable(true);
        
        inPanel.setLayout(new BorderLayout());
        
        /* Upper Panel*/
        
        tp = new TargetPanel();
        
        Sequence seq = view.getModel().getTargetSequence();
        
        /* Set the sequence of the loaded protein*/
        Chain chain = structure.getChain(0);
        try{
            //tp.setTarget(chain.getBJSequence());
            tp.setTarget(seq);
        }catch(Exception e){
            view.printStringLn("Error in SelectFragments Panel: " + e);
            return;
        }
        
        /* Check the offset of the loaded protein (i.e., the first AA position) */
        List groups = chain.getAtomGroups("amino");
        
        /* First and last AA */
        AminoAcid firstAA = (AminoAcid) groups.get(0);
        AminoAcid lastAA = (AminoAcid) groups.get(groups.size() - 1);
        int firstPosition = Utilities.getAAPosition(firstAA);
        int lastPosition = Utilities.getAAPosition(lastAA);
        
        if(firstPosition == -1 || lastPosition == -1){
            view.printStringLn("Error in calculating the offset of the protein");
            return;
        }
        
        /* Internal panel */
        setFragmentPanel = new SetFragmentPanel(ctr, firstPosition, lastPosition);

        
        /* Lower Panel */
        msArea = new MessageArea(3, 10);
        
        /* Add the scroll bar and set auto-scroll */
        scroll = new JScrollPane(msArea);
        scroll.getVerticalScrollBar().addAdjustmentListener(new AdjustmentListener(){
            @Override
            public void adjustmentValueChanged(AdjustmentEvent e){
               msArea.select(msArea.getHeight() + 1000, 0);
            }
        });
        
        /* Add panels */
        inPanel.add(tp, BorderLayout.NORTH);
        inPanel.add(setFragmentPanel, BorderLayout.CENTER);
        inPanel.add(scroll, BorderLayout.SOUTH);
        
        /* Print some infos */
        msArea.writeln("Please, set the fragment between the first and the "
                + "last residue", false);
    }//setup
    
    @Override
    public void run() {
        Container ct = getContentPane();
        ct.add(inPanel);
        pack();
        setLocationRelativeTo(view);
        setVisible(true);
    }//run
    
    private class SetFragmentPanel extends JPanel{
        
        private Controller ctr;
        
        /* Components */
        JPanel setPanel;
        
        JLabel AAStart;
        JLabel AAEnd;
        JLabel AAFirst;
        JLabel AALast;
        
        JTextField AASText;
        JTextField AAEText;

        JButton setFragmentButton;
        
        /* Values */
        int firstAAN;
        int lastAAN;
        int AAStartN;
        int AAEndN;
        
        public SetFragmentPanel(Controller ctr, int first, int last){
            setup(ctr, first, last);
        }
        
        /* Setup */
        private void setup(Controller ctr, int first, int last){
            this.ctr = ctr;
            
            firstAAN = first;
            lastAAN = last;
            AAStartN = -1;
            AAEndN = -1;
            
            setPanel = new JPanel();
            
            /* Labels */
            AAFirst = new JLabel("First_AA: " + firstAAN);
            AALast = new JLabel("Last_AA: " + lastAAN);
            AAStart = new JLabel("Fragment_Start");
            AAEnd = new JLabel("Fragment_End");
            
            /* Text fields */
            AASText = new JTextField();
            AASText.setEnabled(true);
            AAEText = new JTextField();
            AAEText.setEnabled(true);
            
            /* Button layout */
            setFragmentButton = new JButton("Set Fragment");
            String os = System.getProperty("os.name").toLowerCase();
            if((os.indexOf("win") >= 0) || (os.indexOf("nix") >= 0))
                setFragmentButton.setBorder(
                        new javax.swing.border.SoftBevelBorder(
                                javax.swing.border.BevelBorder.RAISED)
                        );
            setFragmentButton.setBackground(Defs.INNERCOLOR);
            setFragmentButton.setToolTipText("Click to set the fragment");
            setFragmentButton.addActionListener(new ActionListener(){
                @Override
                public void actionPerformed(ActionEvent evt) {
                    buttonFragmentEvent();
                }  
            });
            setFragmentButton.setEnabled(true);
             
            setPanel.setLayout(new GridLayout(3, 2));
            setLayout(new BorderLayout());
            
            /* Add components */
            setPanel.add(AAFirst);
            setPanel.add(AALast);
            setPanel.add(AAStart);
            setPanel.add(AASText);
            setPanel.add(AAEnd);
            setPanel.add(AAEText);
            
            add(setPanel, BorderLayout.CENTER);
            add(setFragmentButton, BorderLayout.SOUTH);
        }//setup
        
        /* Set the fragment on Enxtraction panel with the infos within the
         * JTextFields
         */
        private void buttonFragmentEvent(){
            String AA1;
            String AA2;
            
            /* Get the values */
            AA1 = AASText.getText();
            AA2 = AAEText.getText();
            
            /* Check the values */
            try{
                AAStartN = Integer.parseInt(AA1);
                AAEndN = Integer.parseInt(AA2);
            }catch(NumberFormatException nfe){
                AAStartN = -1;
                AAEndN = -1;
                AASText.setText("");
                AAEText.setText("");
                msArea.writeln("Not a number: " + nfe, false);
                return;
            }
            
            if(AAStartN < firstAAN || AAStartN > lastAAN ||
               AAEndN < firstAAN || AAEndN > lastAAN){
               AASText.setText("");
               AAEText.setText("");
               msArea.writeln("Please, put numbers between the first and "
                       + "the last AA", false);
               return; 
            }
            
            /* Infos like a picked AA */
            String[] infoAA1 = new String[7];
            String[] infoAA2 = new String[7];
            
            for(int i = 0; i < 7; i++){
                infoAA1[i] = Defs.EMPTY;
                infoAA2[i] = Defs.EMPTY;
            }
            
            infoAA1[Defs.INFO_RESNUM] = AA1;
            infoAA2[Defs.INFO_RESNUM] = AA2;
            
            /* Check if a residue is already picked */
            if(view.getController().isResidueAlreadyPresent(infoAA1) || 
               view.getController().isResidueAlreadyPresent(infoAA1)){
                AASText.setText("");
                AAEText.setText("");
                msArea.writeln("Residues already selected, please "
                        + "select differents residues", false);
                return;     
            }
            
            /* Set the new fragment */
            view.getController().usePickedInfo(infoAA1, Defs.EXTRACTION);
            view.getController().usePickedInfo(infoAA2, Defs.EXTRACTION);
            
        }//buttonFragmentEvent
        
    }//SetFragmentPanel
    
}//SelectFragmentsPanel
