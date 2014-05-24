package jafatt;

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.GridLayout;
import java.awt.FlowLayout;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.JButton;
import java.awt.Dimension;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import org.biojava.bio.gui.sequence.MultiLineRenderer;
import org.biojava.bio.gui.sequence.OffsetRulerRenderer;
import org.biojava.bio.gui.sequence.SequencePanelWrapper;
import org.biojava.bio.gui.sequence.SymbolSequenceRenderer;
import org.biojava.bio.gui.sequence.tracklayout.SimpleTrackLayout;
import org.biojava.bio.seq.GappedSequence;
import org.biojava.bio.seq.ProteinTools;
import org.biojava.bio.seq.Sequence;
import org.biojava.bio.seq.SequenceTools;
import org.biojava.bio.seq.impl.ViewSequence;
import org.biojava.utils.ChangeVetoException;

public class TargetPanel extends JPanel{
    
    private SequencePanelWrapper sequencePanel;
    private SequencePanelWrapper sequencePanel2;
    private Sequence seq;
    private Sequence seq2;
    private OffsetRulerRenderer offsetRenderer;
    private OpButton cocosButton;
    
    /* Offset string */
    private String offsetString;
    private int targetLength;
    
    UserFrame view;
    Boolean mainPanel;
    
    public TargetPanel(UserFrame view, Boolean mainPanel){
        setupSequencePanel(view, mainPanel);
        setup();
    }
    
    /* Set the sequence visualizer */
    private void setupSequencePanel(UserFrame view, Boolean mainPanel){
        
        this.view = view;
        this.mainPanel = mainPanel;
        
        /* Renderers for the sequences */
        MultiLineRenderer multi = new MultiLineRenderer();
        MultiLineRenderer multi2 = new MultiLineRenderer();
        sequencePanel = new SequencePanelWrapper();
        sequencePanel2 = new SequencePanelWrapper();
        sequencePanel.setSequence(seq);
        sequencePanel2.setSequence(seq2);
        
        /* Set panels and add the ruler */
        sequencePanel.setRenderer(multi);
        sequencePanel2.setRenderer(multi2);
        try{
            multi.addRenderer(new SymbolSequenceRenderer());
            multi.addRenderer( offsetRenderer = new OffsetRulerRenderer() );
            multi2.addRenderer( new SymbolSequenceRenderer() );
            multi2.addRenderer( offsetRenderer = new OffsetRulerRenderer() );
        }catch(ChangeVetoException ex){
            System.out.println("Unable to create the Target Panel");
        }
    }//setupSequencePanel
    
    /* Setup the JPanel */
    private void setup(){
        setLayout( new BorderLayout() );
        JPanel pnl = new JPanel();
        JPanel cocosPanel = new JPanel();
        
        //cocosButton = new JButton("Run Cocos");
        cocosButton = new OpButton("Run Cocos", "") {
            @Override
            public void buttonEvent(ActionEvent evt){
                cocosEvent();
            }
        };

        
        pnl.setLayout(new GridLayout(2, 1));
        sequencePanel.setBackground(Color.WHITE);
        pnl.add(sequencePanel);
        //pnl.add(sequencePanel2);
        cocosPanel.setLayout(new BorderLayout());
        cocosPanel.add(cocosButton, BorderLayout.WEST);
        if(mainPanel){
            pnl.add(cocosPanel);
        }else{
            JPanel empty = new JPanel();
            empty.setBackground(Color.WHITE);
            pnl.add(empty);
        }
        add(new JScrollPane(pnl), BorderLayout.CENTER);
    }//setup
    
    /* Get target sequence (panel) */
    public SequencePanelWrapper getTargetSequence(){
        return sequencePanel;
    }//getTargetSequence
    
    /* Get protein sequence (panel) */
    public SequencePanelWrapper getProteinSequence(){
        return sequencePanel2;
    }//getProteinSequence
    
    /* Set the sequence */
    public void setSequence(ViewSequence seq, int seqType){
        SequencePanelWrapper seqPan;
        if(seqType == Defs.PRIMARY){
            seqPan = sequencePanel;
            
            /* Prepare the offsetString */
            offsetString = "";
            targetLength = seq.length();
            for(int i = 0; i < targetLength; i++)
                offsetString = offsetString + "_";
        }else
            seqPan = sequencePanel2;
        seqPan.setSequence(seq);
        
        /* Layout that wraps the sequence smoothly after a set number of residues */
        seqPan.setTrackLayout(
                new SimpleTrackLayout(seqPan.getSequence(), Defs.INITIALWRAP)
                );
    }//setSequence
    
    /* Set the target sequence */
    public void setTarget(Sequence seq){
        setSequence(SequenceTools.view(seq), Defs.PRIMARY);
    }//setTarget
    
    /* Set the offset sequence by the gapped sequence */
    public void setOffsetTarget(GappedSequence seq){
        setSequence(new ViewSequence(seq), Defs.SECONDARY);
    }//setOffsetTarget
    
    /* Set the offset sequence by the string sequence */
    private void setOffsetTarget(String sequenceString){
        GappedSequence myGappedSequence = null;
        String seqGap = "mySequenceWithGaps";
        String base = "";
        String[] splitted = sequenceString.split("_");

        /* Concatenate residues */
        for(int i = 0; i < splitted.length; i++)
            base = base + splitted[ i ];

        /* Create a gapped sequence from the residue's sequence */
        try{
            myGappedSequence = ProteinTools.createGappedProteinSequence(base, seqGap);
        }catch(Exception ex){
            System.out.println("Unable to load offset residues: " + ex);
        }

        /* Add the gaps on the right place into the gapped sequence */
        char tm = '_';
        for(int i = 0; i < sequenceString.length(); i++)
            if(tm == sequenceString.charAt(i))
                myGappedSequence.addGapInView(i + 1);
        
        /* Set the sequence into the Target panel */
        setOffsetTarget(myGappedSequence);
    }//setOffsetTarget
    
    /* Get the right substring of the loaded protein corresponding 
     * to the fragment */
    public void setOffsetString(String fragmentSequenceString,  
            int fragmentOffset){
        offsetString = Utilities.replaceSubstring(offsetString, 
                fragmentSequenceString, fragmentOffset);
        /* Set the string into the Target panel */
        setOffsetTarget(offsetString);
    }//takeFragmentString
    
    public void updatePanel(){
        //int targetLength = offsetString.length();
        offsetString = "";
        for(int i = 0; i < targetLength; i++)
            offsetString = offsetString + "_";
    }
    
    public void cocosEvent(){
        int[] op = {10,0,0};
        OptionsPanel opPanel = new OptionsPanel(view,view.getModel().getTargetPath());
        Thread panel = new Thread(opPanel);
        panel.start();
        /*
        CocosSolver solver = new CocosSolver(view, op,view.getModel().getTargetPath());
        Thread cocos = new Thread(solver);
        cocos.start();
        */
    }

}//TargetPanel
