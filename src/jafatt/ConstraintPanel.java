package jafatt;

import java.awt.BorderLayout;
import java.awt.Container;
import java.awt.Dimension;
import java.awt.Toolkit;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.AdjustmentEvent;
import java.awt.event.AdjustmentListener;
import java.awt.FlowLayout;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.StringTokenizer;
import javax.swing.ImageIcon;
import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.ScrollPaneConstants;
import javax.swing.SpringLayout;
import javax.swing.JComboBox;
import javax.swing.JCheckBox;
import javax.swing.BorderFactory;
import java.awt.GridBagLayout;
import java.awt.GridBagConstraints;
import java.util.List;
import org.biojava.bio.structure.Structure;
import org.biojava.bio.structure.Chain;
import org.biojava.bio.structure.AminoAcid;

public class ConstraintPanel extends JFrame implements Runnable{
    
    private SetConstraintPanel setConstraintPanel;
    private MessageArea msArea;
    private ArrayList<Integer> jmList;
    private ArrayList<Integer> ussList;
    private ArrayList<Integer> dgeqList;
    private ArrayList<Integer> dleqList;
    private ArrayList<Double> uniList;
    private ArrayList<Double> ellList;
    private ArrayList<Double> spList;
    private boolean[] bool;
    private int[] values;
    private int jm = -1;
    private int firstjm, secondjm;
    private int firstnc, secondnc;
    private double firstsp;
    private int secondsp;
    private int voxelS;
    private int firstuss, seconduss;
    private int firstdgeq,  seconddgeq, dgeq;
    private int firstdleq,  seconddleq, dleq;
    int firstPosition, lastPosition;
    private ArrayList<String> choice;
    private ArrayList<Fragment> fragments;
    private Structure structure;
    private UserFrame view;
    
    private JPanel inPanel, inSubPanel, exitPanel;
    private JPanel optionPanel;
    private JScrollPane scroll,scr;
    
    private JButton ok, cancel, info, expand;
    
    private boolean expanded = false;
    //public boolean distance = true;
    
    private ConstraintInfoPanel cip;
    
    private String currentDir = System.getProperty("user.dir");
    private String separator = Utilities.getSeparator();
    private String imSuffix = separator + "images" + separator;
    private String expandImage = currentDir + imSuffix + "expand.png";
    private String reduceImage = currentDir + imSuffix + "reduce.png";
    
    ImageIcon iconExpand = new ImageIcon(expandImage);
    ImageIcon iconReduce = new ImageIcon(reduceImage);

    
    public ConstraintPanel(UserFrame view, Structure structure, ArrayList<Fragment> fragments){
        super("Constraint Panel");
        setup(view, structure, fragments);
    }
    
    /* Setup */
    private void setup(UserFrame view, Structure structure, ArrayList<Fragment> fragments){
        
        this.view = view;
        this.structure = structure;
        this.fragments = fragments;
        
        Chain chain = structure.getChain(0);
        List groups = chain.getAtomGroups("amino");
        AminoAcid firstAA = (AminoAcid) groups.get(0);
        AminoAcid lastAA = (AminoAcid) groups.get(groups.size() - 1);
        firstPosition = Utilities.getAAPosition(firstAA);
        lastPosition = Utilities.getAAPosition(lastAA);
        
        inPanel = new JPanel(new BorderLayout());
        inSubPanel = new JPanel(new BorderLayout());
        optionPanel = new JPanel();
        exitPanel = new JPanel();
        
        /* Setup layout */
        
        bool = new boolean[10];
        Arrays.fill(bool, Boolean.FALSE);
        values = new int[10];
        
        msArea = new MessageArea(3, 10);
        
        /* Internal panel */
        setConstraintPanel = new SetConstraintPanel();
        setConstraintPanel.setBorder(BorderFactory.createTitledBorder("Options"));
        scr = new JScrollPane(setConstraintPanel,
                ScrollPaneConstants.VERTICAL_SCROLLBAR_AS_NEEDED,
                ScrollPaneConstants.HORIZONTAL_SCROLLBAR_AS_NEEDED);
        
        /* Add the scroll bar and set auto-scroll */
        scroll = new JScrollPane(msArea);
        scroll.getVerticalScrollBar().addAdjustmentListener(new AdjustmentListener(){
            @Override
            public void adjustmentValueChanged(AdjustmentEvent e){
               msArea.select(msArea.getHeight() + 1000, 0);
            }
        });
        
        ok = new OpButton("Run!", "Set all the constraint and Run Fiasco!") {
            @Override
            public void buttonEvent(ActionEvent evt){
                setConstraintPanel.setConstraints();
                solveEvent();
                setVisible(false);
                dispose();
            }
        };
        
        cancel = new OpButton("Cancel", "Delete all the constraint and exit") {
            @Override
            public void buttonEvent(ActionEvent evt){
                setConstraintPanel.cancelEvent();
                setVisible(false);
                dispose();
            }
        };
        
        info = new OpButton("?", "View info about the constraints") {
            @Override
            public void buttonEvent(ActionEvent evt){
                infoEvent();                
            }
        };
        
        expand = new OpButton(iconExpand, "Show more options"){
            @Override
            public void buttonEvent(ActionEvent evt){
                expandEvent();
                setConstraintPanel.update(expanded);    
                //getRootPane().revalidate();
                update();
            }
        };
        expand.setPreferredSize(new Dimension(140, 15));
        expand.setFocusPainted(false);
        expand.setBorderPainted(false);
        
        exitPanel.add(ok);
        exitPanel.add(cancel);
        exitPanel.add(info);
        
        optionPanel.setLayout(new BorderLayout());
        
        optionPanel.add(expand, BorderLayout.NORTH);
        optionPanel.add(exitPanel, BorderLayout.CENTER);
		
        inSubPanel.add(scr, BorderLayout.CENTER);
        inSubPanel.add(optionPanel, BorderLayout.SOUTH);
        
        /* Add panels */
        inPanel.add(inSubPanel, BorderLayout.CENTER);
        inPanel.add(scroll, BorderLayout.SOUTH);
        /* Print some infos */
        msArea.writeln("Please set the constraint to the Fiasco.", false);
    }//setup
    
    @Override
    public void run() {
        Container ct = getContentPane();
        ct.add(inPanel);
        update();
        setVisible(true);
    }//run
    
    private void update(){
        pack();
        setLocationRelativeTo(view);
    }
    
    private void solveEvent(){
        
        ((AssemblingPanel) view.getPanel(Defs.ASSEMBLING)).runFiasco();

    }
    
    public void infoEvent(){
        cip = new ConstraintInfoPanel();
                
        /* Create the thread */
        Thread threadSelectFragments;
        threadSelectFragments = new Thread(cip);
        threadSelectFragments.start();
    }
    
    public void expandEvent(){
        if(expanded){
            expanded = false;
            expand.setIcon(iconExpand);
        }else{
            expanded = true;
            expand.setIcon(iconReduce);            
        }
    }
    
    public boolean measures(){
        return setConstraintPanel.measures.isSelected();
    }
    
    private class SetConstraintPanel extends JPanel{
        
        /* Components */
        JPanel setPanel;
        JPanel distancePanel;
        JPanel uniPanel;
        JPanel ellPanel;
        JPanel jmPanel, ussPanel;
        JPanel advancedPanel;
        
        JLabel dsLabel, msLabel;
        JLabel tsLabel, ttLabel;
        JLabel jmLabel, clustLabel;
        JLabel spLabel, ussLabel;
        JLabel voxLabel;
        JLabel dgeqLabel, dleqLabel;
        JLabel uniLabel, ellLabel;
        JLabel sumRadiiLabel;
        JLabel armstrong, aminoacid;
        
        HintTextField domainSizeText;
        HintTextField maxSolText;
        HintTextField timeOutSearchText;
        HintTextField timeOutTotalText;
        HintTextField fdgeqText, sdgeqText, dgeqText;
        HintTextField fdleqText, sdleqText, dleqText;
        HintTextField uniText, uniVoxText;
        HintTextField ellText, sumRadiiText;
        HintTextField jm1Text, jm2Text;
        HintTextField clust1Text, clust2Text;
        HintTextField sp1Text, sp2Text;
        HintTextField uss1Text, uss2Text;
        HintTextField voxText;
        HintTextField x1Text, y1Text, z1Text;
        HintTextField x2Text, y2Text, z2Text;
        
        //JButton addDS, addMS, addTS, addTT;
        OpButton addJM, addUSS;
        OpButton addDGEQ, addDLEQ;
        OpButton addUniform, addEllipsoid;
        
        JComboBox arrows;
        JCheckBox measures;
        
        GridBagConstraints c;

        /* Values */
        
        public SetConstraintPanel(){
            setup();
        }
        
        /* Setup */
        private void setup(){
            
            setPanel = new JPanel(new SpringLayout());
            distancePanel = new JPanel(new SpringLayout());
            uniPanel = new JPanel();
            ellPanel = new JPanel();
            jmPanel = new JPanel();
            ussPanel = new JPanel();
            advancedPanel = new JPanel(new GridBagLayout());
            
            jmList = new ArrayList<Integer>();
            ussList = new ArrayList<Integer>();
            spList = new ArrayList<Double>();
            dgeqList = new ArrayList<Integer>();
            dleqList = new ArrayList<Integer>();
            uniList = new ArrayList<Double>();
            ellList = new ArrayList<Double>();
            choice = new ArrayList<String>();
            
            c = new GridBagConstraints();
            
            addDGEQ = new OpButton("Add D-GEQ", "Click to set a Distant GEQ Constraint") {
                @Override
                public void buttonEvent(ActionEvent evt){
                    addDGEQEvent();
                }
            };
            
            addDLEQ = new OpButton("Add D-LEQ", "Click to set a Distant LEQ Constraint") {
                @Override
                public void buttonEvent(ActionEvent evt){
                    addDLEQEvent();
                }
            };
            
            addUniform = new OpButton("Add Uniform", "Click to set a uniform Constraint") {
                @Override
                public void buttonEvent(ActionEvent evt){
                    addUniformEvent();
                }
            };
            
            addEllipsoid = new OpButton("Add Ellipsoid", "Click to set a ellipsoid Constraint") {
                @Override
                public void buttonEvent(ActionEvent evt){
                    addEllEvent();
                }
            };            
            
            /*Add JM Button*/
            
            addJM = new OpButton("Add JM", "Click to set a JM Constraint") {
                @Override
                public void buttonEvent(ActionEvent evt){
                    addJMEvent();
                }
            };            
            
            /*Add unique source sinks Button*/
            
            addUSS = new OpButton("Add USS", "Click to set a USS Constraint") {
                @Override
                public void buttonEvent(ActionEvent evt){
                    addUSSEvent();
                }
            };            
            
            arrows = new JComboBox();
            arrows.addItem("->");
            //arrows.addItem("<-");   //remove comment after possibly implementation
            //arrows.addItem("<->");
            arrows.setSelectedIndex(0);
            
            measures = new JCheckBox("Add Distance Constraints.",true);
            //msArea.writeln("Distance Constraint could affect"
            //        + " the number of the solutions.");

            measures.addActionListener(
                    new ActionListener(){
                        @Override
                        public void actionPerformed(ActionEvent e){
                            if(measures.isSelected()){
                                msArea.writeln("Distance Constraint could affect"
                                        + " the number of the solutions.");
                                //distance = true;
                            }else{
                                //distance = false;
                            }
                            
                        }
                    });

             /* Text fields */
            domainSizeText = new HintTextField(" Int ", 10);
            domainSizeText.setHintText("10");
            domainSizeText.setEnabled(true);
            
            maxSolText = new HintTextField(" Int ", 10);
            maxSolText.setHintText("1000000");
            maxSolText.setEnabled(true);
            
            timeOutSearchText = new HintTextField(" Sec ", 10);
            timeOutSearchText.setHintText("60");
            timeOutSearchText.setEnabled(true);
            
            timeOutTotalText = new HintTextField(" Sec ", 10);
            timeOutTotalText.setHintText("120");
            timeOutTotalText.setEnabled(true);
            
            fdgeqText = new HintTextField(" Start AA ", 10);
            fdgeqText.setEnabled(true);
            sdgeqText = new HintTextField(" End AA ", 10);
            sdgeqText.setEnabled(true);
            dgeqText = new HintTextField(" Distance in \u212B ", 10);
            dgeqText.setEnabled(true);
            
            fdleqText = new HintTextField(" Start AA ", 10);
            fdleqText.setEnabled(true);
            sdleqText = new HintTextField(" End AA ", 10);
            sdleqText.setEnabled(true);
            dleqText = new HintTextField(" Distance in \u212B ", 10);
            dleqText.setEnabled(true);
            
            uniText = new HintTextField(" List of AA ", 15);
            uniText.setEnabled(true);
            uniText.setPreferredSize(addUniform.getPreferredSize());
            uniVoxText = new HintTextField(" Voxel in \u212B ",  10);
            uniVoxText.setEnabled(true);
            uniVoxText.setPreferredSize(addUniform.getPreferredSize());
            
            ellText = new HintTextField(" List of AA ", 15);
            ellText.setEnabled(true);
            ellText.setPreferredSize(addEllipsoid.getPreferredSize());
            x1Text= new HintTextField(" x ", 3);
            x1Text.setEnabled(true);
            x1Text.setPreferredSize(addEllipsoid.getPreferredSize());
            y1Text= new HintTextField(" y ", 3);
            y1Text.setEnabled(true);
            y1Text.setPreferredSize(addEllipsoid.getPreferredSize());
            z1Text= new HintTextField(" z ", 3);
            z1Text.setEnabled(true);
            z1Text.setPreferredSize(addEllipsoid.getPreferredSize());
            x2Text= new HintTextField(" x ", 3);
            x2Text.setEnabled(true);
            x2Text.setPreferredSize(addEllipsoid.getPreferredSize());
            y2Text= new HintTextField(" y ", 3);
            y2Text.setEnabled(true);
            y2Text.setPreferredSize(addEllipsoid.getPreferredSize());
            z2Text= new HintTextField(" z ", 3);
            z2Text.setEnabled(true);
            z2Text.setPreferredSize(addEllipsoid.getPreferredSize());
            sumRadiiText= new HintTextField(" Int ", 3);
            sumRadiiText.setEnabled(true);            
            sumRadiiText.setPreferredSize(addEllipsoid.getPreferredSize());
            
            jm1Text = new HintTextField(" Start AA ", 6);
            jm1Text.setEnabled(true);
            jm1Text.setPreferredSize(addJM.getPreferredSize());
            jm2Text = new HintTextField(" End AA ", 6);
            jm2Text.setEnabled(true);
            jm2Text.setPreferredSize(addJM.getPreferredSize());
            clust1Text = new HintTextField(" Min ", 3);
            clust1Text.setEnabled(true);
            clust1Text.setPreferredSize(addJM.getPreferredSize());
            clust2Text = new HintTextField(" Max ", 3);
            clust2Text.setEnabled(true);
            clust2Text.setPreferredSize(addJM.getPreferredSize());
            sp1Text = new HintTextField(" Tol in \u212B ", 7);
            sp1Text.setEnabled(true);
            sp1Text.setPreferredSize(addJM.getPreferredSize());
            sp2Text = new HintTextField(" Tol in deg ", 7);
            sp2Text.setEnabled(true);
            sp2Text.setPreferredSize(addJM.getPreferredSize());
            
            uss1Text = new HintTextField(" Start AA ", 6);
            uss1Text.setEnabled(true);
            uss1Text.setPreferredSize(addUSS.getPreferredSize());
            uss2Text = new HintTextField(" End AA ", 6);
            uss2Text.setEnabled(true);
            uss2Text.setPreferredSize(addUSS.getPreferredSize());
            voxText = new HintTextField(" Voxel in \u212B ", 10);
            voxText.setEnabled(true);
            voxText.setPreferredSize(addUSS.getPreferredSize());


            setPanel.add(dsLabel = new JLabel(" Domain Size: ", JLabel.TRAILING));
            dsLabel.setLabelFor(domainSizeText);
            setPanel.add(domainSizeText);
            //setPanel.add(addDS);
            
            setPanel.add(msLabel = new JLabel(" Max Solutions: ", JLabel.TRAILING));
            msLabel.setLabelFor(maxSolText);
            setPanel.add(maxSolText);
            //setPanel.add(addMS);

            setPanel.add(tsLabel = new JLabel(" Timeout Search (sec.): ", JLabel.TRAILING));
            tsLabel.setLabelFor(timeOutSearchText);
            setPanel.add(timeOutSearchText);
            //setPanel.add(addTS);
            
            setPanel.add(ttLabel = new JLabel(" Timeout Total (sec.): ", JLabel.TRAILING));
            ttLabel.setLabelFor(timeOutTotalText);
            setPanel.add(timeOutTotalText);
            //setPanel.add(addTT);
            
            distancePanel.add(dgeqLabel = new JLabel( " Distance greater than equal: ", JLabel.TRAILING));
            dgeqLabel.setLabelFor(fdgeqText);
            distancePanel.add(fdgeqText);
            distancePanel.add(sdgeqText);
            distancePanel.add(dgeqText);
            distancePanel.add(addDGEQ);
            
            distancePanel.add(dleqLabel = new JLabel( " Distance less than equal: ", JLabel.TRAILING));
            dleqLabel.setLabelFor(fdleqText);
            distancePanel.add(fdleqText);
            distancePanel.add(sdleqText);
            distancePanel.add(dleqText);
            distancePanel.add(addDLEQ);

            uniPanel.add(uniLabel = new JLabel( " Unique: ", JLabel.TRAILING));
            uniLabel.setLabelFor(uniText);
            uniPanel.add(uniText);
            uniPanel.add(new JLabel (" Voxel Side: "));
            uniPanel.add(uniVoxText);
            uniPanel.add(addUniform);
            
            ellPanel.add(ellLabel = new JLabel( " Ellipsoid: ", JLabel.TRAILING));
            ellLabel.setLabelFor(ellText);
            ellPanel.add(ellText);
            ellPanel.add(new JLabel (" focus1: "));
            ellPanel.add(x1Text);
            ellPanel.add(y1Text);
            ellPanel.add(z1Text);
            ellPanel.add(new JLabel (" focus2: "));
            ellPanel.add(x2Text);
            ellPanel.add(y2Text);
            ellPanel.add(z2Text);
            ellPanel.add(new JLabel (" Sum-radii: "));
            ellPanel.add(sumRadiiText);
            ellPanel.add(addEllipsoid);
            
            
            jmPanel.setLayout (new FlowLayout());
            jmPanel.add(new JLabel(" JM: "));
            jmPanel.add(jm1Text);
            jmPanel.add(arrows);
            jmPanel.add(jm2Text);
            jmPanel.add(new JLabel(" Number of Clusters: "));
            jmPanel.add(clust1Text);
            jmPanel.add(clust2Text);
            jmPanel.add(new JLabel(" Sim Parameters: "));
            jmPanel.add(sp1Text);
            jmPanel.add(sp2Text);
            jmPanel.add(addJM);
            
            
            ussPanel.setLayout (new FlowLayout());
            ussPanel.add(new JLabel(" Unique Source Sinks: "));
            ussPanel.add(uss1Text);
            ussPanel.add(uss2Text);
            ussPanel.add(new JLabel (" Voxel Side: "));
            ussPanel.add(voxText);
            ussPanel.add(addUSS);

            SpringUtilities.makeCompactGrid(setPanel,4,2,10,10,10,10);
            SpringUtilities.makeCompactGrid(distancePanel,2,5,10,10,10,10);
            
            
            setLayout(new GridBagLayout());
            
            advancedPanel.setBorder(BorderFactory.createTitledBorder("Advanced Options"));
            c.gridy = 0;
            advancedPanel.add(distancePanel, c);
            c.gridy = 1;
            advancedPanel.add(uniPanel, c);
            c.gridy = 2;
            advancedPanel.add(ellPanel, c);
            c.gridy = 3;
            advancedPanel.add(jmPanel, c);
            c.gridy = 4;
            advancedPanel.add(ussPanel, c);
            
            c.fill = GridBagConstraints.VERTICAL;
            c.gridx = 0;
            c.gridy = 0;
            
            add(setPanel, c);
            c.gridy = 1;
            add(measures,c);
            c.gridy = 2;
            add(advancedPanel,c);
            
            update(false);
            
            
        }//setup
        
        public void update(Boolean expanded){
            
            advancedPanel.setVisible(expanded);
        }
        
        private void addJMEvent(){
            
            bool[Defs.JM] = true;
            
            String jm1, jm2;
            String nc1, nc2;
            String sp1, sp2;
            
            /* Get the values */
            jm1 = jm1Text.getText();
            jm2 = jm2Text.getText();
            nc1 = clust1Text.getText();
            nc2 = clust2Text.getText();
            sp1 = sp1Text.getText();
            sp2 = sp2Text.getText();
            
            /* Check the values */
            try{
                firstjm = Integer.parseInt(jm1);
                secondjm = Integer.parseInt(jm2);
                firstnc = Integer.parseInt(nc1);
                secondnc = Integer.parseInt(nc2);
                firstsp = Double.parseDouble(sp1);
                secondsp = Integer.parseInt(sp2);
            }catch(NumberFormatException nfe){
                jm1Text.setHintText("");
                jm2Text.setHintText("");
                clust1Text.setHintText("");
                clust2Text.setHintText("");
                sp1Text.setHintText("");
                sp2Text.setHintText("");
                msArea.writeln("Not a number: " + nfe, true);
                return;
            }
            
            // if the jm starts before the protein's starting point
            if (firstjm < firstPosition) {
                //msArea.writeln("Error in JM", true);
                errorJM(jm);
                return;
            }
            // if the jm ends after the protein's ending point
            if (secondjm > lastPosition) {
                //msArea.writeln("Error in JM", true);
                errorJM(jm);
                return;
            }
                      
            for (int i = 0; i < fragments.size() - 1; i++) {
                // if the jm starts inside a fragment
                if (firstjm < Integer.parseInt(fragments.get(i).getParameters()[Defs.FRAGMENT_END_AA])
                        && firstjm > Integer.parseInt(fragments.get(i).getParameters()[Defs.FRAGMENT_START_AA])) {
                    //msArea.writeln("Error in JM", true);
                    errorJM(jm);
                    return;
                }
                // if the jm ends inside a fragment
                if (secondjm > Integer.parseInt(fragments.get(i).getParameters()[Defs.FRAGMENT_START_AA])
                        && secondjm < Integer.parseInt(fragments.get(i).getParameters()[Defs.FRAGMENT_END_AA])) {
                    //msArea.writeln("Error in JM", true);
                    errorJM(jm);
                    return;
                }
            }
            
            if (jm != -1){  //if a JM constraint is already been fixed
                if(firstjm <= jm){  //check whether two JM overlap
                    //msArea.writeln("Error in JM", true);
                    errorJM(jm);
                    return;
                }
            }
            
            if(firstjm >= secondjm){
               jm1Text.setHintText("");
               jm2Text.setHintText("");
               msArea.writeln("Please, select proper JM values", true);
               return; 
            }
            
            jm = secondjm;           
            
            
            arrows.addActionListener(new ActionListener(){
                @Override
                public void actionPerformed(ActionEvent evt) {
                    choice.add((String)arrows.getSelectedItem());
                }  
            });
            choice.add((String)arrows.getSelectedItem());
            jmList.add(firstjm);
            jmList.add(secondjm);
            jmList.add(firstnc);
            jmList.add(secondnc);
            spList.add(firstsp);
            jmList.add(secondsp);
            
            msArea.writeln("Constraint \"--jm " + firstjm + " '"
                    + (String)arrows.getSelectedItem() + "' "
                    + secondjm + " : numof-clusters= " + firstnc + " "
                    + secondnc + " " + " sim-param=" + firstsp + " "
                    + secondsp + "\" added", true);
            
            jm1Text.setHintText("");
            jm2Text.setHintText("");
            clust1Text.setHintText("");
            clust2Text.setHintText("");
            sp1Text.setHintText("");
            sp2Text.setHintText("");
            
        }
        
        private void errorJM(int jm){
            msArea.write("JM constraints should be fixed between the following interval(s): ", true);
            for(int i=0; i<fragments.size()-1;i++){
                int value = Integer.parseInt(fragments.get(i).getParameters()[Defs.FRAGMENT_END_AA]);
                if (jm > Integer.parseInt(fragments.get(i).getParameters()[Defs.FRAGMENT_END_AA]) &&
                    jm < Integer.parseInt(fragments.get(i+1).getParameters()[Defs.FRAGMENT_START_AA])){
                    value = jm + 1;
                }
                msArea.write("[" + value + "," + Integer.parseInt(fragments.get(i+1).getParameters()[Defs.FRAGMENT_START_AA])
                        + "] ");
            }
            msArea.write(". ");
            if(jm != -1){
                msArea.write("The next JM has to start in position " + (jm+1) + " in the current interval.");
                //msArea.write("The next JM has to start in position " + (jm+1) + " in the current interval.");
            }
            msArea.writeln();
        }
        
        private void addUSSEvent(){
            
            bool[Defs.USS] = true;
            
            String uss1, uss2;
            String vox;
            
            /* Get the values */
            uss1 = uss1Text.getText();
            uss2 = uss2Text.getText();
            vox = voxText.getText();
            
            /* Check the values */
            try{
                firstuss = Integer.parseInt(uss1);
                seconduss = Integer.parseInt(uss2);
                voxelS = Integer.parseInt(vox);
            }catch(NumberFormatException nfe){
                uss1Text.setHintText("");
                uss2Text.setHintText("");
                voxText.setHintText("");
                msArea.writeln("Not a number: " + nfe, true);
                return;
            }
            
            if(firstuss >= seconduss){
               uss1Text.setHintText("");
               uss2Text.setHintText("");
               msArea.writeln("Please, select proper USS values", true);
               return; 
            }
            
            ussList.add(firstuss);
            ussList.add(seconduss);
            ussList.add(voxelS);
            
            msArea.writeln("Constraint \"--unique-source-sinks " + firstuss
                    + " '->' " + " " + seconduss + " : voxel-side "
                    + voxelS + "\" added", true);
            
            uss1Text.setHintText("");
            uss2Text.setHintText("");
            voxText.setHintText("");
            
            
        }
        
        private void addDSEvent(){
            
            bool[Defs.DOMAIN] = true;
            
            String ds;
            
            /* Get the values */
            ds = domainSizeText.getText();
            
            /* Check the values */
            try{
                if (Integer.parseInt(ds) <= 0){
                    msArea.writeln("Please select proper Domain Size value.");
                    return;
                }
                values[Defs.DOMAIN] = Integer.parseInt(ds);
            }catch(NumberFormatException nfe){
                domainSizeText.setHintText("");
                msArea.writeln("Not a number: " + nfe, true);
                return;
            }
            
            msArea.writeln("Constraint \"--domain-size " +
                    + values[Defs.DOMAIN] + "\" added", true);
            
        }
        
        private void addMSEvent(){
            
            bool[Defs.SOLUTIONS] = true;
            
            String ms;
            
            /* Get the values */
            ms = maxSolText.getText();
            
            /* Check the values */
            try{
                if (Integer.parseInt(ms) <= 0){
                    msArea.writeln("Please select proper Maximum Sulutions value.");
                    return;
                }
                values[Defs.SOLUTIONS] = Integer.parseInt(ms);
            }catch(NumberFormatException nfe){
                domainSizeText.setHintText("");
                msArea.writeln("Not a number: " + nfe, true);
                return;
            }
            
            msArea.writeln("Constraint \"--ensembles " +
                    + values[Defs.SOLUTIONS] + "\" added", true);
        }
        
        
        private void addTSEvent(){
            
            bool[Defs.TIMEOUT_SEARCH] = true;
            
            String ts;
            
            /* Get the values */
            ts = timeOutSearchText.getText();
            
            /* Check the values */
            try{
                if (Integer.parseInt(ts) <= 0){
                    msArea.writeln("Please select proper Timeout Search value.");
                    return;
                }
                values[Defs.TIMEOUT_SEARCH] = Integer.parseInt(ts);
            }catch(NumberFormatException nfe){
                domainSizeText.setHintText("");
                msArea.writeln("Not a number: " + nfe, true);
                return;
            }
            
            msArea.writeln("Constraint \"--timeout-search " +
                    + values[Defs.TIMEOUT_SEARCH] + "\" added", true);
            
        }
        
        private void addTTEvent(){
            
            bool[Defs.TIMEOUT_TOTAL] = true;
            
            String tt;
            
            /* Get the values */
            tt = timeOutTotalText.getText();
            
            /* Check the values */
            try{
                if (Integer.parseInt(tt) <= 0){
                    msArea.writeln("Please select proper Timeout Total value.");
                    return;
                }
                values[Defs.TIMEOUT_TOTAL] = Integer.parseInt(tt);
            }catch(NumberFormatException nfe){
                domainSizeText.setHintText("");
                msArea.writeln("Not a number: " + nfe, true);
                return;
            }
            
            msArea.writeln("Constraint \"--timeout-total " +
                    + values[Defs.TIMEOUT_TOTAL] + "\" added", true);
            
        }
        
        public void setConstraints(){
            addDSEvent();
            addMSEvent();
            addTSEvent();
            addTTEvent();
        }
		
	public void cancelEvent(){
            
            jmList.clear();
            spList.clear();
            ussList.clear();
            uniList.clear();
            ellList.clear();
            dleqList.clear();
            dgeqList.clear();
            
            for(int j=0;j<10;j++){
                values[j] = -1;
            }
            
            for(int j=0;j<10;j++){
                bool[j] = false;
            }
            
            msArea.writeln("All constraints deleted", true);
        }
        
        private void addDGEQEvent(){
            
            bool[Defs.DGEQ] = true;
            
            String aa1, aa2;
            String d;
            
            /* Get the values */
            aa1 = fdgeqText.getText();
            aa2 = sdgeqText.getText();
            d = dgeqText.getText();
            
            /* Check the values */
            try{
                firstdgeq = Integer.parseInt(aa1);
                seconddgeq = Integer.parseInt(aa2);
                dgeq = Integer.parseInt(d);
            }catch(NumberFormatException nfe){
                fdgeqText.setHintText("");
                sdgeqText.setHintText("");
                dgeqText.setHintText("");
                msArea.writeln("Not a number: " + nfe, true);
                return;
            }
            
            if(firstdgeq >= seconddgeq){
               fdgeqText.setHintText("");
               sdgeqText.setHintText("");
               dgeqText.setHintText("");
               msArea.writeln("Please, select proper DGEQ values", true);
               return;
            }
            
            dgeqList.add(firstdgeq);
            dgeqList.add(seconddgeq);
            dgeqList.add(dgeq);
            
            msArea.writeln("--Constraint \"--distance-geq " +
                    + firstdgeq + " " + seconddgeq + " "
                    + dgeq + "\" added", true);
            
            fdgeqText.setHintText("");
            sdgeqText.setHintText("");
            dgeqText.setHintText("");
            
        }
        
        private void addDLEQEvent(){
            
            bool[Defs.DLEQ] = true;
            
            String aa1, aa2;
            String d;
            
            /* Get the values */
            aa1 = fdleqText.getText();
            aa2 = sdleqText.getText();
            d = dleqText.getText();
            
            /* Check the values */
            try{
                firstdleq = Integer.parseInt(aa1);
                seconddleq = Integer.parseInt(aa2);
                dleq = Integer.parseInt(d);
            }catch(NumberFormatException nfe){
                fdleqText.setHintText("");
                sdleqText.setHintText("");
                dleqText.setHintText("");
                msArea.writeln("Not a number: " + nfe, true);
                return;
            }
            
            if(firstdleq >= seconddleq){
               fdleqText.setHintText("");
               sdleqText.setHintText("");
               dleqText.setHintText("");
               msArea.writeln("Please, select proper values", true);
               return; 
            }
            
            dleqList.add(firstdleq);
            dleqList.add(seconddleq);
            dleqList.add(dleq);
            
            msArea.writeln("Constraint \"--distance-leq " +
                    + firstdleq + " " + seconddleq + " "
                    + dleq + "\" added", true);
            
            fdleqText.setHintText("");
            sdleqText.setHintText("");
            dleqText.setHintText("");
            
        }
        
        private void addUniformEvent(){
            
            bool[Defs.UNIFORM] = true;
            
            String uniString;
            String voxel;
            
            /* Get the values */
            uniString = uniText.getText();
            voxel = uniVoxText.getText();
            
            try{
                StringTokenizer st = new StringTokenizer(uniString);
                while (st.hasMoreTokens()) {
                    uniList.add(Double.parseDouble(st.nextToken()));
                }
            }catch(NumberFormatException nfe){
                uniText.setHintText("");
                uniVoxText.setHintText("");
                msArea.writeln("Not a number: " + nfe, true);
                return;
            }
            
            uniList.add(-1.0);
            
            /* Check the values */
            try{
                uniList.add(Double.parseDouble(voxel));
            }catch(NumberFormatException nfe){
                uniText.setHintText("");
                uniVoxText.setHintText("");
                msArea.writeln("Not a number: " + nfe, true);
                return;
            }
            
            msArea.writeln("Constraint \"--uniform "
                    + uniString + " : voxel-side= "
                    + voxel + "\" added", true);
            
            uniText.setHintText("");
            uniVoxText.setHintText("");
            
        }
        
        public void addEllEvent(){
            
            bool[Defs.ELLIPSOID] = true;
            
            String ellString;
            String x1,y1,z1,x2,y2,z2;
            String sumRadii;
            
            /* Get the values */
            ellString = ellText.getText();
            x1 = x1Text.getText();
            y1 = y1Text.getText();
            z1 = z1Text.getText();
            x2 = x2Text.getText();
            y2 = y2Text.getText();
            z2 = z2Text.getText();
            sumRadii = sumRadiiText.getText();
            
            try{
                StringTokenizer st = new StringTokenizer(ellString);
                while (st.hasMoreTokens()) {
                    ellList.add(Double.parseDouble(st.nextToken()));
                }
            }catch(NumberFormatException nfe){
                uniText.setHintText("");
                uniVoxText.setHintText("");
                msArea.writeln("Not a number: " + nfe, true);
                return;
            }
            ellList.add(-1.0);
            
            /* Check the values */
            try{
                ellList.add(Double.parseDouble(x1));
                ellList.add(Double.parseDouble(y1));
                ellList.add(Double.parseDouble(z1));
                ellList.add(Double.parseDouble(x2));
                ellList.add(Double.parseDouble(y2));
                ellList.add(Double.parseDouble(z2));                
                ellList.add(Double.parseDouble(sumRadii));
            }catch(NumberFormatException nfe){
                ellText.setHintText("");
                x1Text.setHintText("");
                y1Text.setHintText("");
                z1Text.setHintText("");
                x2Text.setHintText("");
                y2Text.setHintText("");
                z2Text.setHintText("");
                sumRadiiText.setHintText("");
                msArea.writeln("Not a number: " + nfe, true);
                return;
            }
            
            msArea.writeln("Constraint \"--ellipsoid "
                    + ellString + " : f1= "
                    + x1 + " " + y1 + " " + z1 + " f2= "
                    + x2 + " " + y2 + " " + z2 + " f2= "
                    + " sum-radii " + sumRadii + "\" added", true);
            
            ellText.setHintText("");
            x1Text.setHintText("");
            y1Text.setHintText("");
            z1Text.setHintText("");
            x2Text.setHintText("");
            y2Text.setHintText("");
            z2Text.setHintText("");
            sumRadiiText.setHintText("");
        
        }
        
    }//ConstraintPanel
    
    public ArrayList<String> getChoice(){
        return choice;
    }
    
    public boolean[] getFilled(){
        return bool;
    }
    
    public int[] getValues(){
        return values;
    }
    
    public ArrayList<Integer> getDGEQList(){
        return dgeqList;
    }
    
    public ArrayList<Integer> getDLEQList(){
        return dleqList;
    }
    
    public ArrayList<Double> getUniformList(){
        return uniList;
    }
    
    public ArrayList<Double> getEllList(){
        return ellList;
    }
    
    public ArrayList<Double> getSpList(){
        return spList;
    }
    
    public ArrayList<Integer> getJmList(){
        return jmList;
    }
    
    public ArrayList<Integer> getUSSList(){
        return ussList;
    }
    
    public void cancel(){
        setConstraintPanel.cancelEvent();
    }
    
}//SelectConstraintPanel
