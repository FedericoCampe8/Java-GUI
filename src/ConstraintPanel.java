package jafatt;

import java.awt.BorderLayout;
import java.awt.Container;
import java.awt.Dimension;
import java.awt.Toolkit;
import java.awt.event.ActionEvent;
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
import javax.swing.JTextField;
import javax.swing.SpringLayout;
import javax.swing.JComboBox;
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
    //private Structure structure;
    
    private JPanel inPanel, inSubPanel, exitPanel;
    private JPanel optionPanel;
    private JScrollPane scroll;
    
    private JButton ok, cancel, info, expand;
    
    private boolean expanded = false;
    
    private ConstraintInfoPanel cip;
    
    Toolkit t = Toolkit.getDefaultToolkit();
    Dimension screensize = t.getScreenSize();
    
    double widthFrame = (screensize.getWidth() * 50.0) / 100.0;  //960
    double heighFrame = (screensize.getHeight() * 51.0) / 100.0;  //540
    
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
        
        this.fragments = fragments;
        //this.structure = structure;
        
        Chain chain = structure.getChain(0);
        List groups = chain.getAtomGroups("amino");
        AminoAcid firstAA = (AminoAcid) groups.get(0);
        AminoAcid lastAA = (AminoAcid) groups.get(groups.size() - 1);
        firstPosition = Utilities.getAAPosition(firstAA);
        lastPosition = Utilities.getAAPosition(lastAA);
        
        inPanel = new JPanel();
        inSubPanel = new JPanel();
        optionPanel = new JPanel();
        exitPanel = new JPanel();
        
        /* Setup layout */
        setLocation((int)(view.getX() + (int)(view.getWidth()/4)),
                    (int)(view.getY() + (int)(view.getHeight()/4)));
        setPreferredSize(new Dimension((int)widthFrame/2, (int)heighFrame-250));
        setMinimumSize(new Dimension((int)widthFrame/2, (int)heighFrame-250));
        //setResizable(false);
        
        inPanel.setLayout(new BorderLayout());
        inSubPanel.setLayout(new BorderLayout());
        
        bool = new boolean[10];
        Arrays.fill(bool, Boolean.FALSE);
        values = new int[10];
        
        /* Internal panel */
        setConstraintPanel = new SetConstraintPanel();
        
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
        
        ok = new OpButton("Ok", "Set all the constraint and exit") {
            @Override
            public void buttonEvent(ActionEvent evt){
                setConstraintPanel.setConstraints();
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
            }
        };
        expand.setPreferredSize(new Dimension(140, 15));
        expand.setFocusPainted(false);
        //expand.setBorderPainted(false);
        
        exitPanel.add(ok);
        exitPanel.add(cancel);
        exitPanel.add(info);
        
        optionPanel.setLayout(new BorderLayout());
        
        optionPanel.add(expand, BorderLayout.NORTH);
        optionPanel.add(exitPanel, BorderLayout.CENTER);
		
        inSubPanel.add(setConstraintPanel, BorderLayout.CENTER);
        inSubPanel.add(optionPanel, BorderLayout.SOUTH);
        
        /* Add panels */
        inPanel.add(inSubPanel, BorderLayout.CENTER);
        inPanel.add(scroll, BorderLayout.SOUTH);
        /* Print some infos */
        msArea.writeln("Please set the constraint to the model.", false);
        msArea.writeln("Add a constraint to the textfield and then press the related button.", false);
        msArea.writeln("If a textfield is not filled, the constraint will not be added.", false);
    }//setup
    
    @Override
    public void run() {
	pack();
        Container ct = getContentPane();
        ct.add(inPanel);
        setVisible(true);
    }//run
    
    public void infoEvent(){
        cip = new ConstraintInfoPanel();
                
        /* Create the thread */
        Thread threadSelectFragments;
        threadSelectFragments = new Thread(cip);
        threadSelectFragments.start();
    }
    
    public void expandEvent(){
        if(expanded){
            setMinimumSize(new Dimension((int)widthFrame/2, (int)heighFrame-250));
            this.setSize(new Dimension((int)widthFrame/2, (int)heighFrame-250));
            expanded = false;
            expand.setIcon(iconExpand);
        }else{
            setMinimumSize(new Dimension((int)widthFrame, (int)heighFrame));
            this.setSize(new Dimension((int)widthFrame, (int)heighFrame));
            expanded = true;
            expand.setIcon(iconReduce);
        }
    }
    
    public class SetConstraintPanel extends JPanel{
        
        /* Components */
        JPanel setPanel;
        JPanel distancePanel;
        JPanel uniPanel;
        JPanel ellPanel;
        JPanel jmPanel, ussPanel;
        
        JLabel dsLabel, msLabel;
        JLabel tsLabel, ttLabel;
        JLabel jmLabel, clustLabel;
        JLabel spLabel, ussLabel;
        JLabel voxLabel;
        JLabel dgeqLabel, dleqLabel;
        JLabel uniLabel, ellLabel;
        JLabel sumRadiiLabel;
        
        JTextField domainSizeText;
        JTextField maxSolText;
        JTextField timeOutSearchText;
        JTextField timeOutTotalText;
        JTextField fdgeqText, sdgeqText, dgeqText;
        JTextField fdleqText, sdleqText, dleqText;
        JTextField uniText, uniVoxText;
        JTextField ellText, sumRadiiText;
        JTextField jm1Text, jm2Text;
        JTextField clust1Text, clust2Text;
        JTextField sp1Text, sp2Text;
        JTextField uss1Text, uss2Text;
        JTextField voxText;
        JTextField x1Text, y1Text, z1Text;
        JTextField x2Text, y2Text, z2Text;
        
        //JButton addDS, addMS, addTS, addTT;
        JButton addJM, addUSS;
        JButton addDGEQ, addDLEQ;
        JButton addUniform, addEllipsoid;
        
        JComboBox arrows;
        
        GridBagConstraints c = new GridBagConstraints();

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
            
            jmList = new ArrayList<Integer>();
            ussList = new ArrayList<Integer>();
            spList = new ArrayList<Double>();
            dgeqList = new ArrayList<Integer>();
            dleqList = new ArrayList<Integer>();
            uniList = new ArrayList<Double>();
            ellList = new ArrayList<Double>();
            choice = new ArrayList<String>();
            
            
            /*Add domain size Button
            
            
            addDS = new OpButton("Add DS", "Click to set a domain size constraint") {
                @Override
                public void buttonEvent(ActionEvent evt){
                    addDSEvent();
                }
            };
            
            /*Add maximum solution Button
            
            addMS = new OpButton("Add MS", "Click to set a maximum solution constraint") {
                @Override
                public void buttonEvent(ActionEvent evt){
                    addMSEvent();
                }
            };
            
            /*Add timeout search Button
            
            addTS = new OpButton("Add TS", "Click to set a timeout search constraint") {
                @Override
                public void buttonEvent(ActionEvent evt){
                    addTSEvent();
                }
            };
            
            /*Add total timeout Button
            
            addTT = new OpButton("Add TT", "Click to set a total timeout") {
                @Override
                public void buttonEvent(ActionEvent evt){
                    addTTEvent();
                }
            };

             */
            
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
            
             /* Text fields */
            domainSizeText = new JTextField(10);
            domainSizeText.setText("10");
            domainSizeText.setEnabled(true);
            
            maxSolText = new JTextField(10);
            maxSolText.setText("1000000");
            maxSolText.setEnabled(true);
            
            timeOutSearchText = new JTextField(10);
            timeOutSearchText.setText("60");
            timeOutSearchText.setEnabled(true);
            
            timeOutTotalText = new JTextField(10);
            timeOutTotalText.setText("120");
            timeOutTotalText.setEnabled(true);
            
            fdgeqText = new JTextField(5);
            fdgeqText.setEnabled(true);
            sdgeqText = new JTextField(5);
            sdgeqText.setEnabled(true);
            dgeqText = new JTextField(5);
            dgeqText.setEnabled(true);
            
            fdleqText = new JTextField(5);
            fdleqText.setEnabled(true);
            sdleqText = new JTextField(5);
            sdleqText.setEnabled(true);
            dleqText = new JTextField(5);
            dleqText.setEnabled(true);
            
            uniText = new JTextField(15);
            uniText.setEnabled(true);
            uniText.setPreferredSize(addUniform.getPreferredSize());
            uniVoxText = new JTextField(5);
            uniVoxText.setEnabled(true);
            uniVoxText.setPreferredSize(addUniform.getPreferredSize());
            
            ellText = new JTextField(15);
            ellText.setEnabled(true);
            ellText.setPreferredSize(addEllipsoid.getPreferredSize());
            x1Text= new JTextField(3);
            x1Text.setEnabled(true);
            x1Text.setPreferredSize(addEllipsoid.getPreferredSize());
            y1Text= new JTextField(3);
            y1Text.setEnabled(true);
            y1Text.setPreferredSize(addEllipsoid.getPreferredSize());
            z1Text= new JTextField(3);
            z1Text.setEnabled(true);
            z1Text.setPreferredSize(addEllipsoid.getPreferredSize());
            x2Text= new JTextField(3);
            x2Text.setEnabled(true);
            x2Text.setPreferredSize(addEllipsoid.getPreferredSize());
            y2Text= new JTextField(3);
            y2Text.setEnabled(true);
            y2Text.setPreferredSize(addEllipsoid.getPreferredSize());
            z2Text= new JTextField(3);
            z2Text.setEnabled(true);
            z2Text.setPreferredSize(addEllipsoid.getPreferredSize());
            sumRadiiText= new JTextField(3);
            sumRadiiText.setEnabled(true);            
            sumRadiiText.setPreferredSize(addEllipsoid.getPreferredSize());
            
            jm1Text = new JTextField(5);
            jm1Text.setEnabled(true);
            jm1Text.setPreferredSize(addJM.getPreferredSize());
            jm2Text = new JTextField(5);
            jm2Text.setEnabled(true);
            jm2Text.setPreferredSize(addJM.getPreferredSize());
            sp1Text = new JTextField(5);
            sp1Text.setEnabled(true);
            sp1Text.setPreferredSize(addJM.getPreferredSize());
            sp2Text = new JTextField(5);
            sp2Text.setEnabled(true);
            sp2Text.setPreferredSize(addJM.getPreferredSize());
            clust1Text = new JTextField(5);
            clust1Text.setEnabled(true);
            clust1Text.setPreferredSize(addJM.getPreferredSize());
            clust2Text = new JTextField(5);
            clust2Text.setEnabled(true);
            clust2Text.setPreferredSize(addJM.getPreferredSize());
            
            uss1Text = new JTextField(5);
            uss1Text.setEnabled(true);
            uss1Text.setPreferredSize(addUSS.getPreferredSize());
            uss2Text = new JTextField(5);
            uss2Text.setEnabled(true);
            uss2Text.setPreferredSize(addUSS.getPreferredSize());
            voxText = new JTextField(5);
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
            
            uniPanel.add(uniLabel = new JLabel( " Uniform: ", JLabel.TRAILING));
            uniLabel.setLabelFor(uniText);
            uniPanel.add(uniText);
            uniPanel.add(new JLabel (" Voxel Side: "));
            uniPanel.add(uniVoxText);
            uniPanel.add(addUniform);
            
            ellPanel.add(ellLabel = new JLabel( " Ellipsoid: ", JLabel.TRAILING));
            ellLabel.setLabelFor(ellText);
            ellPanel.add(ellText);
            ellPanel.add(new JLabel (" f1: "));
            ellPanel.add(x1Text);
            ellPanel.add(y1Text);
            ellPanel.add(z1Text);
            ellPanel.add(new JLabel (" f2: "));
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
            
            //setLayout(new FlowLayout(FlowLayout.CENTER));
            //setLayout(new BoxLayout(this, BoxLayout.Y_AXIS));
            //setLayout(new GridLayout(6,0));
            setLayout(new GridBagLayout());
            
            update(false);
            
            /*
            add(setPanel, c);
            //c.fill = GridBagConstraints.VERTICAL;
            //c.gridx = 0;
            c.gridy = 1;
            /*add(distancePanel, c);
            //c.fill = GridBagConstraints.VERTICAL;
            //c.gridx = 0;
            c.gridy = 2;
            add(uniPanel, c);
            //c.fill = GridBagConstraints.VERTICAL;
            //c.gridx = 0;
            c.gridy = 3;
            add(ellPanel, c);
            //c.fill = GridBagConstraints.VERTICAL;
            //c.gridx = 0;
            c.gridy = 4;
            add(jmPanel, c);
            //c.fill = GridBagConstraints.VERTICAL;
            //c.gridx = 0;
            c.gridy = 5;
            add(ussPanel, c);*/
        }//setup
        
        public void update(Boolean expanded){
            
            c.fill = GridBagConstraints.VERTICAL;
            c.gridx = 1;
            c.gridy = 0;
            
            add(setPanel, c);
            c.gridy = 1;
        
            if(expanded){
                add(distancePanel, c);
                c.gridy = 2;
                add(uniPanel, c);
                c.gridy = 3;
                add(ellPanel, c);
                c.gridy = 4;
                add(jmPanel, c);
                c.gridy = 5;
                add(ussPanel, c);                
            }else{
                remove(distancePanel);
                remove(uniPanel);
                remove(ellPanel);
                remove(jmPanel);
                remove(ussPanel);
            }
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
                jm1Text.setText("");
                jm2Text.setText("");
                clust1Text.setText("");
                clust2Text.setText("");
                sp1Text.setText("");
                sp2Text.setText("");
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
               jm1Text.setText("");
               jm2Text.setText("");
               msArea.writeln("Please, select proper JM values", true);
               return; 
            }
            
            jm = secondjm;           
            
            
            /*arrows.addActionListener(new ActionListener(){
                @Override
                public void actionPerformed(ActionEvent evt) {
                    choice.add((String)arrows.getSelectedItem());
                }  
            });*/
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
            
            jm1Text.setText("");
            jm2Text.setText("");
            clust1Text.setText("");
            clust2Text.setText("");
            sp1Text.setText("");
            sp2Text.setText("");
            
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
                uss1Text.setText("");
                uss2Text.setText("");
                voxText.setText("");
                msArea.writeln("Not a number: " + nfe, true);
                return;
            }
            
            if(firstuss >= seconduss){
               uss1Text.setText("");
               uss2Text.setText("");
               msArea.writeln("Please, select proper USS values", true);
               return; 
            }
            
            ussList.add(firstuss);
            ussList.add(seconduss);
            ussList.add(voxelS);
            
            msArea.writeln("Constraint \"--unique-source-sinks " + firstuss
                    + " '->' " + " " + seconduss + " : voxel-side "
                    + voxelS + "\" added", true);
            
            uss1Text.setText("");
            uss2Text.setText("");
            voxText.setText("");
            
            
        }
        
        private void addDSEvent(){
            
            bool[Defs.Domain_Size] = true;
            
            String ds;
            
            /* Get the values */
            ds = domainSizeText.getText();
            
            /* Check the values */
            try{
                if (Integer.parseInt(ds) <= 0){
                    msArea.writeln("Please select proper Domain Size value.");
                    return;
                }
                values[Defs.Domain_Size] = Integer.parseInt(ds);
            }catch(NumberFormatException nfe){
                domainSizeText.setText("");
                msArea.writeln("Not a number: " + nfe, true);
                return;
            }
            
            msArea.writeln("Constraint \"--domain-size " +
                    + values[Defs.Domain_Size] + "\" added", true);
            
        }
        
        private void addMSEvent(){
            
            bool[Defs.Maximum_Solutions] = true;
            
            String ms;
            
            /* Get the values */
            ms = maxSolText.getText();
            
            /* Check the values */
            try{
                if (Integer.parseInt(ms) <= 0){
                    msArea.writeln("Please select proper Maximum Sulutions value.");
                    return;
                }
                values[Defs.Maximum_Solutions] = Integer.parseInt(ms);
            }catch(NumberFormatException nfe){
                domainSizeText.setText("");
                msArea.writeln("Not a number: " + nfe, true);
                return;
            }
            
            msArea.writeln("Constraint \"--ensembles " +
                    + values[Defs.Maximum_Solutions] + "\" added", true);
        }
        
        
        private void addTSEvent(){
            
            bool[Defs.Timeout_Search] = true;
            
            String ts;
            
            /* Get the values */
            ts = timeOutSearchText.getText();
            
            /* Check the values */
            try{
                if (Integer.parseInt(ts) <= 0){
                    msArea.writeln("Please select proper Timeout Search value.");
                    return;
                }
                values[Defs.Timeout_Search] = Integer.parseInt(ts);
            }catch(NumberFormatException nfe){
                domainSizeText.setText("");
                msArea.writeln("Not a number: " + nfe, true);
                return;
            }
            
            msArea.writeln("Constraint \"--timeout-search " +
                    + values[Defs.Timeout_Search] + "\" added", true);
            
        }
        
        private void addTTEvent(){
            
            bool[Defs.Timeout_Total] = true;
            
            String tt;
            
            /* Get the values */
            tt = timeOutTotalText.getText();
            
            /* Check the values */
            try{
                if (Integer.parseInt(tt) <= 0){
                    msArea.writeln("Please select proper Timeout Total value.");
                    return;
                }
                values[Defs.Timeout_Total] = Integer.parseInt(tt);
            }catch(NumberFormatException nfe){
                domainSizeText.setText("");
                msArea.writeln("Not a number: " + nfe, true);
                return;
            }
            
            msArea.writeln("Constraint \"--timeout-total " +
                    + values[Defs.Timeout_Total] + "\" added", true);
            
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
                fdgeqText.setText("");
                sdgeqText.setText("");
                dgeqText.setText("");
                msArea.writeln("Not a number: " + nfe, true);
                return;
            }
            
            if(firstdgeq >= seconddgeq){
               fdgeqText.setText("");
               sdgeqText.setText("");
               dgeqText.setText("");
               msArea.writeln("Please, select proper DGEQ values", true);
               return;
            }
            
            dgeqList.add(firstdgeq);
            dgeqList.add(seconddgeq);
            dgeqList.add(dgeq);
            
            msArea.writeln("--Constraint \"--distance-geq " +
                    + firstdgeq + " " + seconddgeq + " "
                    + dgeq + "\" added", true);
            
            fdgeqText.setText("");
            sdgeqText.setText("");
            dgeqText.setText("");
            
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
                fdleqText.setText("");
                sdleqText.setText("");
                dleqText.setText("");
                msArea.writeln("Not a number: " + nfe, true);
                return;
            }
            
            if(firstdleq >= seconddleq){
               fdleqText.setText("");
               sdleqText.setText("");
               dleqText.setText("");
               msArea.writeln("Please, select proper values", true);
               return; 
            }
            
            dleqList.add(firstdleq);
            dleqList.add(seconddleq);
            dleqList.add(dleq);
            
            msArea.writeln("Constraint \"--distance-leq " +
                    + firstdleq + " " + seconddleq + " "
                    + dleq + "\" added", true);
            
            fdleqText.setText("");
            sdleqText.setText("");
            dleqText.setText("");
            
        }
        
        private void addUniformEvent(){
            
            bool[Defs.Uniform] = true;
            
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
                uniText.setText("");
                uniVoxText.setText("");
                msArea.writeln("Not a number: " + nfe, true);
                return;
            }
            
            uniList.add(-1.0);
            
            /* Check the values */
            try{
                uniList.add(Double.parseDouble(voxel));
            }catch(NumberFormatException nfe){
                uniText.setText("");
                uniVoxText.setText("");
                msArea.writeln("Not a number: " + nfe, true);
                return;
            }
            
            msArea.writeln("Constraint \"--uniform "
                    + uniString + " : voxel-side= "
                    + voxel + "\" added", true);
            
            uniText.setText("");
            uniVoxText.setText("");
            
        }
        
        public void addEllEvent(){
            
            bool[Defs.Ellipsoid] = true;
            
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
                uniText.setText("");
                uniVoxText.setText("");
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
                ellText.setText("");
                x1Text.setText("");
                y1Text.setText("");
                z1Text.setText("");
                x2Text.setText("");
                y2Text.setText("");
                z2Text.setText("");
                sumRadiiText.setText("");
                msArea.writeln("Not a number: " + nfe, true);
                return;
            }
            
            msArea.writeln("Constraint \"--ellipsoid "
                    + ellString + " : f1= "
                    + x1 + " " + y1 + " " + z1 + " f2= "
                    + x2 + " " + y2 + " " + z2 + " f2= "
                    + " sum-radii " + sumRadii + "\" added", true);
            
            ellText.setText("");
            x1Text.setText("");
            y1Text.setText("");
            z1Text.setText("");
            x2Text.setText("");
            y2Text.setText("");
            z2Text.setText("");
            sumRadiiText.setText("");
        
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
