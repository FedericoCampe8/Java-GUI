package jafatt;

import java.awt.BorderLayout;
import java.awt.FlowLayout;
import java.awt.GridLayout;
import java.awt.Container;
import java.awt.Dimension;
import java.awt.Toolkit;
import java.awt.event.ActionListener;
import java.awt.event.ActionEvent;
import javax.swing.JButton;
import javax.swing.JComboBox;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.JTextField;
import javax.swing.JCheckBox;
import javax.swing.SpringLayout;
import javax.swing.ImageIcon;
import javax.swing.BoxLayout;
import java.io.BufferedWriter;
import java.io.PrintWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.io.FileOutputStream;

public class OptionsPanel extends JFrame implements Runnable {

    private SetConstraintPanel setConstraintPanel;
    private AdvancedPanel advancedPanel;
    private UserFrame view;
    private String targetPath;
    private JPanel inPanel, inSubPanel;
    private JPanel optionPanel, exitPanel;
    private JScrollPane scroll;
    private OpButton ok, cancel, expand;
    
    private String currentDir = System.getProperty("user.dir");
    private String separator = Utilities.getSeparator();
    private String imSuffix = separator + "images" + separator;
    private String expandImage = currentDir + imSuffix + "expand.png";
    private String reduceImage = currentDir + imSuffix + "reduce.png";
    
    ImageIcon iconExpand = new ImageIcon(expandImage);
    ImageIcon iconReduce = new ImageIcon(reduceImage);
    
    private boolean advanced = false;
    
    double widthFrame, heighFrame;

    public OptionsPanel(UserFrame view, String targetPath) {
        super("Constraint Panel");
        setup(view);
        this.view = view;
        this.targetPath = targetPath;
    }

    /* Setup */
    private void setup(UserFrame view) {
        Toolkit t = Toolkit.getDefaultToolkit();
        Dimension screensize = t.getScreenSize();

        widthFrame = (screensize.getWidth() * 30.0) / 100.0;  //960
        heighFrame = (screensize.getHeight() * 27.0) / 100.0;  //540  

        inPanel = new JPanel();
        inSubPanel = new JPanel();
        optionPanel = new JPanel();
        exitPanel = new JPanel();

        /* Setup layout */
        setLocation((int) (view.getX() + (int) (view.getWidth() / 4)),
                (int) (view.getY() + (int) (view.getHeight() / 4)));
        setPreferredSize(new Dimension((int) widthFrame, (int) heighFrame));
        setMinimumSize(new Dimension((int) widthFrame, (int) heighFrame));
        //setResizable(false);

        inPanel.setLayout(new BorderLayout());
        inSubPanel.setLayout(new BorderLayout());

        /* Internal panel */
        setConstraintPanel = new SetConstraintPanel();
        advancedPanel = new AdvancedPanel();

        /* Lower Panel */

        ok = new OpButton("Run", "") {

            @Override
            public void buttonEvent(ActionEvent evt) {
                int[] options = setConstraintPanel.runEvent();
                setVisible(false);
                runSolver(options);
                dispose();
            }
        };

        cancel = new OpButton("Cancel", "") {

            @Override
            public void buttonEvent(ActionEvent evt) {
                setVisible(false);
                dispose();
            }
        };
        
        expand = new OpButton(iconExpand, "Show more options"){
            @Override
            public void buttonEvent(ActionEvent evt){
                //expandEvent();
                //setConstraintPanel.update(expanded);             
            }
        };
        expand.setPreferredSize(new Dimension(140, 15));
        expand.setFocusPainted(false);
        
        exitPanel.add(ok);
        exitPanel.add(cancel);
        
        optionPanel.setLayout(new BorderLayout());
        optionPanel.add(expand, BorderLayout.NORTH);
        optionPanel.add(exitPanel, BorderLayout.CENTER); 
        
        inSubPanel.add(setConstraintPanel, BorderLayout.CENTER);

        /* Add panels */
        inPanel.add(inSubPanel, BorderLayout.CENTER);
        inPanel.add(optionPanel, BorderLayout.SOUTH);
        /* Print some infos */
    }//setup

    @Override
    public void run() {
        pack();
        Container ct = getContentPane();
        ct.add(inPanel);
        setVisible(true);
    }//run

    public void runSolver(int[] options) {
        
        if(advanced){
            
            //Delete any out.pdb file existing
            try {
                FileOutputStream cocosFile = new FileOutputStream(Defs.path_prot 
                        + HeaderPdb.getProteinId() + ".out.pdb");
                cocosFile.close();
                
                PrintWriter cocos = new PrintWriter(new BufferedWriter(
                        new FileWriter(Defs.path_prot + HeaderPdb.getProteinId()
                        + ".in.cocos", true)));
                cocos.print("TARGET_PROT " + Defs.path_prot + HeaderPdb.getProteinId());
                cocos.print("\n");
                cocos.print("KNOWN_PROT " + Defs.path_prot + HeaderPdb.getProteinId());
                cocos.print("\n");
                cocos.print(advancedPanel.getCocosFile());
                cocos.close();
            } catch (IOException e) {
                //print exception
            }
        }
        
        CocosSolver cocos = new CocosSolver(view, options, targetPath, advanced);
        System.out.println(targetPath);
        Thread solver = new Thread(cocos);
        solver.start();

    }    
    
    public void expandEvent(){
        if(advanced){
            inSubPanel.remove(advancedPanel);
            revalidate();
            repaint();
            setMinimumSize(new Dimension((int)widthFrame, (int)heighFrame));
            setSize(new Dimension((int)widthFrame, (int)heighFrame));
            advanced = false;
            expand.setIcon(iconExpand);
        }else{
            inSubPanel.add(advancedPanel, BorderLayout.SOUTH);
            revalidate();
            repaint();
            setMinimumSize(new Dimension((int)widthFrame, (int)heighFrame + 50));
            setSize(new Dimension((int)widthFrame, (int)heighFrame + 50));
            advanced = true;
            expand.setIcon(iconReduce);
        }
    }

    private class SetConstraintPanel extends JPanel {

        /* Components */
        JPanel setPanel, cboxPanel;
        JPanel advancedPanel;
        JLabel montecarloLabel, gibbsLabel;
        JLabel inputFileLabel, rmsdLabel, verboseLabel;
        JTextField montecarloText, gibbsText;
        JCheckBox input, rmsd, verbose, gibbs, cgC;

        /* Values */
        public SetConstraintPanel() {
            setup();
        }

        /* Setup */
        private void setup() {

            setPanel = new JPanel(new SpringLayout());
            cboxPanel = new JPanel(new GridLayout(5, 1));
            advancedPanel = new JPanel();

            /* Text fields */
            montecarloText = new JTextField(10);
            montecarloText.setEnabled(true);
            montecarloText.setText("10");

            gibbsText = new JTextField(10);
            gibbsText.setEnabled(true);

            input = new JCheckBox("Create input file for cocos from FASTA", true);
            rmsd = new JCheckBox("Set RMSD as objective function", false);
            verbose = new JCheckBox("Print verbose info during computation", true);
            gibbs = new JCheckBox("Gibbs sampling algorithm (default: MonteCarlo)", false);
            cgC = new JCheckBox("Enable CG constraint", false);


            setPanel.add(montecarloLabel = new JLabel(" Timeout Montecarlo sampling (sec.): ", JLabel.TRAILING));
            montecarloLabel.setLabelFor(montecarloText);
            setPanel.add(montecarloText);

            setPanel.add(gibbsLabel = new JLabel(" Samples for Gibbs sampling (default 10): ", JLabel.TRAILING));
            gibbsLabel.setLabelFor(gibbsText);
            setPanel.add(gibbsText);

            SpringUtilities.makeCompactGrid(setPanel, 2, 2, 10, 10, 10, 10);

            cboxPanel.add(input);
            cboxPanel.add(rmsd);
            cboxPanel.add(verbose);
            cboxPanel.add(gibbs);
            cboxPanel.add(cgC);
            
            //setLayout(new BorderLayout());

            add(setPanel);
            add(cboxPanel);

        }//setup
        
        private int[] runEvent() {

            String[] options = {"", ""};

            int[] values = {0, 0, 0, 0, 0, 0, 0};

            /* Get the values */
            options[0] = montecarloText.getText();
            options[1] = gibbsText.getText();

            /* Check the values */
            try {
                if (!options[0].equals("")) {
                    if (Integer.parseInt(options[0]) > 0) {
                        values[0] = Integer.parseInt(options[0]);
                    }
                }
                if (!options[1].equals("")) {
                    if (Integer.parseInt(options[1]) > 0) {
                        values[1] = Integer.parseInt(options[1]);
                    }
                }
            } catch (NumberFormatException nfe) {
                montecarloText.setText("");
            }

            values[2] = input.isSelected() ? 1 : 0;
            values[3] = rmsd.isSelected() ? 1 : 0;
            values[4] = verbose.isSelected() ? 1 : 0;
            values[5] = gibbs.isSelected() ? 1 : 0;
            values[6] = cgC.isSelected() ? 1 : 0;

            return values;

        }
    }//ConstraintPanel

    private class AdvancedPanel extends JPanel {

        JTextField strStartText, strEndText, strPriorityText;
        JTextField coStartText, coEndText, coPriorityText;
        JTextField scopeStartText, scopeEndText;
        JComboBox structure, strSearch;
        JComboBox coordinator, coSearch;
        
        JPanel strPanel, coPanel;
        OpButton addButton;
        
        String[] strOption = {"", "H", "P", "S"};
        String[] coOption = {"", "B", "C"};
        String[] searchOption = {"", "icm", "gibbs",
            "montecarlo", "complete"};
        
        String cocosFile = "";

        public AdvancedPanel() {
            setup();
        }

        /* Setup */
        private void setup() {
            
            strPanel = new JPanel();
            coPanel = new JPanel();
            
            addButton = new OpButton("Add", "Click to set the Constraint") {
                @Override
                public void buttonEvent(ActionEvent evt){
                    addEvent();
                }
            };
            
            strStartText = new JTextField(5);
            strStartText.setPreferredSize(
                    addButton.getPreferredSize());
            strEndText = new JTextField(5);
            strEndText.setPreferredSize(
                    addButton.getPreferredSize());
            strStartText.setEditable(false);
            strEndText.setEditable(false);
            
            structure = new JComboBox();
            structure.addItem("");
            structure.addItem("Alpha Elix");
            structure.addItem("Polyproline II");
            structure.addItem("Beta Sheets");
            structure.setSelectedIndex(0);
            structure.addActionListener(
                    new ActionListener(){
                        @Override
                        public void actionPerformed(ActionEvent e){
                            if (structure.getSelectedIndex() == 0) {
                                strStartText.setEditable(false);
                                strEndText.setEditable(false);
                            } else {
                                strStartText.setEditable(true);
                                strEndText.setEditable(true);
                            }
                        }
                    });
            
            strPriorityText = new JTextField(5);
            strPriorityText.setPreferredSize(
                    addButton.getPreferredSize());
            
            strSearch = new JComboBox();
            strSearch.addItem("");
            strSearch.addItem("Icm");
            strSearch.addItem("Gibbs");
            strSearch.addItem("Montecarlo");
            strSearch.addItem("Complete");
            strSearch.setSelectedIndex(0);
            
            coStartText = new JTextField(5);
            coStartText.setPreferredSize(
                    addButton.getPreferredSize());
            coEndText = new JTextField(5);
            coEndText.setPreferredSize(
                    addButton.getPreferredSize());
            coStartText.setEditable(false);
            coEndText.setEditable(false);
            
            coordinator = new JComboBox();
            coordinator.addItem("");
            coordinator.addItem("Beta Turn");
            coordinator.addItem("Coil");
            coordinator.setSelectedIndex(0);
            coordinator.addActionListener(
                    new ActionListener(){
                        @Override
                        public void actionPerformed(ActionEvent e){
                            if (coordinator.getSelectedIndex() == 0) {
                                coStartText.setEditable(false);
                                coEndText.setEditable(false);
                            } else {
                                coStartText.setEditable(true);
                                coEndText.setEditable(true);
                            }
                        }
                    });
            
            coPriorityText = new JTextField(5);
            coPriorityText.setPreferredSize(
                    addButton.getPreferredSize());
            
            scopeStartText = new JTextField(5);
            scopeStartText.setPreferredSize(
                    addButton.getPreferredSize());
            scopeEndText = new JTextField(5);
            scopeEndText.setPreferredSize(
                    addButton.getPreferredSize());
            
            coSearch = new JComboBox();
            coSearch.addItem("");
            coSearch.addItem("Icm");
            coSearch.addItem("Gibbs");
            coSearch.addItem("Montecarlo");
            coSearch.addItem("Complete");
            coSearch.setSelectedIndex(0);
            
            strPanel.setLayout (new FlowLayout());
            strPanel.add(new JLabel(" Structure: "));
            strPanel.add(structure);
            strPanel.add(strStartText);
            strPanel.add(strEndText);
            strPanel.add(new JLabel(" Priority: "));
            strPanel.add(strPriorityText);
            strPanel.add(strSearch);
            
            coPanel.setLayout (new FlowLayout());
            coPanel.add(new JLabel(" Coordinator: "));
            coPanel.add(coordinator);
            coPanel.add(coStartText);
            coPanel.add(coEndText);
            coPanel.add(new JLabel(" Scope: "));
            coPanel.add(scopeStartText);
            coPanel.add(scopeEndText);
            coPanel.add(new JLabel(" Priority: "));
            coPanel.add(coPriorityText);
            coPanel.add(coSearch);
            coPanel.add(addButton);
            
            setLayout(new GridLayout(2,1));
            add(strPanel);
            add(coPanel);
            //add(addButton);

        }
        
        public void addEvent (){            
            
            if(structure.getSelectedIndex() != 0){
                cocosFile = cocosFile + strOption[structure.getSelectedIndex()];
                try{
                    int start = Integer.parseInt(strStartText.getText());
                    int end = Integer.parseInt(strEndText.getText());
                    if((start < 0) || (end < 0)){
                        //print exception
                        return;
                    }
                    if(start < end) {
                        cocosFile = cocosFile + " " + start + " " + end + " ";
                    }else{
                    //print exception
                    }
                }catch(NumberFormatException nfe){
                    //print exception
                }
            }
            if (!strPriorityText.getText().equals("")){
                try{
                    int priority = Integer.parseInt(strPriorityText.getText());
                    if(priority < 0){
                        //print exception
                        return;
                    }
                    cocosFile = cocosFile + "p " + priority + " ";
                    
                }catch(NumberFormatException nfe){
                    //print exception
                }
            }
            
            if(strSearch.getSelectedIndex() != 0){
                cocosFile = cocosFile 
                        + searchOption[strSearch.getSelectedIndex()] + " ";
            }
            
            if(coordinator.getSelectedIndex() != 0){
                cocosFile = cocosFile + coOption[coordinator.getSelectedIndex()];
                try{
                    int start = Integer.parseInt(coStartText.getText());
                    int end = Integer.parseInt(coEndText.getText());
                    if((start < 0) || (end < 0)){
                        //print exception
                        return;
                    }
                    if(start < end) {
                        cocosFile = cocosFile + " " + start + " " + end + " ";
                    }else{
                    //print exception
                    }
                }catch(NumberFormatException nfe){
                    //print exception
                }
            }
            
            if ((!scopeStartText.getText().equals("")) &&
                    (!scopeEndText.getText().equals(""))){
                try{
                    int start = Integer.parseInt(scopeStartText.getText());
                    int end = Integer.parseInt(scopeEndText.getText());
                    if((start < 0) || (end < 0)){
                        //print exception
                        return;
                    }
                    if(start < end) {
                        cocosFile = cocosFile + "s[" + start + "," + end + "] ";
                    }else{
                    //print exception
                    }
                }catch(NumberFormatException nfe){
                    //print exception
                }
            }
            
            if (!coPriorityText.getText().equals("")){
                try{
                    int priority = Integer.parseInt(coPriorityText.getText());
                    if(priority < 0){
                        //print exception
                        return;
                    }
                    cocosFile = cocosFile + "p " + priority + " ";
                    
                }catch(NumberFormatException nfe){
                    //print exception
                }
            }
            
            if(coSearch.getSelectedIndex() != 0){
                cocosFile = cocosFile 
                        + searchOption[coSearch.getSelectedIndex()] + " ";
            }
            
            cocosFile = cocosFile + "\n";
            System.out.println(""+cocosFile);
            
        }
        
        public String getCocosFile(){
            return cocosFile;
        }
    }

}//SelectConstraintPanel