package jafatt;

import java.awt.BorderLayout;
import java.awt.FlowLayout;
import java.awt.Insets;
import java.awt.GridBagLayout;
import java.awt.GridBagConstraints;
import java.awt.Container;
import java.awt.Dimension;
import java.awt.Toolkit;
import java.awt.event.ActionListener;
import java.awt.event.ActionEvent;
import java.awt.event.AdjustmentEvent;
import java.awt.event.AdjustmentListener;
import javax.swing.JComboBox;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.ScrollPaneConstants;
import javax.swing.JCheckBox;
import javax.swing.SpringLayout;
import javax.swing.ImageIcon;
import javax.swing.BorderFactory;
import javax.swing.JRadioButton;
import javax.swing.ButtonGroup;

public class OptionsPanel extends JFrame implements Runnable {

    private SetConstraintPanel setConstraintPanel;
    private AdvancedPanel advancedPanel;
    private UserFrame view;
    private String targetPath;
    private JPanel inPanel, inSubPanel;
    private JPanel optionPanel, exitPanel;
    private JPanel mainPanel;
    private JScrollPane scroll, infoScroll;
    private OpButton ok, cancel;
    private MessageArea msArea;
    private boolean cuda;
    
    private String currentDir = System.getProperty("user.dir");
    private String separator = Utilities.getSeparator();
    private String imSuffix = separator + "images" + separator;
    private String expandImage = currentDir + imSuffix + "expand.png";
    private String reduceImage = currentDir + imSuffix + "reduce.png";
    
    ImageIcon iconExpand = new ImageIcon(expandImage);
    ImageIcon iconReduce = new ImageIcon(reduceImage);
    
    double widthFrame, heighFrame;

    public OptionsPanel(UserFrame view, String targetPath, boolean cuda) {
        super("Constraint Panel");
        setup(view);
        this.view = view;
        this.targetPath = targetPath;
        this.cuda = cuda;
    }

    /* Setup */
    private void setup(UserFrame view) {
        Toolkit t = Toolkit.getDefaultToolkit();
        Dimension screensize = t.getScreenSize();

        widthFrame = (screensize.getWidth() * 50.0) / 100.0;  //960
        heighFrame = (screensize.getHeight() * 30.0) / 100.0;  //540  

        inPanel = new JPanel();
        inSubPanel = new JPanel();
        optionPanel = new JPanel();
        exitPanel = new JPanel();
        mainPanel = new JPanel(new BorderLayout());

        /* Setup layout */
        //setLocation((int) (view.getX() + (int) (view.getWidth() / 4)),
        //        (int) (view.getY() + (int) (view.getHeight() / 4)));
        //setPreferredSize(new Dimension((int) widthFrame, (int) heighFrame));
        //setMinimumSize(new Dimension((int) widthFrame, (int) heighFrame));
        //setResizable(false);

        inPanel.setLayout(new BorderLayout());
        inSubPanel.setLayout(new BorderLayout());
        scroll = new JScrollPane(inSubPanel,
                ScrollPaneConstants.VERTICAL_SCROLLBAR_AS_NEEDED,
                ScrollPaneConstants.HORIZONTAL_SCROLLBAR_AS_NEEDED);
        
        msArea = new MessageArea(3, 10);
        
        /* Add the scroll bar and set auto-scroll */
        infoScroll = new JScrollPane(msArea);
        infoScroll.getVerticalScrollBar().addAdjustmentListener(new AdjustmentListener(){
            @Override
            public void adjustmentValueChanged(AdjustmentEvent e){
               msArea.select(msArea.getHeight() + 1000, 0);
            }
        });

        /* Internal panel */
        setConstraintPanel = new SetConstraintPanel();
        advancedPanel = new AdvancedPanel();

        /* Lower Panel */

        ok = new OpButton("Run", "") {

            @Override
            public void buttonEvent(ActionEvent evt) {
                int[] options = setConstraintPanel.runEvent();
                if(Defs.FASTA_OPTION == 0 && options[Defs.PDB_FILE] == 0)
                    return;
                setVisible(false);
                runSolver(options,cuda);
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
                
        exitPanel.add(ok);
        exitPanel.add(cancel);
        
        optionPanel.setLayout(new BorderLayout());
        //optionPanel.add(expand, BorderLayout.NORTH);
        optionPanel.add(exitPanel, BorderLayout.CENTER); 
        
        inSubPanel.add(setConstraintPanel, BorderLayout.CENTER);
        inSubPanel.setBorder(BorderFactory.createTitledBorder("Options"));
        inSubPanel.add(advancedPanel, BorderLayout.SOUTH);
        advancedPanel.setVisible(false);

        /* Add panels */
        inPanel.add(scroll, BorderLayout.CENTER);
        inPanel.add(optionPanel, BorderLayout.SOUTH);
        /* Print some infos */
        mainPanel.add(inPanel, BorderLayout.CENTER);
        mainPanel.add(infoScroll, BorderLayout.SOUTH);
        
        msArea.writeln("Set the constraints for Cocos.",false);
    }//setup

    @Override
    public void run() {
        Container ct = getContentPane();
        ct.add(mainPanel);
        update();
        setVisible(true);
    }//run
    
    private void update(){
        pack();
        setLocationRelativeTo(view);
    }

    public void runSolver(int[] options, boolean cuda) {
        
        CocosSolver cocos = new CocosSolver(view, options, targetPath, 
                advancedPanel.getCocosConstraints(), cuda);
        Thread solver = new Thread(cocos);
        solver.start();

    }    
    
    public void expandEvent(boolean advanced){
        if(!advanced){
            advancedPanel.setVisible(false);
            //setPreferredSize(new Dimension((int)widthFrame, (int)heighFrame));            
            advanced = false;
            update();
        }else{
            advancedPanel.setVisible(true);
            //setPreferredSize(new Dimension((int)widthFrame + 700, (int)heighFrame + 65));
            advanced = true;
            update();
        }
    }

    private class SetConstraintPanel extends JPanel {

        /* Components */
        JPanel setPanel, cboxPanel;
        JPanel advancedPanel, pdbPanel;
        JLabel montecarloLabel, gibbsLabel;
        JLabel inputFileLabel, rmsdLabel, verboseLabel;
        HintTextField montecarloText, gibbsText;
        JCheckBox rmsd, verbose, gibbs, cgC, download;
        JRadioButton inputFasta, inputCocos;
        ButtonGroup group;
        Downloader dPdb;
        
        GridBagConstraints c = new GridBagConstraints();

        /* Values */
        public SetConstraintPanel() {
            setup();
        }

        /* Setup */
        private void setup() {

            setPanel = new JPanel(new SpringLayout());
            cboxPanel = new JPanel(new GridBagLayout());
            advancedPanel = new JPanel();
            pdbPanel = new JPanel(new GridBagLayout());
            group = new ButtonGroup();

            /* Text fields */
            montecarloText = new HintTextField(" Sec ", 10);
            montecarloText.setEnabled(true);
            montecarloText.setHintText("10");

            gibbsText = new HintTextField(" Default 10 Sec ", 10);
            gibbsText.setEnabled(true);
            
            download = new JCheckBox("Download " + HeaderPdb.getProteinId() + ".pdb");
            download.setEnabled(false);
            
            dPdb = new Downloader();

            inputFasta = new JRadioButton("Create input file for Cocos from FASTA", true);
            inputFasta.addActionListener(
                    new ActionListener(){
                        @Override
                        public void actionPerformed(ActionEvent e){
                            if(inputFasta.isSelected()){
                                expandEvent(false);
                                download.setEnabled(false);
                            }
                            
                        }
                    });
            
            inputCocos = new JRadioButton("Create Cocos input File (Pdb Required)", false);
            inputCocos.addActionListener(
                    new ActionListener(){
                        @Override
                        public void actionPerformed(ActionEvent e){
                            if(inputCocos.isSelected())
                                expandEvent(true);
                                download.setEnabled(true);
                        }
                    });
            
            pdbPanel.setBorder(BorderFactory.createTitledBorder("Input File"));
            group.add(inputFasta);
            group.add(inputCocos);
            rmsd = new JCheckBox("Set RMSD as objective function", false);
            verbose = new JCheckBox("Print verbose info during computation", true);
            gibbs = new JCheckBox("Gibbs sampling algorithm (default: MonteCarlo)", false);
            gibbs.setEnabled(false);
            cgC = new JCheckBox("Enable CG constraint", false);
            cgC.setEnabled(false);


            setPanel.add(montecarloLabel = new JLabel(" Timeout Montecarlo sampling : ", JLabel.TRAILING));
            montecarloLabel.setLabelFor(montecarloText);
            setPanel.add(montecarloText);

            setPanel.add(gibbsLabel = new JLabel(" Samples for Gibbs sampling : ", JLabel.TRAILING));
            gibbsLabel.setLabelFor(gibbsText);
            setPanel.add(gibbsText);

            SpringUtilities.makeCompactGrid(setPanel, 2, 2, 10, 10, 10, 10);

            c.fill = GridBagConstraints.HORIZONTAL;
            c.gridy = 0;
            pdbPanel.add(inputFasta,c);
            c.gridy = 1;
            pdbPanel.add(inputCocos,c);
            c.gridy = 2;
            c.insets = new Insets(0,50,0,0);
            pdbPanel.add(download,c);
            c.insets = new Insets(0,0,0,0);
            
            
            c.gridy = 0;
            cboxPanel.add(pdbPanel,c);
            c.gridy = 1;
            cboxPanel.add(rmsd,c);
            c.gridy = 2;
            cboxPanel.add(verbose,c);
            c.gridy = 3;
            cboxPanel.add(gibbs,c);
            c.gridy = 4;
            cboxPanel.add(cgC,c);
            
            setLayout(new GridBagLayout());
            c.gridx = 0;
            add(setPanel,c);
            c.gridx = 1;
            add(cboxPanel,c);

        }//setup
        
        private int[] runEvent() {

            String[] options = {"", ""};

            int[] values = {0, 0, 0, 0, 0, 0, 0, 0};
            
            if(download.isSelected()){
                //int[] downloaded = Utilities.downloadProtein(HeaderPdb.getProteinId(), true);
                int[] downloaded = Downloader.downloadProtein(view,HeaderPdb.getProteinId(), true, false);
                if (downloaded[0] == 0){
                    msArea.writeln("Error downloading " +
                            HeaderPdb.getProteinId() + ".pdb " +
                            "from the rcbs.com protein database. " +
                            "Check your Internet Connection.");
                }
                values[Defs.PDB_FILE] = downloaded[1];
                
            }

            /* Get the values */
            options[0] = montecarloText.getText();
            options[1] = gibbsText.getText();

            /* Check the values */
            try {
                if (!options[0].equals("")) {
                    if (Integer.parseInt(options[0]) > 0) {
                        values[Defs.MONTECARLO_SAMPLING] = Integer.parseInt(options[0]);
                    }
                }
            }catch (NumberFormatException nfe) {
                montecarloText.setHintText("");
            }
            try{
                if (!options[1].equals("")) {
                    if (Integer.parseInt(options[1]) > 0) {
                        values[Defs.GIBBS_SAMPLING] = Integer.parseInt(options[1]);
                    }
                }
            } catch (NumberFormatException nfe) {
                gibbsText.setHintText("");
            }

            values[Defs.FASTA_OPTION] = inputFasta.isSelected() ? 1 : 0;
            values[Defs.RMSD_OPTION] = rmsd.isSelected() ? 1 : 0;
            values[Defs.VERBOSE_OPTION] = verbose.isSelected() ? 1 : 0;
            values[Defs.GIBBS_OPTION] = gibbs.isSelected() ? 1 : 0;
            values[Defs.CGC_OPTION] = cgC.isSelected() ? 1 : 0;

            return values;

        }
    }//ConstraintPanel

    private class AdvancedPanel extends JPanel {

        HintTextField strStartText, strEndText, strPriorityText;
        HintTextField coStartText, coEndText, coPriorityText;
        HintTextField scopeStartText, scopeEndText;
        JComboBox structure, strSearch;
        JComboBox coordinator, coSearch;
        
        GridBagConstraints c;
        
        JPanel strPanel, coPanel;
        OpButton addButton;
        
        String[] strOption = {"", "H", "P", "S"};
        String[] coOption = {"", "B", "C"};
        String[] searchOption = {"", "icm", "gibbs",
            "montecarlo", "complete"};
        
        String cocosConstraints = "";

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
            
            strStartText = new HintTextField(" Start AA ", 6);
            strStartText.setPreferredSize(
                    addButton.getPreferredSize());
            strEndText = new HintTextField(" End AA ", 6);
            strEndText.setPreferredSize(
                    addButton.getPreferredSize());
            strStartText.setEditable(false);
            strStartText.removeListener();
            strEndText.setEditable(false);
            strEndText.removeListener();
            
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
                                strStartText.removeListener();
                                strEndText.removeListener();
                            } else {
                                strStartText.setEditable(true);
                                strEndText.setEditable(true);
                                strStartText.addListener();
                                strEndText.addListener();
                            }
                        }
                    });
            
            strPriorityText = new HintTextField(" Int ", 5);
            strPriorityText.setPreferredSize(
                    addButton.getPreferredSize());
            
            strSearch = new JComboBox();
            strSearch.addItem("");
            strSearch.addItem("Icm");
            strSearch.addItem("Gibbs");
            strSearch.addItem("Montecarlo");
            strSearch.addItem("Complete");
            strSearch.setSelectedIndex(0);
            
            coStartText = new HintTextField(" Start AA ", 6);
            coStartText.setPreferredSize(
                    addButton.getPreferredSize());
            coEndText = new HintTextField(" End AA ", 6);
            coEndText.setPreferredSize(
                    addButton.getPreferredSize());
            coStartText.setEditable(false);
            coEndText.setEditable(false);
            coStartText.removeListener();
            coEndText.removeListener();
            
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
                                coStartText.removeListener();
                                coEndText.removeListener();
                                
                            } else {
                                coStartText.setEditable(true);
                                coEndText.setEditable(true);
                                coStartText.addListener();
                                coEndText.addListener();
                            }
                        }
                    });
            
            coPriorityText = new HintTextField(" Int ", 5);
            coPriorityText.setPreferredSize(
                    addButton.getPreferredSize());
            
            scopeStartText = new HintTextField(" Start AA ", 6);
            scopeStartText.setPreferredSize(
                    addButton.getPreferredSize());
            scopeEndText = new HintTextField(" End AA ", 6);
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
            //coPanel.add(addButton);
            
            setLayout(new GridBagLayout());
            c = new GridBagConstraints();
            c.fill = GridBagConstraints.HORIZONTAL;
            c.gridx = 0; c.gridy = 0;
            add(strPanel,c);
            c.gridy = 1;
            add(coPanel,c);
            c.gridy = 2;
            c.fill = GridBagConstraints.NONE;
            add(addButton,c);
            setBorder(BorderFactory.createTitledBorder("Advanced Options (Pdb Required)"));
            //SpringUtilities.makeCompactGrid(this, 2, 2, 10, 10, 10, 10);
            //add(addButton);

        }
        
        public void addEvent (){            
            
            String cocosLine = "";
            
            if(structure.getSelectedIndex() != 0){
                cocosLine = cocosLine + strOption[structure.getSelectedIndex()];
                try{
                    int start = Integer.parseInt(strStartText.getText());
                    int end = Integer.parseInt(strEndText.getText());
                    if((start < 0) || (end < 0)){
                        cocosLine = cocosLine + " >> " + start + " " + end + " << " + 
                                "negative numbers.";
                        msArea.writeln("Error " + cocosLine);
                        return;
                    }
                    if(start < end) {
                        cocosLine = cocosLine + " " + start + " " + end + " ";
                    }else{
                        cocosLine = cocosLine + " >> " + start + " " + end + " << .";
                        msArea.writeln("Error " + cocosLine);
                        return;
                    }
                }catch(NumberFormatException nfe){
                    cocosLine = cocosLine + " >> " + strStartText.getText() +
                            " " + strEndText.getText() + " << " + 
                                "not numbers.";
                    msArea.writeln("Error " + cocosLine);
                    return;
                }
            }
            if (!strPriorityText.getText().equals("")){
                try{
                    int priority = Integer.parseInt(strPriorityText.getText());
                    if(priority < 0){
                        cocosLine = cocosLine + " >> " + priority + " << " + 
                                "negative number.";
                        msArea.writeln("Error " + cocosLine);
                        return;
                    }
                    cocosLine = cocosLine + "p " + priority + " ";
                    
                }catch(NumberFormatException nfe){
                    cocosLine = cocosLine + " >> " + strPriorityText.getText() +
                             " << " + "not a number.";
                    msArea.writeln("Error " + cocosLine);
                    return;
                }
            }
            
            if(strSearch.getSelectedIndex() != 0){
                cocosLine = cocosLine 
                        + searchOption[strSearch.getSelectedIndex()] + " ";
            }
            
            if(coordinator.getSelectedIndex() != 0){
                cocosLine = cocosLine + coOption[coordinator.getSelectedIndex()];
                try{
                    int start = Integer.parseInt(coStartText.getText());
                    int end = Integer.parseInt(coEndText.getText());
                    if((start < 0) || (end < 0)){
                        cocosLine = cocosLine + " >> " + start + " " + end + " << " + 
                                "negative numbers.";
                        msArea.writeln("Error " + cocosLine);
                        return;
                    }
                    if(start < end) {
                        cocosLine = cocosLine + " " + start + " " + end + " ";
                    }else{
                        cocosLine = cocosLine + " >> " + start + " " + end + " << .";
                        msArea.writeln("Error " + cocosLine);
                        return;
                    }
                }catch(NumberFormatException nfe){
                    cocosLine = cocosLine + " >> " + coStartText.getText() +
                            " " + coEndText.getText() + " << " + 
                                "not numbers.";
                    msArea.writeln("Error " + cocosLine);
                    return;
                }
            }
            
            if ((!scopeStartText.getText().equals("")) &&
                    (!scopeEndText.getText().equals(""))){
                try{
                    int start = Integer.parseInt(scopeStartText.getText());
                    int end = Integer.parseInt(scopeEndText.getText());
                    if((start < 0) || (end < 0)){
                        cocosLine = cocosLine + " >> " + start + " " + end + " << " + 
                                "negative numbers.";
                        msArea.writeln("Error " + cocosLine);
                        return;
                    }
                    if(start < end) {
                        cocosLine = cocosLine + "s[" + start + "," + end + "] ";
                    }else{
                        cocosLine = cocosLine + "s[ >> " + start + " " + end + " << ] .";
                        msArea.writeln("Error " + cocosLine);
                        return;
                    }
                }catch(NumberFormatException nfe){
                    cocosLine = cocosLine + " >> " + scopeStartText.getText() +
                            " " + scopeEndText.getText() + " << " + 
                                "not numbers.";
                    msArea.writeln("Error " + cocosLine);
                    return;
                }
            }
            
            if (!coPriorityText.getText().equals("")){
                try{
                    int priority = Integer.parseInt(coPriorityText.getText());
                    if(priority < 0){
                       cocosLine = cocosLine + "p >> " + priority + 
                            " << negative number.";
                       msArea.writeln("Error " + cocosLine);
                       return;
                    }
                    cocosLine = cocosLine + "p " + priority + " ";
                    
                }catch(NumberFormatException nfe){
                    cocosLine = cocosLine + "p >> " + coPriorityText.getText() + 
                            " << not a number.";
                    msArea.writeln("Error " + cocosLine);
                    return;
                }
            }
            
            if(coSearch.getSelectedIndex() != 0){
                cocosLine = cocosLine 
                        + searchOption[coSearch.getSelectedIndex()] + " ";
            }
            
            cocosConstraints = cocosConstraints + cocosLine + "\n";
            msArea.writeln("Constraint " + cocosLine + " added.");
            
            strStartText.setHintText("");
            strEndText.setHintText("");
            strPriorityText.setHintText("");
            coStartText.setHintText("");
            coEndText.setHintText(""); 
            coPriorityText.setHintText("");
            scopeStartText.setHintText("");
            scopeEndText.setHintText("");
            
        }
        
        
        public String getCocosConstraints(){
            return cocosConstraints;
        }
    }

}//SelectConstraintPanel