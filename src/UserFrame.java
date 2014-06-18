package jafatt;

import java.awt.*;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.util.Observable;
import java.util.Observer;
import javax.swing.*;
import org.biojava.bio.seq.Sequence;
import org.biojava.bio.structure.Structure;
import java.awt.event.ComponentListener;
import java.awt.event.ComponentEvent;
import java.awt.event.KeyEvent;
import java.awt.AWTException;

public class UserFrame extends JFrame implements Observer, MouseListener, ComponentListener {

    public UpdateView upView;
    private final int LOADSAVEBUTTONCOUNT = 3;
    private final int MOVEBUTTONCOUNT = 6;
    private final int TOOLSBUTTONCOUNT = 7;
    private final int TOTALBUTTON = MOVEBUTTONCOUNT + TOOLSBUTTONCOUNT;
    private long w;
    private String PROGTARGET = "  TARGET  ";
    private String PROGEXTRACTION = "  EXTRACTION ";
    private String PROGASSEMBLING = "  ASSEMBLING  ";
    private String PROGOUTPUT = "  OUTPUT  ";
    /* Panels */
    private Container pane;
    private String currentDir;
    private JPanel topPanel;
    private IconToolbar toolbar;
    private UserFrameActions ufa;
    private JPanel progressBar;
    private ProgressIcon targetProgIcon;
    private ProgressIcon extractionProgIcon;
    private ProgressIcon assemblingProgIcon;
    private ProgressIcon outputProgIcon;
    private Controller ctr;
    private Bin bin;
    private TargetPanel targetPanel;
    private InfoPanel infoPanel;
    private Dimension dimScreen;
    private ExtractionPanel extractionPanel;
    private AssemblingPanel assemblingPanel;
    private OutputPanel outputPanel;
    private ProteinModel model;
    public Rmsd rmsd;
    /* Menu */
    private MenuBar topMenuBar;
    /* Icons, buttons and labels*/
    private String separator;
    private String imSuffix;
    private String frameIcon;
    private JButton[] buttons = new JButton[TOTALBUTTON];
    private String[] icons = new String[TOTALBUTTON];
    private String[] descriptions = new String[TOTALBUTTON];
    private String[] info = new String[TOTALBUTTON];
    private boolean fragmentLoaded = false;
    private boolean outputLoaded = false;
    private Robot r;
    public boolean rotate = false, move = false;
    /* Other stuff */
    protected BarPanel barPanel;

    public UserFrame() {
        super("FIASCO");
        pane = getContentPane();
        pane.setLayout(new BorderLayout());
        setupIcons();
        setupScreen();
        setupMenu();
    }

    /* Set icons for the toolbar */
    private void setupIcons() {

        /* Get the right directory separator */
        currentDir = System.getProperty("user.dir");
        separator = Utilities.getSeparator();

        imSuffix = separator + "images" + separator;

        /* Set icons */
        frameIcon = currentDir + imSuffix + "fiasco.gif";

        /* Load and save icons */
        icons[0] = currentDir + imSuffix + "folder.png";
        icons[1] = currentDir + imSuffix + "globe.png";
        icons[2] = currentDir + imSuffix + "gears.png";

        /* Tool icons */
        icons[3] = currentDir + imSuffix + "pin.png";
        icons[4] = currentDir + imSuffix + "pencil-ruler.png";
        icons[5] = currentDir + imSuffix + "home.png";
        icons[6] = currentDir + imSuffix + "chain--pencil.png";

        /* Movement icons */
        icons[7] = currentDir + imSuffix + "arrow-return-000-left.png";
        icons[8] = currentDir + imSuffix + "arrow-return-180-left.png";
        icons[9] = currentDir + imSuffix + "arrow-090.png";
        icons[10] = currentDir + imSuffix + "arrow-270.png";
        icons[11] = currentDir + imSuffix + "arrow.png";
        icons[12] = currentDir + imSuffix + "arrow-180.png";

        /* Description */
        descriptions[0] = "Load";
        descriptions[1] = "Download";
        descriptions[2] = "Back";

        descriptions[3] = "Center";
        descriptions[4] = "Measure";
        descriptions[5] = "Reset";
        descriptions[6] = "ShowConstraints";

        descriptions[7] = "Front";
        descriptions[8] = "Back";
        descriptions[9] = "Top";
        descriptions[10] = "Bottom";
        descriptions[11] = "Right";
        descriptions[12] = "Left";

        /* Info tooltip */
        info[0] = "Load a Protein";
        info[1] = "Download a Protein from www.rcsb.org";
        info[2] = "Run Cocos";

        info[3] = "Click on a fragment to center to it";
        info[4] = "Measurement";
        info[5] = "Reset the protein to its initial conditions";
        info[6] = "Show all constraints set on fragments";

        info[7] = "Front view";
        info[8] = "Back view";
        info[9] = "Top view";
        info[10] = "Bottom view";
        info[11] = "Right view";
        info[12] = "Left view";

    }//setupIcons

    /* Set screen and its dimension */
    private void setupScreen() {

        /* Debug */
        System.out.println("CurrentDirectory: " + currentDir + separator);

        /* Set the frame icon to an image loaded from a file */
        setIconImage(new ImageIcon(frameIcon).getImage());

        /* Set the screen and the Close operation */
        dimScreen = Toolkit.getDefaultToolkit().getScreenSize();
        Dimension dView = new Dimension();
        dView.setSize(dimScreen.getWidth() * 11 / 12, dimScreen.getHeight() * 14 / 15);
        setPreferredSize(dView);
        w = (long) dimScreen.getWidth();
        setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
    }//setupScreen

    /* Set the content of the screen */
    private void setupContent() {

        /* Set the toolbar */
        ufa = new UserFrameActions(ctr, this);
        toolbar = new IconToolbar(ufa, MOVEBUTTONCOUNT, buttons, descriptions,
                icons, info);
        //for(int i = 0; i < TOTALBUTTON; i++){ 
        for (int i = 0; i < 3; i++) {
            toolbar.add(buttons[ i]);
            if (i == LOADSAVEBUTTONCOUNT - 1) {
                toolbar.addSeparator(new Dimension(20, 0));
            }
            if (i == TOOLSBUTTONCOUNT - 1) {
                toolbar.addSeparator(new Dimension(30, 0));
            }
        }

        Dimension dView = new Dimension();
        dView.setSize(dimScreen.getWidth(), dimScreen.getHeight() / 25);
        toolbar.setPreferredSize(dView);

        /* Set the working panels */
        infoPanel = new InfoPanel(true);
        targetPanel = new TargetPanel();
        extractionPanel = new ExtractionPanel(this, PROGEXTRACTION);
        assemblingPanel = new AssemblingPanel(this, PROGASSEMBLING);
        outputPanel = new OutputPanel(this, PROGOUTPUT);

        /* Add the listener to the panels to identify the position of the user*/
        extractionPanel.addMouseListener(this);
        assemblingPanel.addMouseListener(this);
        outputPanel.addMouseListener(this);
        extractionPanel.getJMolPanel().addMouseListener(this);
        //extractionPanel.addMouseListener(this);
        //assemblingPanel.addMouseListener(this);
        assemblingPanel.getJMolPanel().addMouseListener(this);
        outputPanel.getJMolPanel().addMouseListener(this);

        rmsd = new Rmsd();

        this.addComponentListener(this);

        /* Add subpanels to the main panel */
        bin = new Bin(this, targetPanel, extractionPanel, assemblingPanel, outputPanel);
    }//setupContent

    /* Set the position of the objects in the UserFrame */
    private void setupPosition() {
        progressBar = new JPanel();
        progressBar.setBorder(BorderFactory.createMatteBorder(0, 0, 1, 0, Defs.INNERCOLOR));
        progressBar.setBackground(Defs.INNERCOLOR);
        FlowLayout fl = new FlowLayout();

        /* Set the top layout of the UserFrame */
        topPanel = new JPanel();
        topPanel.setLayout(new GridLayout(1, 2));

        /* Set the progress bar */
        fl.setAlignment(FlowLayout.RIGHT);
        progressBar.setLayout(fl);
        targetProgIcon = new ProgressIcon(PROGTARGET, currentDir + imSuffix);
        extractionProgIcon = new ProgressIcon(PROGEXTRACTION, currentDir + imSuffix);
        assemblingProgIcon = new ProgressIcon(PROGASSEMBLING, currentDir + imSuffix);
        outputProgIcon = new ProgressIcon(PROGOUTPUT, currentDir + imSuffix);
        progressBar.add(targetProgIcon);
        progressBar.add(extractionProgIcon);
        progressBar.add(assemblingProgIcon);
        progressBar.add(outputProgIcon);

        /* Add the components to the top bar */
        topPanel.add(toolbar);
        topPanel.add(progressBar);

        /* Add the "working" components */
        pane.add(topPanel, BorderLayout.NORTH);
        pane.add(bin, BorderLayout.CENTER);
        pane.add(infoPanel, BorderLayout.SOUTH);
    }//setupPosition

    /* Create the main menu */
    private void setupMenu() {
        topMenuBar = new MenuBar();
        topMenuBar.setView(this);

        /* Set the menu bar */
        setJMenuBar(topMenuBar);
    }//setupMenu

    /* Set controller and model */
    public void setConMod(Controller ctr, ProteinModel model) {
        this.ctr = ctr;
        this.model = model;

        setupContent();
        setupPosition();

        /* Now we can also setup the model for the UpdateView instance */
        upView = new UpdateView(this, model);
    }//setConMod

    /* Get the panels */
    public JPanel getPanel(int panel) {
        switch (panel) {
            case Defs.ASSEMBLING:
                return assemblingPanel;
            case Defs.EXTRACTION:
                return extractionPanel;
            case Defs.TARGET:
                return targetPanel;
            case Defs.INFOPANEL:
                return infoPanel;
            case Defs.OUTPUT:
                return outputPanel;
            default:
                return null;
        }
    }//getPanel

    /* Get the controller to communicate the user's input.
     * Method used by sub-panels.
     */
    public Controller getController() {
        return ctr;
    }//getController

    public ProteinModel getModel() {
        return model;
    }

    @Override
    public void componentHidden(ComponentEvent e) {
    }

    @Override
    public void componentMoved(ComponentEvent e) {
    }

    @Override
    public void componentResized(ComponentEvent e) {
        w = e.getComponent().getBounds().width;
        bin.update(w);
    }

    @Override
    public void componentShown(ComponentEvent e) {
    }

    /* Implement all abstract methods for MouseListener */
    @Override
    public void mouseExited(MouseEvent e) {
    }

    @Override
    public void mouseClicked(MouseEvent e) {
    }

    @Override
    public void mousePressed(MouseEvent e) {
        if (e.getComponent().equals(assemblingPanel)
                || e.getComponent().equals(assemblingPanel.getJMolPanel())) {
            if (move) {
                upView.execute(Utilities.deleteConnectionString(
                        model.getDimensionFragmentsA()),
                        Defs.ASSEMBLING);
                try {
                    r = new Robot();
                    r.keyPress(KeyEvent.VK_ALT);
                    r.keyPress(KeyEvent.VK_SHIFT);
                } catch (AWTException awte) {
                }
            }
            if (rotate) {
                upView.execute(Utilities.deleteConnectionString(
                        model.getDimensionFragmentsA()),
                        Defs.ASSEMBLING);
                try {
                    r = new Robot();
                    r.keyPress(KeyEvent.VK_ALT);
                } catch (AWTException awte) {
                }
            }
        }

    }

    @Override
    public void mouseReleased(MouseEvent e) {
        if (move || rotate) {
            try {
                r = new Robot();
                r.keyRelease(KeyEvent.VK_ALT);
            } catch (AWTException awte) {
            }

            if (move) {
                try {
                    r = new Robot();
                    r.keyRelease(KeyEvent.VK_SHIFT);
                } catch (AWTException awte) {
                }
            }
            upView.connectFragments(false);
            //model.setCurrentFragmentA(model.getCurrentFragmentA());
        }
    }

    @Override
    public void mouseEntered(MouseEvent e) {
        if (e.getComponent().equals(assemblingPanel)
                || e.getComponent().equals(assemblingPanel.getJMolPanel())) {
            if (ctr != null) {
                ctr.setCurrentPanel(Defs.ASSEMBLING);
            }
        } else if (e.getComponent().equals(extractionPanel)
                || e.getComponent().equals(extractionPanel.getJMolPanel())) {
            if (ctr != null) {
                ctr.setCurrentPanel(Defs.EXTRACTION);
            }
        } else if (e.getComponent().equals(outputPanel)
                || e.getComponent().equals(outputPanel.getJMolPanel())) {
            if (ctr != null) {
                ctr.setCurrentPanel(Defs.OUTPUT);
            }
        }
    }//mouseEntered

    public void expand(int panel) {
        bin.setSeparators(panel, w);
    }

    /* Notify the loaded target */
    public void targetLoaded() {
        topMenuBar.openProteinItem.setEnabled(true);
        targetProgIcon.enableProgress();
        printStringLn("Target Loaded");
        HeaderPdb.targetId(model.getTargetPath());
        toolbar.enableCocos();
    }//targetLoaded

    /* Init the loading of the protein */
    public void initProteinLoaded() {
        /* Create a new thread */
        Thread threadTransferBar;

        /* Create a bar for the transfer process */
        barPanel = new BarPanel("Loading in progress...", this);

        /* Create the thread */
        threadTransferBar = new Thread(barPanel);
        threadTransferBar.start();
        printStringLn("Start the loading of the protein into "
                + "the Extration panel");
    }//initProtinLoaded

    /* Notify the loaded protein */
    public void proteinLoaded() {

        /* Stop the progress bar */
        barPanel.stop();

        /* Enable icon */
        extractionProgIcon.enableProgress();
        topMenuBar.extractItem.setEnabled(true);
       /* Print some infos */
        printStringLn("Protein " + model.idProteinCode + " Loaded");
        printStringLn("Note:");
        printStringLn("If something goes wrong with the selection of "
                + "the fragments, please check the PDB file's header first", false);
    }//proteinLoaded

    public void proteinNotLoaded() {
        barPanel.stop();
        printStringLn("Error on loading the protein");
    }//proteinLoaded

    /* Init the transfer of fragments from Extraction panel to 
     * Assembling panel */
    public void initTransfer() {

        /* Create a new thread */
        Thread threadTransferBar;

        /* Create a bar for the transfer process */
        barPanel = new BarPanel("Transfer in progress...", this);

        /* Create the thread */
        threadTransferBar = new Thread(barPanel);
        threadTransferBar.start();
        printStringLn("Start the transfer of fragments from Extration panel "
                + "to Assembling panel");
    }//initTransfer

    /* Notify the completion of the transfer of fragments */
    public void fragmentsLoaded() {

        fragmentLoaded = true;
        topMenuBar.assembleItem.setEnabled(true);
        /* Stop the progress bar */
        barPanel.stop();

        /* Enable icon */
        assemblingProgIcon.enableProgress();

        /* Print some infos */
        printStringLn("Fragments transferred on Assembling panel");
    }//proteinLoaded

    public void cacheToPdb() {
        ctr.saveToPdb();
    }

    public boolean fragmentLoaded() {
        return fragmentLoaded;
    }

    /* Get the UserFrame Actions */
    public UserFrameActions getViewActions() {
        return ufa;
    }//getViewActions

    /* Show the target sequence on target panel */
    public void showTargetSequence(Sequence seq) {
        targetPanel.setTarget(seq);
    }//showTargetSequence

    /* Print strings on the info panel */
    public void printStringLn(String str) {
        infoPanel.printStringLn(str);
    }//printStringLn

    public void printStringLn(String str, boolean b) {
        infoPanel.printStringLn(str, b);
    }//printStringLn

    public void printStringLn() {
        infoPanel.printStringLn();
    }//printStringLn

    public void printString(String str) {
        infoPanel.printString(str);
    }//printString

    /* Observer implementation */
    @Override
    public void update(Observable o, Object info) {
        switch ((Integer) info) {
            case Defs.NEW_TARGET:
                showTargetSequence(model.getTargetSequence());
                break;
            case Defs.NEW_PROTEIN:
                upView.newStructure();
                break;
            case Defs.DELETE_FRAGMENT:
                upView.colorFragment(model.getDeletedFragmentE(),
                        Defs.COLOR_DESELECT_FRAGMENT, Defs.EXTRACTION);
                upView.colorAllFragments(Defs.COLOR_ADD_FRAGMENT, Defs.EXTRACTION);
                printStringLn(Utilities.deleteFragmentInfo(
                        model.getDeletedFragmentE()));
                break;
            case Defs.ADD_FRAGMENT_ON_EXTRACTION:
                upView.colorFragment(model.getCurrentFragmentE(),
                        Defs.COLOR_ADD_FRAGMENT, Defs.EXTRACTION);
                printStringLn(Utilities.addFragmentInfo(
                        model.getCurrentFragmentE()));
                break;
            case Defs.SET_OFFSET:
                upView.setOffsetOnTarget(model.getCurrentFragmentE());
                break;
            case Defs.SET_FRAGMENT_ON_ASSEMBLING:
                upView.addFragment(model.getCurrentFragmentA());
                break;
            case Defs.SELECT_FRAGMENT_ON_ASSEMBLING:
                //upView.deselectAll();
                //upView.colorFragment(model.getCurrentFragmentA(), 
                //                     Defs.COLOR_ADD_FRAGMENT, Defs.ASSEMBLING);
                upView.colorAllFragments(Defs.COLOR_ADD_FRAGMENT, Defs.ASSEMBLING_S);
                //upView.colorConstraints(model.getCurrentFragmentA());
                printStringLn(
                        Utilities.selectFragmentInfo(model.getCurrentFragmentA()));
                break;
            case Defs.DESELECT_FRAGMENT_ON_ASSEMBLING:
                upView.deselectAll();
                upView.colorAllFragments(Defs.COLOR_ADD_FRAGMENT, Defs.ASSEMBLING_S);
                printStringLn(
                        Utilities.deselectFragmentInfo(model.getCurrentFragmentA()));
                break;
            case Defs.SET_CONSTRAINT_BLOCK:
                upView.deselectAll();
                upView.colorConstraints(model.getCurrentFragmentA());
                break;
        }

        /* if((Integer) info == Defs.NEW_TARGET)
        showTargetSequence(model.getTargetSequence());
        else if((Integer) info == Defs.NEW_PROTEIN)
        upView.newStructure();
        else if((Integer) info == Defs.DELETE_FRAGMENT){
        upView.colorFragment(model.getDeletedFragmentE(), 
        Defs.COLOR_DESELECT_FRAGMENT, Defs.EXTRACTION);
        upView.colorAllFragments(Defs.COLOR_ADD_FRAGMENT, Defs.EXTRACTION);
        printStringLn(Utilities.deleteFragmentInfo(
        model.getDeletedFragmentE())
        );
        }else if((Integer)info == Defs.ADD_FRAGMENT_ON_EXTRACTION){
        upView.colorFragment(model.getCurrentFragmentE(), 
        Defs.COLOR_ADD_FRAGMENT, Defs.EXTRACTION);
        printStringLn(Utilities.addFragmentInfo(
        model.getCurrentFragmentE())
        );
        }else if((Integer) info == Defs.SET_OFFSET){
        upView.setOffsetOnTarget(model.getCurrentFragmentE());
        }else if((Integer) info == Defs.SET_FRAGMENT_ON_ASSEMBLING){
        upView.addFragment(model.getCurrentFragmentA());
        }else if((Integer) info == Defs.SELECT_FRAGMENT_ON_ASSEMBLING){
        upView.deselectAll();
        upView.colorFragment(model.getCurrentFragmentA(), 
        Defs.COLOR_ADD_FRAGMENT, Defs.ASSEMBLING);
        printStringLn(Utilities.selectFragmentInfo(model.getCurrentFragmentA())
        );
        }else if((Integer) info == Defs.DESELECT_FRAGMENT_ON_ASSEMBLING){
        upView.deselectAll();
        } */
    }//update

    public void clearAssemblingPanel() {
        upView.deleteAllFragmentsAssembling();
        assemblingPanel.executeCmd("delete $*;");
        assemblingPanel.executeCmd("DELETE;");
    }

    public void preparePanel(int panel) {
        String cmd = Defs.COMMAND_PREPARE_STRUCURE_BASE
                + "background hover red; "
                //+ "background labels black; axes molecular; axes on; "
                + "background labels black; "
                + "frame all; select *;"; // center;";

        //upView.execute(Defs.COMMAND_PREPARE_STRUCURE, panel);
        upView.execute(cmd, panel);
    }

    public void clearExtractionPanel() {
        model.deleteFragmentE();
    }

    public void loadOutput(Structure structure) {
        String protein = Defs.PROTEINS_PATH
                + model.idProteinCode + ".out.pdb" + "; ";
        //view.printStringLn("Loading protein...");
        outputPanel.executeCmd("load " + protein);
        outputPanel.executeCmd(Defs.COMMAND_PREPARE_STRUCURE);
        rmsd.clearDisplay();
        ctr.loadProteinRmsd();
        rmsd.loadProtein(protein);
        computeRmsd();
        outputLoaded();
    }

    public void computeRmsd() {
        int modelDisplayed = outputPanel.molViewPanel.getDisplayedModel();
        String[] out = rmsd.computeRmds(modelDisplayed);
        outputPanel.updateRmsd(out[0] + " - " + out[1]);
    }

    public void outputLoaded() {

        outputLoaded = true;

        topMenuBar.outputItem.setEnabled(true);

        /* Stop the progress bar */

        /* Enable icon */
        outputProgIcon.enableProgress();

        /* Print some infos */
    }//proteinLoaded
}//UserFrame
