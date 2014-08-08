package jafatt;

import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.KeyEvent;
import javax.swing.JMenu;
import javax.swing.JMenuBar;
import javax.swing.JMenuItem;
import javax.swing.JOptionPane;

public class MenuBar extends JMenuBar{
    public JMenuItem openProteinItem, saveItem;
    
    private UserFrame view;
    
    private JMenu fileMenu, viewMenu, toolsMenu, showMenu, helpMenu;
    private JMenuItem openItem, buildItem, downloadItem, exitItem, compileItem,
            centerItem, resetItem, showItem, measureItem, JmolItem, 
            alignmentItem, helpItem, aboutItem;
    
    public JMenuItem assembleItem, extractItem, outputItem;
    
    public MenuBar(){
        setup();
    }
    
    /* Create the main menu */
    /*
    private void setup(){
        fileMenu = new JMenu();
        openItem = new JMenuItem();
        downloadItem = new JMenuItem();
        openProteinItem = new JMenuItem();
        saveItem = new JMenuItem();
        exitItem = new JMenuItem();
        viewMenu = new JMenu();
        editMenu = new JMenu();
        wrapItem = new JMenuItem();
        centerItem = new JMenuItem();
        resetItem = new JMenuItem();
        showMenu = new JMenu();
        showItem = new JMenuItem();
        toolsMenu = new JMenu();
        measureItem = new JMenuItem();
        JmolItem = new JMenuItem();
        alignmentItem = new JMenuItem();
        helpMenu = new JMenu();
        helpItem = new JMenuItem();
        aboutItem = new JMenuItem();
        

        fileMenu.setText("File");
        fileMenu.setMnemonic(KeyEvent.VK_F);
        add(fileMenu);
        
        openItem.setText("Load Target");
        openItem.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent evt) {
                loadTargetEvent();
            }
        });
        openItem.setMnemonic(KeyEvent.VK_O);
        fileMenu.add(openItem);
        
        downloadItem.setText("Download Protein");
        downloadItem.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent evt) {
                downloadProtein();
            }
        });
        downloadItem.setMnemonic(KeyEvent.VK_O);
        fileMenu.add(downloadItem);
        
        openProteinItem.setText("Load Protein");
        openProteinItem.setEnabled(false);
        openProteinItem.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent evt) {
                loadProteinEvent();
            }
        });
        fileMenu.add(openProteinItem);
        
        saveItem.setText("Save Fragments");
        saveItem.setEnabled(false);
        saveItem.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent evt) {
                saveEvent();
            }
        });
        saveItem.setMnemonic(KeyEvent.VK_S);
        //fileMenu.add(saveItem);
        
        exitItem.setText("Exit");
        exitItem.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent evt) {
                exitEvent();
            }
        });
        exitItem.setMnemonic(KeyEvent.VK_X);
        fileMenu.add(exitItem);
        
        editMenu.setText("Edit");
        editMenu.setMnemonic(KeyEvent.VK_E);
        //add(editMenu);

        wrapItem.setText("Wrap Target");
        wrapItem.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent evt) {
                wrapEvent(evt);
            }
        });
        editMenu.add(wrapItem);
        
        centerItem.setText("Center on fragment");
        centerItem.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent evt) {
                centerEvent();
            }
        });
        editMenu.add(centerItem);

        resetItem.setText("Reset oritentation");
        resetItem.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent evt) {
                resetEvent();
            }
        });
        editMenu.add(resetItem);

        showMenu.setText("Show");
        //add(showMenu);

        showItem.setText("Show Constratins");
        showItem.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent evt) {
                showEvent();
            }
        });
        showMenu.add(showItem);

        toolsMenu.setText("Tools");
        toolsMenu.setMnemonic(KeyEvent.VK_T);
        //add(toolsMenu);

        measureItem.setText("Measure");
        measureItem.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent evt) {
                measureEvent();
            }
        });
        toolsMenu.add(measureItem);
        
        JmolItem.setText("Jmol");
        JmolItem.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent evt) {
                JmolEvent();
            }
        });
        toolsMenu.add(JmolItem);
        
        alignmentItem.setText("Alignment");
        alignmentItem.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent evt) {
                alignmentEvent();
            }
        });
        toolsMenu.add(alignmentItem);

        helpMenu.setText("Help");
        helpMenu.setMnemonic(KeyEvent.VK_HELP);
        add(helpMenu);

        helpItem.setText("Help...");
        helpItem.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent evt) {
                helpEvent();
            }
        });
        helpMenu.add(helpItem);

        aboutItem.setText("About");
        aboutItem.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent evt) {
                aboutEvent();
            }
        });
        helpMenu.add(aboutItem);
    }//setup
    */
    
    private void setup() {
        
        fileMenu = new JMenu();
        openItem = new JMenuItem();
        buildItem = new JMenuItem();
        downloadItem = new JMenuItem();
        openProteinItem = new JMenuItem();
        saveItem = new JMenuItem();
        exitItem = new JMenuItem();
        viewMenu = new JMenu();
        assembleItem = new JMenuItem();
        extractItem = new JMenuItem();
        outputItem = new JMenuItem();
        toolsMenu = new JMenu();
        compileItem = new JMenuItem();
        helpMenu = new JMenu();
        helpItem = new JMenuItem();
        aboutItem = new JMenuItem();
        
        /* Set the items */
        fileMenu.setText("File");
        fileMenu.setMnemonic(KeyEvent.VK_F);
        add(fileMenu);

        openItem.setText("Load Target");
        openItem.addActionListener(new ActionListener() {

            @Override
            public void actionPerformed(ActionEvent evt) {
                loadTargetEvent();
            }
        });
        openItem.setMnemonic(KeyEvent.VK_O);
        fileMenu.add(openItem);
        
        buildItem.setText("Build Target");
        buildItem.addActionListener(new ActionListener() {

            @Override
            public void actionPerformed(ActionEvent evt) {
                buildTargetEvent();
            }
        });
        buildItem.setMnemonic(KeyEvent.VK_O);
        fileMenu.add(buildItem);

        downloadItem.setText("Download Protein");
        downloadItem.addActionListener(new ActionListener() {

            @Override
            public void actionPerformed(ActionEvent evt) {
                downloadProtein();
            }
        });
        downloadItem.setMnemonic(KeyEvent.VK_O);
        fileMenu.add(downloadItem);

        openProteinItem.setText("Load Protein");
        openProteinItem.setEnabled(false);
        openProteinItem.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent evt) {
                loadProteinEvent();
            }
        });
        fileMenu.add(openProteinItem);

        saveItem.setText("Save Fragments");
        saveItem.setEnabled(false);
        saveItem.addActionListener(new ActionListener() {

            @Override
            public void actionPerformed(ActionEvent evt) {
                saveEvent();
            }
        });
        saveItem.setMnemonic(KeyEvent.VK_S);
        //fileMenu.add(saveItem);

        exitItem.setText("Exit");
        exitItem.addActionListener(new ActionListener() {

            @Override
            public void actionPerformed(ActionEvent evt) {
                exitEvent();
            }
        });
        exitItem.setMnemonic(KeyEvent.VK_X);
        fileMenu.add(exitItem);
        
        viewMenu.setText("View");
        viewMenu.setMnemonic(KeyEvent.VK_V);
        
        extractItem.setText("Extraction Panel");
        extractItem.setMnemonic(KeyEvent.VK_E);
        extractItem.setEnabled(false);
        extractItem.addActionListener(new ActionListener() {

            @Override
            public void actionPerformed(ActionEvent evt) {
                viewEvent(Defs.EXTRACTION);
            }
        });
        viewMenu.add(extractItem);
        
        assembleItem.setText("Assembling Panel");
        assembleItem.setMnemonic(KeyEvent.VK_A);
        assembleItem.setEnabled(false);
        assembleItem.addActionListener(new ActionListener() {

            @Override
            public void actionPerformed(ActionEvent evt) {
                viewEvent(Defs.ASSEMBLING);
            }
        });
        viewMenu.add(assembleItem);
        
        outputItem.setText("OutPut Panel");
        outputItem.setMnemonic(KeyEvent.VK_O);
        outputItem.setEnabled(false);
        outputItem.addActionListener(new ActionListener() {

            @Override
            public void actionPerformed(ActionEvent evt) {
                viewEvent(Defs.OUTPUT);
            }
        });
        viewMenu.add(outputItem);
        
        add(viewMenu);

        toolsMenu.setText("Tools");
        toolsMenu.setMnemonic(KeyEvent.VK_T);
        add(toolsMenu);
        
        compileItem.setText("Compile");
        compileItem.setMnemonic(KeyEvent.VK_C);
        compileItem.addActionListener(new ActionListener() {

            @Override
            public void actionPerformed(ActionEvent evt) {
                compileEvent();
            }
        });
        
        toolsMenu.add(compileItem);

        helpMenu.setText("Help");
        helpMenu.setMnemonic(KeyEvent.VK_HELP);
        add(helpMenu);

        helpItem.setText("Help...");
        helpItem.addActionListener(new ActionListener() {

            @Override
            public void actionPerformed(ActionEvent evt) {
                helpEvent();
            }
        });
        helpMenu.add(helpItem);

        aboutItem.setText("About");
        aboutItem.addActionListener(new ActionListener() {

            @Override
            public void actionPerformed(ActionEvent evt) {
                aboutEvent();
            }
        });
        helpMenu.add(aboutItem);
    }//setup
    
    /* Load the target */
   private void loadTargetEvent(){
        view.getViewActions().loadTargetEvent();
   }//loadTargetEvent
   
   private void buildTargetEvent(){
        view.getViewActions().buildTargetEvent();
   }//loadTargetEvent
   
   private void downloadProtein(){
       view.getViewActions().downloadProtein();       
   }
   
   /* Load a protein from which extract the fragments */
   private void loadProteinEvent(){
       view.getViewActions().loadProteinEvent();                                            
   }//loadProteinEvent
   
   /* Save data */
   private void saveEvent(){
       
   }//saveEvent
   
   /* Exit */
   private void exitEvent(){
       int result = JOptionPane.showConfirmDialog(
               view, "Do you really want to exit?", "Exit", 
               JOptionPane.YES_NO_OPTION);
       if(result == JOptionPane.YES_OPTION){
           view.printStringLn("Exiting...");
           view.setVisible(false);
           System.exit(0);
       }else
           return;      
   }//exitEvent
   
   private void viewEvent(int panel){
       
       Thread threadViewOptions;
       ViewOptionsPanel vop = new ViewOptionsPanel(view, panel);

       /* Create the thread */
       threadViewOptions = new Thread(vop);
       threadViewOptions.start();
       
   }
   
   private void compileEvent(){
       Thread threadCompilePanel;
       CompilePanel cp = new CompilePanel(view);

       /* Create the thread */
       threadCompilePanel = new Thread(cp);
       threadCompilePanel.start();
   }
   
   /* Wrap the target sequence */
   private void wrapEvent(ActionEvent evt){
       view.getViewActions().wrapTarget(evt);
   }//wrapEvent
   
   /* Center the fragment */
   private void centerEvent(){
       
   }//centerEvent
   
   /* Reset the view */
   private void resetEvent(){
       
   }//resetEvent
   
   /* Show all the constraints on the fragments*/
   private void showEvent(){
       
   }//showEvent
   
   /* Measure */
   private void measureEvent(){
       
   }//measureEvent
   
   /* Opens a Jmol visualizer */
   private void JmolEvent(){
       
   }//JmolEvent
   
   /* Opens a panel to align structures */
   private void alignmentEvent(){
       
   }//alignmentEvent
   
   /* Opens the help panel */
   private void helpEvent(){
       
   }//helpEvent
   
   /* Some infos */
   private void aboutEvent(){
       
   }//aboutEvent
   
   /* Set the view component (UserFrame) */
   public void setView(UserFrame view){
       this.view = view;
   }//setView
    
}
