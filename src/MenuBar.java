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
    
    private JMenu fileMenu, editMenu, showMenu, toolsMenu, helpMenu;
    private JMenuItem openItem, downloadItem, exitItem, wrapItem, 
        centerItem, resetItem, showItem, measureItem, JmolItem, alignmentItem, helpItem, aboutItem;
    
    public MenuBar(){
        setup();
    }
    
    /* Create the main menu */
    private void setup(){
        fileMenu = new JMenu();
        openItem = new JMenuItem();
        downloadItem = new JMenuItem();
        openProteinItem = new JMenuItem();
        saveItem = new JMenuItem();
        exitItem = new JMenuItem();
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
        
        downloadItem.setText("Download Target");
        downloadItem.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent evt) {
                DownloadFastaPanel dfp = new DownloadFastaPanel(view);
                Thread threadFasta;
                threadFasta = new Thread(dfp);
                threadFasta.start();
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
        fileMenu.add(saveItem);
        
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
        add(editMenu);

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
        add(showMenu);

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
        add(toolsMenu);

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
    
    /* Load the target */
   private void loadTargetEvent(){
        view.getViewActions().loadTargetEvent();
   }//loadTargetEvent
   
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
