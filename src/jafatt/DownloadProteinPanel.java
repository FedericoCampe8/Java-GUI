package jafatt;

import javax.swing.JFrame;
import javax.swing.JPanel;
import javax.swing.JCheckBox;
import javax.swing.BorderFactory;
import java.awt.Container;
import java.awt.event.ActionEvent;
import java.awt.GridLayout;
import java.awt.BorderLayout;

public class DownloadProteinPanel extends JFrame implements Runnable {

    private UserFrame view;
    private JPanel panel, subPanel;
    private JPanel checkPanel, buttonPanel;
    private OpButton okButton, cancelButton;
    private HintTextField proteinText;
    private JCheckBox pdb, fasta;
    
    public DownloadProteinPanel (UserFrame view){
        
        super("Fiasco - Download Protein");
        
        this.view = view;

        panel = new JPanel(new BorderLayout());
        subPanel = new JPanel(new BorderLayout());
        checkPanel = new JPanel(new GridLayout(1,2));
        buttonPanel = new JPanel();
        proteinText = new HintTextField(" Download a protein from www.rcsb.org ", 50);
        okButton = new OpButton("Ok", "") {
            @Override
            public void buttonEvent(ActionEvent evt){
                buttonEvent(true);
            }
        };
        
        cancelButton = new OpButton("Cancel", "") {
            @Override
            public void buttonEvent(ActionEvent evt){
                buttonEvent(false);
            }
        };
        
        fasta = new JCheckBox("Fasta",true);
        pdb = new JCheckBox("Pdb");
        
        /* Setup layout */
        
        buttonPanel.add(okButton);
        buttonPanel.add(cancelButton);
        checkPanel.add(fasta);
        checkPanel.add(pdb);
        
        subPanel.add(proteinText, BorderLayout.CENTER);
        subPanel.add(checkPanel, BorderLayout.EAST);
        subPanel.setBorder(BorderFactory.createTitledBorder("Download a Protein")); 
        panel.add(subPanel, BorderLayout.CENTER);
        panel.add(buttonPanel, BorderLayout.SOUTH);

    }    
    @Override
    public void run() {
        Container ct = getContentPane();
        ct.add(panel);
        update();
        setVisible(true);
        fasta.requestFocus();
    }//run
    
     private void update(){
        pack();
        setLocationRelativeTo(view);
    }
    
    private void buttonEvent(boolean ok){
        
        if(ok){
            String proteinID = proteinText.getText();
            if(proteinID.equals("")){
                view.printStringLn("Select a proper Protein Name.");
                return;
            }
            int[] output = Downloader.downloadProtein(view, proteinID, pdb.isSelected(), fasta.isSelected());
            
            if(fasta.isSelected()){
                //if (Utilities.downloadProtein(proteinID,false)[1] == 1) {
                if (output[1] == 1) {
                    boolean loaded;
                    loaded = view.getController().loadStructure(
                            Defs.PROTEINS_PATH + proteinID + ".fasta", Defs.TARGET, false);
                    if (loaded) {
                        view.getViewActions().loadATarget = true;
                        view.targetLoaded();
                    } else {
                        view.printStringLn("Error on loading the target");
                    }
                    setVisible(false);
                    dispose();
                }else{
                    view.printStringLn("Error downloading " + proteinID + ".fasta " +
                            "from rcbs.com protein database. " +
                            "Check your Internet Connection " +
                            "or the protein name. ");
                }
            }
            if(pdb.isSelected()){
                if(output[0] == 1){
                    setVisible(false);
                    dispose();
                }else{
                    view.printStringLn("Error downloading " + proteinID + ".pdb " +
                            "from rcbs.com protein database. " +
                            "Check your Internet Connection " +
                            "or the protein name. ");
                }
            }
        }else{
            setVisible(false);
            dispose();            
        }
    }

}
