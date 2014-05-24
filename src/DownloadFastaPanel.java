package jafatt;


import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.InputStream;
import java.net.URL;
import java.util.Scanner;
import javax.swing.JFrame;
import javax.swing.JPanel;
import javax.swing.JTextField;
import java.awt.Container;
import java.awt.Toolkit;
import java.awt.Dimension;
import java.awt.event.ActionEvent;
import java.awt.GridLayout;

public class DownloadFastaPanel extends JFrame implements Runnable {

    private UserFrame view;
    private JPanel panel, buttonPanel;
    private OpButton okButton, cancelButton;
    private JTextField proteinText;
    
    public DownloadFastaPanel (UserFrame view){
        
        super("Download fasta");

        Toolkit t = Toolkit.getDefaultToolkit();
        Dimension screensize = t.getScreenSize();
        double widthFrame = (screensize.getWidth() * 20.0) / 100.0;
        double heighFrame = (screensize.getHeight() * 45.0) / 500.0;

        this.view = view;
        panel = new JPanel(new GridLayout(2,1));
        buttonPanel = new JPanel();
        proteinText = new JTextField(100);
        proteinText.setPreferredSize(new Dimension(100, 25));
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

        /* Setup layout */
        setLocation((int) (view.getX() + (int) (view.getWidth() / 2)),
                (int) (view.getY() + (int) (view.getHeight() / 2)));
        setPreferredSize(new Dimension((int) widthFrame, (int) heighFrame));
        setResizable(true);
        
        buttonPanel.add(okButton);
        buttonPanel.add(cancelButton);
        
        panel.add(proteinText);
        panel.add(buttonPanel);

    }    
    
    public void run() {
        pack();
        Container ct = getContentPane();
        ct.add(panel);
        setVisible(true);
    }//run
    
    private void buttonEvent(boolean ok){
        if(ok){
            String proteinID = proteinText.getText();
            if(downloadFasta(proteinID)){
                view.getController().loadStructure(
                        Defs.path_prot + proteinID+".fa", Defs.TARGET, false);
                setVisible(false);
                dispose();
            }
        }else{
            setVisible(false);
            dispose();
            
        }
    }
    
    public boolean downloadFasta(String protID){

        InputStream url;
        boolean downloaded = false;
        protID = protID.split("\\.")[0];

        try{
            url = new URL(
                    "http://www.rcsb.org/pdb/download/downloadFile.do?fileFormat=FASTA&compression=NO&structureId=" 
                    + protID).openStream();
            Scanner fasta = new Scanner(url);

            BufferedWriter bw = new BufferedWriter(
                    new FileWriter(Defs.path_prot + protID + ".fa", true));

            //output file is prepared.

            while (fasta.hasNextLine()) {
                bw.write(fasta.nextLine() + "\n");
            }
            bw.flush();
            bw.close();
            
            downloaded = true;

        }catch (Exception e){
            System.out.println("File input error on:"+protID);
            downloaded = false;
        }
        
        return downloaded;
    }

}
