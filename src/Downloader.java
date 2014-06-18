package jafatt;

import java.util.Scanner;
import java.net.URL;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JOptionPane;
import java.awt.GridLayout;
import java.io.*;

public class Downloader {


    public static int[] downloadProtein(String protID, boolean pdbSelected, boolean fastaSelected) {

        InputStream url;
        int[] downloaded = {0, 0};
        protID = protID.split("\\.")[0];

        if (pdbSelected) {
            String protPath = Defs.PROTEINS_PATH + protID.toUpperCase();
            try {
                if (new File(protPath + ".pdb").isFile()) {
                    OverwritePanel op = new OverwritePanel(protID, pdbSelected);
                    int result = JOptionPane.showConfirmDialog(null, op, "Fiasco",
                            JOptionPane.OK_CANCEL_OPTION, JOptionPane.PLAIN_MESSAGE);
                    if (result == JOptionPane.CANCEL_OPTION) {
                        downloaded[0] = 1;
                        return downloaded;
                    } else {
                        try {
                            FileOutputStream pdb = new FileOutputStream(
                                    protPath + ".pdb");
                            pdb.close();
                        } catch (IOException e) {
                        }
                    }
                }
                url = new URL(
                        "http://www.rcsb.org/pdb/download/downloadFile.do?fileFormat=PDB&compression=NO&structureId="
                        + protID).openStream();
                Scanner pdbScan = new Scanner(url);

                BufferedWriter bw = new BufferedWriter(
                        new FileWriter(protPath + ".pdb", true));

                //output file is prepared.

                while (pdbScan.hasNextLine()) {
                    bw.write(pdbScan.nextLine() + "\n");
                }
                bw.flush();
                bw.close();

                downloaded[0] = 1;

            } catch (Exception e) {
            }

        }
        if (fastaSelected) {

            String protPath = Defs.PROTEINS_PATH + protID;
            try {

                if (new File(protPath + ".fasta").isFile()) {
                    OverwritePanel op = new OverwritePanel(protID, !fastaSelected);
                    int result = JOptionPane.showConfirmDialog(null, op, "Fiasco",
                            JOptionPane.OK_CANCEL_OPTION, JOptionPane.PLAIN_MESSAGE);
                    if (result == JOptionPane.CANCEL_OPTION) {
                        downloaded[1] = 1;
                        return downloaded;
                    } else {
                        try {
                            FileOutputStream fasta = new FileOutputStream(
                                    protPath + ".fasta");
                            fasta.close();
                        } catch (IOException e) {
                        }
                    }
                }

                url = new URL(
                        "http://www.rcsb.org/pdb/download/downloadFile.do?fileFormat=FASTA&compression=NO&structureId="
                        + protID).openStream();
                Scanner fastaScan = new Scanner(url);

                BufferedWriter bw = new BufferedWriter(
                        new FileWriter(protPath + ".fasta", true));

                //output file is prepared.

                while (fastaScan.hasNextLine()) {
                    bw.write(fastaScan.nextLine() + "\n");
                }
                bw.flush();
                bw.close();

                downloaded[1] = 1;

            } catch (Exception e) {
            }
        }
        
        return downloaded;
    }

    

    private static class OverwritePanel extends JPanel {

        public OverwritePanel(String protID, boolean isPdb) {
            setLayout(new GridLayout(3, 2));
            if (isPdb) {
                add(new JLabel(protID.toUpperCase() + ".pdb already exist in folder"));
            } else {
                add(new JLabel(protID + ".fasta already exist in folder"));
            }
            add(new JLabel(Defs.PROTEINS_PATH));
            add(new JLabel("Download it anyway?"));
        }
    }//ConfirmtPanel  
}
