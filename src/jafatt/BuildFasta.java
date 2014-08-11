package jafatt;

import javax.swing.JFrame;
import javax.swing.JPanel;
import javax.swing.JLabel;
import javax.swing.BorderFactory;
import javax.swing.JScrollPane;
import javax.swing.ScrollPaneConstants;
import javax.swing.JOptionPane;
import java.awt.Container;
import java.awt.event.ActionEvent;
import java.awt.GridBagLayout;
import java.awt.GridBagConstraints;
import java.awt.BorderLayout;
import java.awt.GridLayout;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.File;
import java.util.Arrays;

public class BuildFasta extends JFrame implements Runnable {

    private UserFrame view;
    private JPanel panel, subPanel;
    private JPanel buttonPanel;
    private JScrollPane scroll;
    private OpButton okButton, cancelButton;
    private HintTextField targetName, fastaSequence;
    private GridBagConstraints c;

    public BuildFasta(UserFrame view) {

        super("Fiasco - Build Sequence");

        this.view = view;

        panel = new JPanel(new BorderLayout());
        subPanel = new JPanel(new GridBagLayout());
        buttonPanel = new JPanel();
        targetName = new HintTextField("Protein Name ", 10);
        targetName.setHintText("PROTEIN");
        fastaSequence = new HintTextField(" Type a sequence of Amino Acids ", 50);

        okButton = new OpButton("Ok", "") {

            @Override
            public void buttonEvent(ActionEvent evt) {
                buttonEvent(true);
            }
        };

        cancelButton = new OpButton("Cancel", "") {

            @Override
            public void buttonEvent(ActionEvent evt) {
                buttonEvent(false);
            }
        };

        c = new GridBagConstraints();
        c.gridx = 0;
        c.gridy = 0;
        c.anchor = GridBagConstraints.EAST;
        subPanel.add(new JLabel("Protein Name: "), c);
        c.gridx = 1;
        c.anchor = GridBagConstraints.WEST;
        subPanel.add(targetName, c);
        c.gridx = 0;
        c.gridy = 1;
        subPanel.add(new JLabel("Fasta Sequence: "), c);
        c.gridx = 1;
        subPanel.add(fastaSequence, c);

        subPanel.setBorder(BorderFactory.createTitledBorder("Build a Sequence"));

        scroll = new JScrollPane(subPanel,
                ScrollPaneConstants.VERTICAL_SCROLLBAR_AS_NEEDED,
                ScrollPaneConstants.HORIZONTAL_SCROLLBAR_AS_NEEDED);

        buttonPanel.add(okButton);
        buttonPanel.add(cancelButton);

        panel.add(scroll, BorderLayout.CENTER);
        panel.add(buttonPanel, BorderLayout.SOUTH);

    }

    @Override
    public void run() {
        Container ct = getContentPane();
        ct.add(panel);
        update();
        setVisible(true);
        okButton.requestFocus();
    }//run

    private void update() {
        pack();
        setLocationRelativeTo(view);
    }

    private void buttonEvent(boolean ok) {

        String fasta;
        String[] AA = {"A","C","D","E","F","G","H","I","K","L","M","N","P","Q","R","S","T","V","W","Y"};

        if (ok) {

            String protID = targetName.getText().toLowerCase();
            if (protID.equals("")) {
                view.printStringLn("Set a protein name");
                return;
            }
            String sequence = fastaSequence.getText().toUpperCase();
            if (sequence.equals("")) {
                view.printStringLn("Set an Amino Acid Sequence");
                return;
            }
            String checkedSequence = "";
            for (int i = 0; i < sequence.length(); i++) {
                if(!Arrays.asList(AA).contains(String.valueOf(sequence.charAt(i)))){
                    view.printStringLn(sequence.charAt(i) + " Invalid Amino Acid, char skipped.");
                }else{
                    checkedSequence += sequence.charAt(i);
                }
            }

            String protPath = Defs.PROTEINS_PATH + protID;
            try {

                if (new File(protPath + ".fasta").isFile()) {
                    OverwritePanel op = new OverwritePanel(protID);
                    int result = JOptionPane.showConfirmDialog(view, op, "Fiasco",
                            JOptionPane.OK_CANCEL_OPTION, JOptionPane.PLAIN_MESSAGE);
                    if (result == JOptionPane.CANCEL_OPTION) {
                    } else {
                        try {
                            FileOutputStream output = new FileOutputStream(
                                    protPath + ".fasta");
                            output.close();
                        } catch (IOException e) {
                        }
                    }
                }
                fasta = ">" + protID.toUpperCase() + ":A|PDBID|CHAIN|SEQUENCE\n";
                fasta += checkedSequence + "\n";

                BufferedWriter bw = new BufferedWriter(
                        new FileWriter(protPath + ".fasta", true));
                bw.write(fasta);
                bw.flush();
                bw.close();

            } catch (Exception e) {}
            
            boolean loaded;
            loaded = view.getController().loadStructure(
                    Defs.PROTEINS_PATH + protID + ".fasta", Defs.TARGET, false);
            if (loaded) {
                view.getViewActions().loadATarget = true;
                view.targetLoaded();
            } else {
                view.printStringLn("Error on loading the target");
            }
            setVisible(false);
            dispose();

        } else {
            setVisible(false);
            dispose();
        }
    }

    private static class OverwritePanel extends JPanel {

        public OverwritePanel(String protID) {
            setLayout(new GridLayout(3, 2));
            add(new JLabel(protID + ".fasta already exist in folder"));
            add(new JLabel(Defs.PROTEINS_PATH));
            add(new JLabel("Save it anyway?"));
        }
    }//ConfirmtPanel  
}
