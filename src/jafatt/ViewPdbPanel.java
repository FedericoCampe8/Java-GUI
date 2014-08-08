package jafatt;

import java.io.FileNotFoundException;
import java.io.File;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.Container;
import java.awt.event.WindowEvent;
import java.awt.event.WindowAdapter;
import java.util.Scanner;
import javax.swing.JFrame;
import javax.swing.JTextPane;
import javax.swing.JScrollPane;
import javax.swing.text.SimpleAttributeSet;
import javax.swing.text.StyleConstants;
import javax.swing.text.DefaultStyledDocument;

public class ViewPdbPanel extends JFrame implements Runnable {

    UserFrame view;
    private JTextPane ip;
    private JScrollPane scroll;
    private String pdb;

    /* Setup */
    public ViewPdbPanel(UserFrame view, String proteinPath) {

        super("Pdb - " + proteinPath);

        this.view = view;

        double w = view.getBounds().width;
        double h = view.getBounds().height;
        double widthFrame = (w * 70.0) / 100.0;  //960
        double heightFrame = (h * 50.0) / 100.0;  //540

        /* Setup layout */
        setPreferredSize(new Dimension((int) widthFrame, (int) heightFrame));
        setResizable(true);

        DefaultStyledDocument myDocument = new DefaultStyledDocument();
        SimpleAttributeSet attributeSet = new SimpleAttributeSet();
        StyleConstants.setBold(attributeSet, true);
        StyleConstants.setAlignment(attributeSet, StyleConstants.ALIGN_JUSTIFIED);
        myDocument.setParagraphAttributes(0, 1, attributeSet, true);

        ip = new JTextPane(myDocument);
        ip.setBackground(Color.WHITE);
        ip.setForeground(Color.BLACK);
        ip.setEditable(false);

        addWindowListener(new WindowAdapter() {

            @Override
            public void windowClosing(WindowEvent e) {
                setVisible(false);
                dispose();
            }
        });
        try {
            pdb = new Scanner(new File(proteinPath)).useDelimiter("\\Z").next();
        } catch (FileNotFoundException ex) {
        }
        ip.setText(pdb);
        
        scroll = new JScrollPane(ip);

    }//setup

    @Override
    public void run() {
        Container ct = getContentPane();
        ct.add(scroll);
        update();
        setVisible(true);
    }//run

    private void update() {
        pack();
        setLocationRelativeTo(view);
    }
}