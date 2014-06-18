package jafatt;

import java.awt.Color;
import java.awt.FlowLayout;
import java.awt.GridLayout;
import java.awt.GridBagLayout;
import java.awt.GridBagConstraints;
import java.awt.BorderLayout;
import java.awt.Insets;
import javax.swing.JPanel;

public abstract class OpPanel extends JPanel {

    protected JPanel main;
    protected JPanel tool;
    protected JPanel expand;
    protected GridBagConstraints c;

    public OpPanel(boolean expandable) {
        setup(expandable);
    }

    /* Set the layout of the panel */
    /*
    private void setup(){
    setBorder(new javax.swing.border.SoftBevelBorder(javax.swing.border.BevelBorder.RAISED));
    setBackground(Defs.BACKGROUNDCOLOR);
    setLayout(new GridLayout(2, 1));
    topA = new JPanel();
    topA.setBackground(Defs.INNERCOLOR);
    topA.setLayout(new FlowLayout());
    topB = new JPanel();
    topB.setBackground(Defs.INNERCOLOR);
    topB.setLayout(new FlowLayout());
    add(topA);
    add(topB);
    }//setup
     */
    private void setup(boolean expandable) {

        c = new GridBagConstraints();
        setBorder(new javax.swing.border.SoftBevelBorder(javax.swing.border.BevelBorder.RAISED));
        setBackground(Color.BLACK);
        setLayout(new GridBagLayout());
        main = new JPanel(new FlowLayout());
        main.setBackground(Color.BLACK);
        c.fill = GridBagConstraints.VERTICAL;
        c.gridx = 0;
        c.gridy = 0;

        if (expandable) {

            expand = new JPanel(new BorderLayout());
            expand.setBackground(Color.BLACK);
            tool = new JPanel(new FlowLayout());
            tool.setBackground(Color.BLACK);
            add(expand, c);
            c.gridy = 1;
            add(main, c);
        } else {
            c.insets = new Insets(6,0,0,0);
            add(main, c);
        }

    }
    /*
    private void setup1(){
    setBorder(new javax.swing.border.SoftBevelBorder(javax.swing.border.BevelBorder.RAISED));
    setBackground(Color.BLACK);
    setLayout(new GridLayout(1, 1));
    topA = new JPanel();
    topA.setBackground(Color.BLACK);
    topA.setLayout(new FlowLayout());
    add(topA);
    //add(topB); 
    }//setup
    
    private void setup2(){
    setBorder(new javax.swing.border.SoftBevelBorder(javax.swing.border.BevelBorder.RAISED));
    setBackground(Color.BLACK);
    setLayout(new BorderLayout());
    topA = new JPanel();
    topA.setBackground(Color.BLACK);
    topA.setLayout(new FlowLayout());
    topB = new JPanel();
    topB.setBackground(Color.BLACK);
    topB.setLayout(new FlowLayout());
    add(topA);
    }//setup
    
    /* Operational Buttons */
}//OpPanel
/*
setBorder(new javax.swing.border.SoftBevelBorder(javax.swing.border.BevelBorder.RAISED));
setBackground(Color.BLACK);
setLayout(new BorderLayout());
expand = new JPanel(new BorderLayout());
expand.setBackground(Color.BLACK);
mainPanel = new JPanel(new GridBagLayout());
mainPanel.setBackground(Color.BLACK);
topA = new JPanel(new FlowLayout());
topA.setBackground(Color.BLACK);
topB = new JPanel(new FlowLayout());
topB.setBackground(Color.BLACK);
c.fill = GridBagConstraints.VERTICAL;
c.gridx = 0; c.gridy = 0;
mainPanel.add(topA);
c.gridy = 1;
mainPanel.add(topB);
add(expand, BorderLayout.NORTH);
add(mainPanel, BorderLayout.CENTER);
 */
            //add(topA);