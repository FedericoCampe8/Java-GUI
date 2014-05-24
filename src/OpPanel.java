package jafatt;

import java.awt.Color;
import java.awt.FlowLayout;
import java.awt.GridLayout;
import javax.swing.JPanel;




public abstract class OpPanel extends JPanel{
    
    protected JPanel topA;
    protected JPanel topB;
    
    public OpPanel(){
        setup2();
    }
    
    /* Set the layout of the panel */
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
    
    private void setup2(){
        setBorder(new javax.swing.border.SoftBevelBorder(javax.swing.border.BevelBorder.RAISED));
        setBackground(Color.BLACK);
        setLayout(new GridLayout(2, 1));
        topA = new JPanel();
        topA.setBackground(Color.BLACK);
        topA.setLayout(new FlowLayout());
        topB = new JPanel();
        topB.setBackground(Color.BLACK);
        topB.setLayout(new FlowLayout());
        add(topA);
        add(topB); 
    }//setup
    
    /* Operational Buttons */
    
}//OpPanel
