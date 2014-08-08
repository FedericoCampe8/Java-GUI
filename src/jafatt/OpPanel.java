package jafatt;

import java.awt.Color;
import java.awt.FlowLayout;
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
}//OpPanel