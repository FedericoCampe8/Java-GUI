package jafatt;

import java.awt.Dimension;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import javax.swing.JButton;
import javax.swing.ImageIcon;


public abstract class OpButton extends JButton{
    
    public OpButton(String buttonName, String desc){
        super(buttonName);
        setupButton(desc);
    }

    public OpButton(ImageIcon icon, String desc) {
        super(icon);
        setupButton(desc);
    }

    /* Set button */
    private void setupButton(String desc) {
        String os = System.getProperty("os.name").toLowerCase();
        if ((os.indexOf("win") >= 0) || (os.indexOf("nix") >= 0)) {
            setBorder(new javax.swing.border.SoftBevelBorder(javax.swing.border.BevelBorder.RAISED));
        }
        setBackground(Defs.INNERCOLOR);
        setPreferredSize(new Dimension(140, 25));
        if (!desc.equals("")) {
            setToolTipText(desc);
        }
        addActionListener(new ActionListener() {

            @Override
            public void actionPerformed(ActionEvent evt) {
                buttonEvent(evt);
            }
        });
    }//setupButton

    public abstract void buttonEvent(ActionEvent evt);

}//OpButton