package jafatt;

import javax.swing.JFrame;
import javax.swing.JPanel;
import java.awt.GridBagLayout;
import java.awt.Color;
import javax.swing.JLabel;
import javax.swing.ImageIcon;
import java.awt.Container;
import java.awt.Dimension;
import java.awt.Toolkit;

public class LoadingPanel extends JPanel {
    JLabel imageLabel = new JLabel();

    public LoadingPanel(UserFrame view) {
        
        Toolkit t = Toolkit.getDefaultToolkit();
        Dimension screensize = t.getScreenSize();
        double widthFrame = (screensize.getWidth() * 50.0) / 100.0;
        double heighFrame = (screensize.getHeight() * 75.0) / 100.0;
        
        /* Setup layout */
        setLocation((int)(view.getX() + (int)(view.getWidth()/10)),
                    (int)(view.getY() + (int)(view.getHeight()/10)));
        setPreferredSize(new Dimension((int)widthFrame, (int)heighFrame));
        
        setBackground(Color.black);
        
        setLayout(new GridBagLayout());
        String dir = System.getProperty("user.dir");
        
        ImageIcon ii = new ImageIcon(dir 
                + Utilities.getSeparator() 
                + "images" 
                + Utilities.getSeparator()
                + "load.gif");
        
        imageLabel.setIcon(ii);
        add(imageLabel);
    }
}

