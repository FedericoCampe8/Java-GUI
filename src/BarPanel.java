package jafatt;

import java.awt.Container;
import java.awt.Dimension;
import java.awt.GridLayout;
import java.awt.Toolkit;
import java.awt.event.WindowEvent;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JProgressBar;

public class BarPanel extends JFrame implements Runnable{
    
    private UserFrame view;
    private JPanel inPanel;
    private JProgressBar transferBar;
    private JLabel progressLabel;
    
    public BarPanel(String title, UserFrame view){
        setup(title, view);
    }
    
    /* Setup */
    private void setup(String title, UserFrame view){
        Toolkit t = Toolkit.getDefaultToolkit();
        Dimension screensize = t.getScreenSize();
        double widthFrame = (screensize.getWidth() * 22.0) / 100.0;
        double heighFrame = (screensize.getHeight() * 10.0) / 100.0;
        
        this.view = view;
        inPanel = new JPanel(new GridLayout(2, 1));
        transferBar = new JProgressBar();
        transferBar.setIndeterminate(true);
        progressLabel = new JLabel(title);
        
        /* Setup layout */
        setLocation((int)(view.getX() + (int)(view.getWidth()/2)),
                    (int)(view.getY() + (int)(view.getHeight()/2)));
        setPreferredSize(new Dimension((int)widthFrame, (int)heighFrame));
        setResizable(true);
        
        /* Add the progress bar */
        inPanel.add(progressLabel);
        inPanel.add(transferBar);
        
    }//setup
    
    @Override
    public void run(){
        pack();
        Container ct = getContentPane();
        ct.add(inPanel);
        setVisible(true);
    }//run
    
    public void stop(){
        WindowEvent close = new WindowEvent(this, WindowEvent.WINDOW_CLOSING);
        dispatchEvent(close);
    }//stop
    
}//TransferBarPanel
