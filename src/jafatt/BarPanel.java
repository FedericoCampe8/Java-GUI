package jafatt;

import java.awt.Container;
import java.awt.Dimension;
import java.awt.GridLayout;
import java.awt.event.WindowEvent;
import javax.swing.ImageIcon;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JProgressBar;

public class BarPanel extends JFrame implements Runnable{
    
    private UserFrame view;
    private JPanel inPanel;
    
    public BarPanel(String title, UserFrame view){
        setup(title, view);
    }
    
    /* Setup */
    private void setup(String title, UserFrame view){
        
        this.view = view;
        
        setIconImage(new ImageIcon(view.frameIcon).getImage());
        
        JProgressBar transferBar;
        JLabel progressLabel;
        
        double w = view.getBounds().width;
        double h = view.getBounds().height;
        double widthFrame = (w * 35.0) / 100.0;  //960
        double heightFrame = (h * 10.0) / 100.0;  //540
        
        inPanel = new JPanel(new GridLayout(2, 1));
        transferBar = new JProgressBar();
        transferBar.setIndeterminate(true);
        progressLabel = new JLabel(title);
        
        setPreferredSize(new Dimension((int)widthFrame, (int)heightFrame));
        setResizable(true);

        /* Add the progress bar */
        inPanel.add(progressLabel);
        inPanel.add(transferBar);
        
    }//setup
    
    @Override
    public void run(){
        Container ct = getContentPane();
        ct.add(inPanel);
        pack();
        setLocationRelativeTo(view);
        setVisible(true);
        //getRootPane().revalidate();
       
    }//run
    
    public void stop(){
        WindowEvent close = new WindowEvent(this, WindowEvent.WINDOW_CLOSING);
        dispatchEvent(close);
    }//stop
    
}//TransferBarPanel
