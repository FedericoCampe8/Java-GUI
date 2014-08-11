package jafatt;

import java.io.IOException;
import java.io.File;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import javax.swing.JFrame;
import javax.swing.JFrame;
import javax.swing.JPanel;
import javax.swing.JLabel;
import javax.swing.JProgressBar;
import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.Container;
import java.awt.FlowLayout;
import java.awt.event.WindowEvent;
import java.awt.event.WindowAdapter;
import javax.swing.ImageIcon;


public class FiascoSolver extends JFrame implements Runnable {
    
    UserFrame view;
    private InfoPanel ip;
    private JPanel inPanel, inSubPanel;
    private JProgressBar runningBar;
    private JLabel progressLabel;
    Process p;
    Boolean running = true;
    int exitValue;
    /* Setup */
    public FiascoSolver(UserFrame view){
        
        super("Fiasco");
         
        this.view = view;
        
        setIconImage(new ImageIcon(view.frameIcon).getImage());
        
        double w = view.getBounds().width;
        double h = view.getBounds().height;
        double widthFrame = (w * 70.0) / 100.0;  //960
        double heightFrame = (h * 50.0) / 100.0;  //540
        
        /* Setup layout */
        setPreferredSize(new Dimension((int)widthFrame, (int)heightFrame));
        setResizable(true);
        
        ip = new InfoPanel(false);
        inPanel = new JPanel(new BorderLayout());
        
        inSubPanel = new JPanel(new FlowLayout(FlowLayout.LEFT));
        runningBar = new JProgressBar();
        runningBar.setIndeterminate(true);
        progressLabel = new JLabel("Running...  ");
        
        runningBar.setForeground(Color.GREEN);
        runningBar.setBackground(Color.BLACK);
        
        setPreferredSize(new Dimension((int)widthFrame, (int)heightFrame));
        setResizable(true);

        inSubPanel.add(progressLabel);
        inSubPanel.add(runningBar);
        
        inPanel.add(ip, BorderLayout.CENTER);
        inPanel.add(inSubPanel, BorderLayout.SOUTH);
        
        /* Setup layout */
        addWindowListener(new WindowAdapter() {

            @Override
            public void windowClosing(WindowEvent e) {
                running = false;
                p.destroy();
                setVisible(false);
                dispose();
            }
        });    
        
    }//setup
    
    
    @Override
    public void run(){
       pack();
       setLocationRelativeTo(view);
       Container ct = getContentPane();
       ct.add(inPanel);
       setVisible(true);
       String line = "";
       exitValue = -1;
       try{
            ProcessBuilder process = new ProcessBuilder("./solve.sh");
            process.directory(new File(Defs.FIASCO_PATH));
            p = process.start();
            //p.waitFor();
            BufferedReader reader = new BufferedReader(new InputStreamReader(p.getInputStream()));
            try {
                if(running){
                    while ((line = reader.readLine()) != null){
                        ip.printStringLn(line);
                    }
                }
            }catch (IOException e) {
            }
            p.destroy();
            p.waitFor();
            exitValue = p.exitValue();
            running = false;
       }catch(Exception e){
            e.printStackTrace(System.out);
       }
        //this.dispose();
       if(exitValue != 0){
           view.printStringLn("Solver has finished improperly");
           return;
       }
       /*
       view.printStringLn("Solver has finished");
        */
       dispose();
       view.initProteinLoaded(Defs.OUTPUT);
       ((AssemblingPanel)view.getPanel(Defs.ASSEMBLING)).loadOutput();
        
    }
    
}
