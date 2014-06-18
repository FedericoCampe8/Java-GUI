package jafatt;

import java.io.*;
import javax.swing.JFrame;
import java.awt.Dimension;
import java.awt.Toolkit;
import java.awt.Container;
import java.awt.event.WindowEvent;
import java.awt.event.WindowListener;


public class FiascoSolver extends JFrame implements Runnable, WindowListener {
    
    UserFrame view;
    private InfoPanel ip;
    Process p;
    Boolean running = true;
    int exitValue;
    /* Setup */
    public FiascoSolver(UserFrame view){
        
        super("Fiasco");
         
        this.view = view;
        
        Toolkit t = Toolkit.getDefaultToolkit();
        Dimension screensize = t.getScreenSize();
        double widthFrame = 1200;
        double heightFrame = 450;
        
        /* Setup layout */
        setPreferredSize(new Dimension((int)widthFrame, (int)heightFrame));
        int x = (int) ((screensize.width / 2) - (widthFrame / 2));
        int y = (int) ((screensize.height / 2) - (heightFrame / 2));    
        this.setLocation(x, y);
        setResizable(true);
        addWindowListener(this);
        
        ip = new InfoPanel(false);        
        
    }//setup
    
    
    @Override
    public void run(){
       pack();
       setLocationRelativeTo(null);
       Container ct = getContentPane();
       ct.add(ip);
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
           view.printStringLn("Solver finished improperly");
           return;
       }

       view.printStringLn("Solver finished");
       ((AssemblingPanel)view.getPanel(Defs.ASSEMBLING)).loadOutput();
       
       this.dispose();
        
    }
    
    @Override
    public void windowClosed(WindowEvent e) {}
    @Override
    public void windowClosing(WindowEvent e) {
        running = false;
        p.destroy();
        setVisible(false);
        dispose();
    }
    @Override
    public void windowOpened(WindowEvent e) {}
    @Override
    public void windowIconified(WindowEvent e) {}
    @Override
    public void windowDeiconified(WindowEvent e) {}
    @Override
    public void windowActivated(WindowEvent e) {}
    @Override
    public void windowDeactivated(WindowEvent e) {}
    
}
