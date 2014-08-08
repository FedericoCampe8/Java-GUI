package jafatt;

import java.io.IOException;
import java.io.File;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import javax.swing.JFrame;
import java.awt.Dimension;
import java.awt.Container;
import java.awt.event.WindowEvent;
import java.awt.event.WindowAdapter;


public class FiascoSolver extends JFrame implements Runnable {
    
    UserFrame view;
    private InfoPanel ip;
    Process p;
    Boolean running = true;
    int exitValue;
    /* Setup */
    public FiascoSolver(UserFrame view){
        
        super("Fiasco");
         
        this.view = view;
        
        double w = view.getBounds().width;
        double h = view.getBounds().height;
        double widthFrame = (w * 70.0) / 100.0;  //960
        double heightFrame = (h * 50.0) / 100.0;  //540
        
        /* Setup layout */
        setPreferredSize(new Dimension((int)widthFrame, (int)heightFrame));
        setResizable(true);
        
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
        
        ip = new InfoPanel(false);        
        
    }//setup
    
    
    @Override
    public void run(){
       pack();
       setLocationRelativeTo(view);
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
