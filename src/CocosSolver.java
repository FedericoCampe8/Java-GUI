package jafatt;

import java.io.*;
import javax.swing.JFrame;
import java.awt.Dimension;
import java.awt.Toolkit;
import java.awt.Container;
import java.awt.event.WindowEvent;
import java.awt.event.WindowListener;


public class CocosSolver extends JFrame implements Runnable, WindowListener {
    
    UserFrame view;
    private InfoPanel ip;
    Process p;
    Boolean running = true;
    int exitValue;
    int[] options;
    String targetPath;
    Boolean inCocos;
    /* Setup */
    public CocosSolver(UserFrame view, int[] options, String targetPath, boolean inCocos){
        
        super("Cocos");
         
        this.view = view;
        this.options = options;
        this.targetPath = targetPath;
        this.inCocos = inCocos;
        
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
       Container ct = getContentPane();
       ct.add(ip);
       setVisible(true);
       boolean ok = false;
       String line = "";
       exitValue = -1;
       buildScript(options, inCocos);
       try{
            FileOutputStream cocosSh = new FileOutputStream(Defs.path_prot +
                HeaderPdb.getProteinId() + ".cocos.pdb");
            cocosSh.close();
        }catch (IOException e) {
            view.printStringLn("Error: " + e);
        }
       try{
            ProcessBuilder process = new ProcessBuilder("./cocosScript.sh");
            process.directory(new File(Defs.path_cocos));
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
       if(exitValue != 0){
           view.printStringLn("Solver finished improperly");
           return;
       }
       String outputPath = Defs.path_prot + HeaderPdb.getProteinId() + ".cocos.pdb";
       view.initProtinLoaded();
       ok = view.getController().loadStructure(outputPath, Defs.EXTRACTION, true);
       
       if(ok){
           view.proteinLoaded();
       }else{
           /* Stop the progress bar */
           view.barPanel.stop();
           view.printStringLn("Error on loading the protein");
       }
       this.dispose();
       /*
       view.printStringLn("Solver finished");
       ((AssemblingPanel)view.getPanel(Defs.ASSEMBLING)).loadOutput();
       */
    }
    
    public void buildScript(int[] options, boolean inCocos){
        //controls of the values TO DO
        String cmd = "./cocos -i ";
        //input path
        cmd = cmd + targetPath + " ";
        //output file targetID.cocos.pdb
        //HeaderPdb.targetId(targetPath);
        cmd = cmd + "-o " + Defs.path_prot +
                HeaderPdb.getProteinId() + ".cocos.pdb";
        if (!inCocos){
            cmd = cmd + " -a";
        }
        if (options[Defs.MONTECARLO_SAMPLING] != 0)
            cmd = cmd + " -c " + options[Defs.MONTECARLO_SAMPLING];
        if (options[Defs.GIBBS_SAMPLING] != 0)
            cmd = cmd + " -g " + options[Defs.GIBBS_SAMPLING];
        if (options[Defs.RMSD_OPTION] != 0)
            cmd = cmd + " -r";
        if (options[Defs.VERBOSE_OPTION] != 0)
            cmd = cmd + " -v";
        if (options[Defs.GIBBS_OPTION] != 0)
            cmd = cmd + " -q";
        if (options[Defs.CGC_OPTION] != 0)
            cmd = cmd + " -e";
        try{
            FileOutputStream cocosSh = new FileOutputStream(Defs.path_cocos + "/cocosScript.sh");
            cocosSh.write(cmd.getBytes());
            cocosSh.flush();
            cocosSh.close();
        }catch (IOException e) {
            view.printStringLn("Error: " + e);
        }
    }
    
    @Override
    public void windowClosed(WindowEvent e) {}
    @Override
    public void windowClosing(WindowEvent e) {
        running = false;
        p.destroy();
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
