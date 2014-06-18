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
    Process p;
    private InfoPanel ip;
    private Boolean running = true;
    private int exitValue;
    private int[] options;
    private String targetPath, cocosFile;
    /* Setup */
    public CocosSolver(UserFrame view, int[] options, String targetPath, String cocosFile){
        
        super("Cocos");
         
        this.view = view;
        this.options = options;
        this.targetPath = targetPath;
        this.cocosFile = cocosFile;
        
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
        ip = new InfoPanel(false);
        
        addWindowListener(this);
        
    }//setup
    
    
    @Override
    public void run(){
       pack();
       Container ct = getContentPane();
       ct.add(ip);
       setVisible(true);
       setLocationRelativeTo(null);
       boolean ok = false;
       String line = "";
       exitValue = -1;
       buildScript(options);
       try{
            FileOutputStream cocosPdb = new FileOutputStream(Defs.PROTEINS_PATH +
                HeaderPdb.getProteinId() + ".cocos.pdb");
            cocosPdb.close();
        }catch (IOException e) {
            view.printStringLn("Error: " + e);
        }
       try{
            ProcessBuilder process = new ProcessBuilder("./cocosScript.sh");
            process.directory(new File(Defs.COCOS_PATH));
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
       String outputPath = Defs.PROTEINS_PATH + HeaderPdb.getProteinId() + ".cocos.pdb";
       view.initProteinLoaded();
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
    
    public void buildScript(int[] options){
        
        String cmd = "./cocos -i ";
        
        if(options[Defs.FASTA_OPTION]==0){
            
            //Delete any in.cocos file existing
            try {
                FileOutputStream cocosStream = new FileOutputStream(Defs.PROTEINS_PATH 
                        + HeaderPdb.getProteinId() + ".in.cocos");
                cocosStream.close();
                
                PrintWriter cocos = new PrintWriter(new BufferedWriter(
                        new FileWriter(Defs.PROTEINS_PATH + HeaderPdb.getProteinId()
                        + ".in.cocos", true)));
                cocos.print("TARGET_PROT " + Defs.PROTEINS_PATH + HeaderPdb.getProteinId());
                cocos.print("\n");
                cocos.print("KNOWN_PROT  " + Defs.PROTEINS_PATH + HeaderPdb.getProteinId());
                cocos.print("\n");
                cocos.print(cocosFile);
                cocos.close();
            } catch (IOException e) {
                //print exception
            }
            
            cmd = cmd + Defs.PROTEINS_PATH + HeaderPdb.getProteinId() + ".in.cocos ";
        }else{
            cmd = cmd + targetPath + " ";
        }
        
        //output file targetID.cocos.pdb
        //HeaderPdb.targetId(targetPath);
        cmd = cmd + "-o " + Defs.PROTEINS_PATH +
                HeaderPdb.getProteinId() + ".cocos.pdb";
        if (options[Defs.FASTA_OPTION]!=0)
            cmd = cmd + " -a";
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
            FileOutputStream cocosSh = new FileOutputStream(Defs.COCOS_PATH 
                    + "cocosScript.sh");
            cocosSh.write(cmd.getBytes());
            cocosSh.flush();
            cocosSh.close();
        }catch (IOException e) {
            view.printStringLn("Error: " + e);
        }
        
         try{
            Process permission = new ProcessBuilder("chmod", "+x",
                    Defs.COCOS_PATH + "cocosScript.sh").start();
            permission.waitFor();
        }catch(Exception e){
            e.printStackTrace(System.out);
        }
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
