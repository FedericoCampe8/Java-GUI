package jafatt;

import java.io.FileOutputStream;
import java.io.IOException;
import java.io.BufferedWriter;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.io.FileWriter;
import java.io.File;
import javax.swing.JFrame;
import java.awt.Dimension;
import java.awt.Container;
import java.awt.event.WindowEvent;
import java.awt.event.WindowAdapter;


public class CocosSolver extends JFrame implements Runnable {
    
    UserFrame view;
    Process p;
    private InfoPanel ip;
    private Boolean running = true;
    private int exitValue;
    private int[] options;
    private String targetPath, cocosFile;
    private String solverPath;
    /* Setup */
    public CocosSolver(UserFrame view, int[] options,
            String targetPath, String cocosFile, boolean cuda){
        
        super("Cocos");
         
        this.view = view;
        this.options = options;
        this.targetPath = targetPath;
        this.cocosFile = cocosFile;
        
        if(cuda)
            solverPath = Defs.COCOS_GPU_PATH;
        else
            solverPath = Defs.COCOS_PATH;
        
        double w = view.getBounds().width;
        double h = view.getBounds().height;
        double widthFrame = (w * 70.0) / 100.0;  //960
        double heightFrame = (h * 50.0) / 100.0;  //540
        
        /* Setup layout */
        setPreferredSize(new Dimension((int)widthFrame, (int)heightFrame));
        setResizable(true);
        ip = new InfoPanel(false);
        
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
       Container ct = getContentPane();
       ct.add(ip);
       pack();
       setLocationRelativeTo(view);
       setVisible(true);
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
            process.directory(new File(solverPath));
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
           view.printStringLn("Solver has finished improperly");
           return;
       }
       String outputPath = Defs.PROTEINS_PATH + HeaderPdb.getProteinId() + ".cocos.pdb";
       view.initProteinLoaded(Defs.EXTRACTION);
       view.topMenuBar.openProteinItem.setEnabled(true);
       ok = view.getController().loadStructure(outputPath, Defs.EXTRACTION, true);
       
       if(ok){
           view.proteinLoaded();
       }else{
           /* Stop the progress bar */
           view.proteinNotLoaded();
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
            FileOutputStream cocosSh;
            cocosSh = new FileOutputStream(solverPath + "cocosScript.sh");
            cocosSh.write(cmd.getBytes());
            cocosSh.flush();
            cocosSh.close();
        }catch (IOException e) {
            view.printStringLn("Error: " + e);
        }
        
         try{
             Process permission = new ProcessBuilder("chmod", "+x", solverPath + "cocosScript.sh").start();
             permission.waitFor();
        }catch(Exception e){
            e.printStackTrace(System.out);
        }
    }
}
