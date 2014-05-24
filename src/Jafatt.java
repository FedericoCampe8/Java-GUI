package jafatt;

import java.io.PrintStream;
import javax.swing.UIManager;

public class Jafatt {
    
    /* Controller */
    static Controller ctr;
    
    /* View */
    static UserFrame frame;
    
    /* Model */
    static ProteinModel protModel;
    
    public static void main(String[] args){
        String os = System.getProperty("os.name").toLowerCase();
        if(os.indexOf("win") >= 0)
            try{
                UIManager.setLookAndFeel("com.sun.java.swing.plaf.windows.WindowsLookAndFeel");
            }catch(Exception e){}
        else if(os.indexOf("mac") >= 0)
            try{
                UIManager.setLookAndFeel("com.sun.java.swing.plaf.mac.MacLookAndFeel");
            }catch(Exception e){}
        else if(os.indexOf("nix") >= 0)
            try{
                UIManager.setLookAndFeel("com.sun.java.swing.plaf.motif.MotifLookAndFeel");
            }catch(Exception e){}
        
        /* Test some methods and debug*/
        //test();
        
        //HeaderPdb.info("/home/matteo/NetBeansProjects/jafatt/solver/proteins/pr.pdb");
        //HeaderPdb.info("/home/matteo/NetBeansProjects/jafatt/solver/proteins/2G33.pdb");
        //HeaderPdb.getProteinIdTarget("/home/matteo/NetBeansProjects/jafatt/solver/proteins/1ZDD.fasta.txt");
        
        /* Initialize components */
        initialize();
    }//main
    
    /* Initialize components */
    private static void initialize(){
       ctr = new Controller();
       frame = new UserFrame();
       protModel = new ProteinModel();
       setup();
       frame.pack();
       frame.setVisible(true);
       frame.setExtendedState(UserFrame.MAXIMIZED_BOTH); 
    }//initialize
    
    private static void setup(){
        ctr.setViewMod( frame, protModel );
        frame.setConMod( ctr, protModel );
        protModel.setView( frame );
    }//setup
    
    /* This is necessary to redirect the output to application */

    static{
        final PrintStream currentOut = System.out;
        PrintStream newOut = new PrintStream(currentOut){
            @Override
            public void println(String str){
                if(ctr != null){
                    
                    /* Error information */
                    if(str.startsWith("eval ERROR")){
                        ctr.printStringLn(str);
                    }
                    
                    /* Parse Jmol output */
                    String[] outParsed = StringParser.parse(str);
                    
                    if (!outParsed[0].equals("") && !outParsed[0].equals("MEASURE"))
                        ctr.usePickedInfo(outParsed);
                    else if(outParsed[0].equals("MEASURE")){
                        //ctr.setMeasure(outParsed);
                        ctr.printStringLn(outParsed[0]);
                    }else{
                        print("\n");
                       // ctr.printString(outParsed[1]);
                    }
                }
                
                /* Print the string anyway */
                print(str + "\n");
            }
        };
       System.setOut(newOut);
    }//override of println method
    
    /* Method for debugging */
    private static void test(){
        String prova = Utilities.replaceSubstring("___CIAO___ABCabcdefghi2", "!!", 3);
        prova = Utilities.replaceSubstring(prova, "^^", 11);
        System.out.println(prova);   
    }//test
}
