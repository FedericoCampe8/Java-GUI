package jafatt;

import javax.swing.JPanel;
import org.jmol.api.JmolViewer;
import org.jmol.api.JmolAdapter;
import java.util.regex.Pattern;
import java.util.regex.Matcher;

public class Rmsd extends JPanel {
        
    private JmolViewer viewer;
    private JmolAdapter adapter;
    
    public Rmsd(){
        super();
        viewer  = JmolViewer.allocateViewer(this, adapter);
    }  

    public String executeCmd(String cmd){
        String script;
        /*
        if (viewer.isScriptExecuting()) 
            System.out.println("viewer is executing");
         * 
         */
        /* Wait for the comnplete execution of I/O commands */
        if ((cmd.startsWith("write")) || (cmd.startsWith("load")) || 
                (cmd.startsWith("load append")) || (cmd.startsWith("undo")) ||
                (cmd.startsWith("redo")) || (cmd.startsWith("compare"))){
        //System.out.println("RMSD " + cmd);    
            /* Just an alias for evalString */
            script = viewer.scriptWait(cmd);
        } else
            script = viewer.script(cmd);
        /* Debug */
        if(!script.equals("pending")){
            System.out.println("Command: " + cmd);
            /* System.out.println("Script results: " + script); */
        }
        return script;
    }//executeCmd
    
    public void loadProtein(String path){
        executeCmd("load append " + path);
    }
    
    /* Clear all */
    public void clearDisplay(){
        executeCmd("zap;");
    }//clearDisplay
    
    public String[] computeRmds(int i){
        String rmsd;
        rmsd = executeCmd("compare {1.1} {2." + i + "}; ");
        System.out.println("****** RMSD ****** " + rmsd);
        //Pattern pattern = Pattern.compile("\"RMSD:(.*?)\\A\"");
        Pattern pattern = Pattern.compile("RMSD(.*?)Angstroms");
        Matcher matcher = pattern.matcher(rmsd);
        String[] result = {"",""};
        while (matcher.find()) {
             result = matcher.group(1).split("-->");
        }
        return result;
    }
}