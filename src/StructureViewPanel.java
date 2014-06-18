package jafatt;

import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.Rectangle;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import javax.swing.JPanel;
import javax.swing.Timer;
import org.biojava.bio.structure.Structure;
import org.biojava.bio.structure.StructureImpl;
import org.jmol.api.JmolAdapter;
import org.jmol.api.JmolViewer;
import org.jmol.popup.JmolPopup;

public class StructureViewPanel extends JPanel{
    
    private final Dimension currentSize = new Dimension();
    private final Rectangle  rectClip = new Rectangle();
    private Controller ctr = null;
    private Timer clickTimer;
    
    /* Viewer used to render client molecules */
    private JmolViewer viewer;
    
    /* API used by the JmolViewer to read external files and fetch atom properties necessary for rendering */
    private JmolAdapter adapter;
    private JmolPopup jmolpopup;
    
    /* Input structure */
    private Structure structure;
    
    public StructureViewPanel(){
        super();
        initJmolInstance();
        initJmolDisplay();
    }
    
    /* Init the Jmol instance from the viewer */
    private void initJmolInstance(){
        String scriptingOn = "set scriptQueue on; ";
        viewer  = JmolViewer.allocateViewer(this, adapter);
        
        /* Add the mouse listener for drag and drop the fragments on the panel */
        addMouseListener(new MouseListener() {
            private static final boolean LEFT = true;
            private static final boolean RIGHT = false;
            @Override
            public void mouseClicked(MouseEvent me) {
                if (ctr != null)
                    if(ctr.currentPanel == Defs.ASSEMBLING)
                        clickAction(me);
            }
            @Override
            public void mousePressed(MouseEvent e) {
                if (ctr != null)
                    if(ctr.currentPanel == Defs.ASSEMBLING){
                        
                        /* Left button */
                        if(e.getButton() == java.awt.event.MouseEvent.BUTTON1)
                            ctr.moveAllBindedFragments(LEFT);
                        else
                            
                        /* Right button */
                            ctr.moveAllBindedFragments(RIGHT);
                    }
            }
            @Override
            public void mouseReleased(MouseEvent e) {}
            @Override
            public void mouseEntered(MouseEvent e) {}
            @Override
            public void mouseExited(MouseEvent e) {}
        });
        
        /* Set the queue for the scripting strings 
         * this is important to make Jmol thread -safe !!
         */
        viewer.evalString(scriptingOn);
    }//initJmolInstance
    
    /* Set the initial view */
    private void initJmolDisplay(){
        executeCmd("select *; spacefill off;");
        executeCmd("set measurements angstroms; ");
        executeCmd("set dynamicMeasurements ON; ");
        executeCmd("frank off; ");
    }//initJmolDisplay
    
    /* Get the current structure */
    public Structure getStructure(){
        return structure;
    }//getStructure
    
    public int getModelNumber(){
        
        int modelNumber = Integer.parseInt(
                (String)viewer.getProperty("string","modelInfo.modelCount",null)
                );
        return modelNumber;   
    }    
    
    public int getDisplayedModel(){
        
        String[] model = ((String)viewer.getProperty(
                "string","animationInfo.displayModelNumber",null)).split("\\.");
        int modelNumber = Integer.parseInt(model[1]);
        return modelNumber;
        
    }
    
    public String getAxesCoordinates(){
        
        String origin = (String)viewer.getProperty("string","centerInfo[0]",null);
        return origin;
        
    }
    
    public String getMeasurements(){
        int i = 1;
        String model;
        String cms = "";
        model = ((String)viewer.getProperty("string","measurementInfo["+i+"]",null));
        
        while(!model.equals("")){
            String aa1 = "measurementinfo["+i+"].atoms[0].info";
            model = ((String)viewer.getProperty("string",aa1,null));
            String resno1 = model.split("(?<=\\D)(?=\\d)|(?<=\\d)(?=\\D)")[1];
            String aa2 = "measurementinfo["+i+"].atoms[1].info";
            model = ((String)viewer.getProperty("string",aa2,null));
            String resno2 = model.split("(?<=\\D)(?=\\d)|(?<=\\d)(?=\\D)")[1];
            String distance = ((String)viewer.getProperty("string","measurementinfo["+i+"].value",null));
            /* Debug
            System.out.println(resno1 + " " + resno2 + " " + distance);
             */
            i++;
            model = ((String)viewer.getProperty("string","measurementInfo["+i+"]",null));
            cms = cms + Defs.tab1 + "--distance-geq " + resno1 +" "+ resno2 + " " + distance + " \\\n";
        }
        //String[] measures = viewer.getProperty(str"measurementinfo");
        return cms;
    }
    
    /* Set the structure to be displayed */
    public void setStructure(Structure structure){
        if (structure == null){
            structure = new StructureImpl();
            initJmolDisplay();
        }
        this.structure = structure;
        
        /* If something is going wrong with Jmol drop it and get a new instance */
        if (viewer.isScriptExecuting()) initJmolInstance();
        viewer.evalString("exit");
        String pdbstr = structure.toPDB();
        viewer.openStringInline(pdbstr);
    }//setStructure
    
    /* Send a RASMOL like command to Jmol
     * @param command - a String containing a RASMOL like command. e.g. "select protein; cartoon on;"
     */
    public void executeCmd(String cmd){
        String script;
        if (viewer.isScriptExecuting()) 
            System.out.println("viewer is executing");
        /* Wait for the comnplete execution of I/O commands */
        if ((cmd.startsWith("write")) || (cmd.startsWith("load")) || 
                (cmd.startsWith("load append")) || (cmd.startsWith("undo")) ||
                (cmd.startsWith("redo")) || (cmd.startsWith("model"))){
            
            /* Just an alias for evalString */
            script = viewer.scriptWait(cmd);
        } else
            script = viewer.script(cmd);
        /* Debug */
        if(!script.equals("pending")){
            System.out.println("Command: " + cmd);
            /* System.out.println("Script results: " + script); */
        }
    }//executeCmd
    
    /* Clear all */
    public void clearDisplay(){
        executeCmd("zap;");
        setStructure(new StructureImpl());
    }//clearDisplay
    
    /* Paint Jmol */
    @Override
    public void paint(Graphics g) {
        getSize(currentSize);
        g.getClipBounds(rectClip);
        try{
            viewer.renderScreenImage(g, currentSize, rectClip); 
        }catch (NullPointerException npe){
            System.out.println("JMolViewer Rendering Error, catched!");
        }
    }//paint
    
    /* Set the Controller instance */
    public void setController(Controller ctr){
        this.ctr = ctr;
    }//setController
    
    /* It permits to (de)select fragments 
     * Note:
     * It doesn't work on Mac without a mouse
     * TO DO
     */
    private void clickAction(MouseEvent me){
        
        /* Delay in msec before processing events */
        final int clickDelay=1000;
        if ((me.getClickCount() == 1)&&(java.awt.event.MouseEvent.BUTTON1 == me.getButton())) {
            clickTimer = new Timer(clickDelay, new ActionListener(){
                @Override
		public void actionPerformed(ActionEvent e){}
	});
            
            /* After expiring once, stop the timer */
            clickTimer.setRepeats(false); //
            clickTimer.start();
       }else if (me.getClickCount() == 2){
           
           /* The single click will not be processed */
           clickTimer.stop(); //
           ctr.deselect();
        } 
    }//clickAction

}//StructureViewPanel
