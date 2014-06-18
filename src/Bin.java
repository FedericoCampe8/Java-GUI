package jafatt;

import java.awt.Dimension;
import java.awt.Toolkit;
import java.awt.Rectangle;
import javax.swing.JPanel;
import javax.swing.JSplitPane;
import javax.swing.JFrame;


public class Bin extends JSplitPane{
    
    JSplitPane splitPaneLeft = new JSplitPane(JSplitPane.HORIZONTAL_SPLIT);
    JSplitPane splitPaneRight = new JSplitPane(JSplitPane.HORIZONTAL_SPLIT);
    Dimension dimBin;
    int def = -1;
    JFrame view;
    
    public Bin(JFrame view, JPanel targetPanel, JPanel extractionPanel,
            JPanel assemblingPanel, JPanel outputPanel){
        super( JSplitPane.VERTICAL_SPLIT, false );
        setup(view, targetPanel, extractionPanel, assemblingPanel, outputPanel);
    }
    
    /* Set the layout */
    private void setup(JFrame view, JPanel targetPanel, JPanel extractionPanel,
            JPanel assemblingPanel, JPanel outputPanel){
        //JSplitPane bottomPane = new JSplitPane(JSplitPane.HORIZONTAL_SPLIT, false, 
        //        extractionPanel, assemblingPanel);
        
        this.view = view;
        
        Rectangle r = view.getBounds();
        long w = r.width;
        
        splitPaneLeft.setLeftComponent( extractionPanel );
        splitPaneLeft.setRightComponent( assemblingPanel );
        splitPaneRight.setLeftComponent( splitPaneLeft );
        splitPaneRight.setRightComponent( outputPanel );

        // put splitPaneRight onto a single panel
        //JPanel bottomPane = new JPanel();
        //bottomPane.add( splitPaneRight );
        dimBin = Toolkit.getDefaultToolkit().getScreenSize();
        /* Set the layout of the bottom panel*/
        //dimBin = this.getSize();
        //bottomPane.setOneTouchExpandable(true);
        splitPaneRight.setOneTouchExpandable(true);
        //bottomPane.setDividerLocation((int) dimBin.getWidth() * 1/3);
        
        //splitPaneRight.setDividerLocation((int) dimBin.getWidth() * 1/3);
        
        
        //splitPaneLeft.setDividerLocation((int) dimBin.getWidth() * 1/3);
        //splitPaneRight.setDividerLocation((int) dimBin.getWidth() * 2/3);
        
        splitPaneLeft.setDividerLocation((int) w * 1/3);
        splitPaneRight.setDividerLocation((int) w * 2/3);
        
        
        this.setOneTouchExpandable(true);
        this.setDividerLocation((int) dimBin.getHeight() * 1/9);
        
        /* Add the components */
        addImpl( targetPanel, JSplitPane.TOP, 0 );
        //addImpl( bottomPane, JSplitPane.BOTTOM, 1 );
        addImpl( splitPaneRight, JSplitPane.BOTTOM, 1 );
  
    }//setup
    
    public void setSeparators (int panel, long w){
        
        dimBin = Toolkit.getDefaultToolkit().getScreenSize();
        //Rectangle r = view.getBounds();
        //long w = r.width;
        if (panel == def){
            //splitPaneLeft.setDividerLocation((int) dimBin.getWidth() * 1/3);
            //splitPaneRight.setDividerLocation((int) dimBin.getWidth() * 2/3);
            splitPaneLeft.setDividerLocation((int) w * 1/3);
            splitPaneRight.setDividerLocation((int) w * 2/3);
            def = -1;
            return;
        }

        if (panel == Defs.EXTRACTION){
            //splitPaneLeft.setDividerLocation((int) dimBin.getWidth() * 1/6);
            //splitPaneRight.setDividerLocation((int) dimBin.getWidth() * 5/6);
            splitPaneLeft.setDividerLocation((int) w * 4/6);
            splitPaneRight.setDividerLocation((int) w * 5/6);
            def = panel;
        }
        if (panel == Defs.ASSEMBLING){
            //splitPaneLeft.setDividerLocation((int) dimBin.getWidth() * 4/6);
            //splitPaneRight.setDividerLocation((int) dimBin.getWidth() * 5/6);
            splitPaneLeft.setDividerLocation((int) w * 1/6);
            splitPaneRight.setDividerLocation((int) w * 5/6);
            def = panel;
        }
        if (panel == Defs.OUTPUT){
            //splitPaneLeft.setDividerLocation((int) dimBin.getWidth() * 1/6);
            //splitPaneRight.setDividerLocation((int) dimBin.getWidth() * 2/6);
            splitPaneLeft.setDividerLocation((int) w * 1/6);
            splitPaneRight.setDividerLocation((int) w * 2/6);
            def = panel;
        }
        
    }
    
    public void update(long w){
        splitPaneLeft.setDividerLocation((int) w * 1/3);
        splitPaneRight.setDividerLocation((int) w * 2/3);
    }
    
}//Bin
