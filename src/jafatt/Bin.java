package jafatt;

import java.awt.Dimension;
import java.awt.Rectangle;
import javax.swing.JPanel;
import javax.swing.JSplitPane;

public class Bin extends JSplitPane {

    JSplitPane splitPaneLeft;
    JSplitPane splitPaneRight;
    Dimension dimBin;
    private int currentPanel = -1;
    private int left = 1;
    private int right = 2;
    private int base = 3;
    UserFrame view;

    public Bin(UserFrame view, JPanel targetPanel, JPanel extractionPanel,
            JPanel assemblingPanel, JPanel outputPanel) {
        super(JSplitPane.VERTICAL_SPLIT, false);
        setup(view, targetPanel, extractionPanel, assemblingPanel, outputPanel);
    }

    /* Set the layout */
    private void setup(UserFrame view, JPanel targetPanel, JPanel extractionPanel,
            JPanel assemblingPanel, JPanel outputPanel) {

        this.view = view;

        splitPaneLeft = new JSplitPane(JSplitPane.HORIZONTAL_SPLIT);
        splitPaneLeft.setOneTouchExpandable(true);
        splitPaneLeft.setContinuousLayout(true);
        splitPaneRight = new JSplitPane(JSplitPane.HORIZONTAL_SPLIT);
        splitPaneRight.setOneTouchExpandable(true);
        splitPaneRight.setContinuousLayout(true);
        Rectangle r = view.getBounds();
        long w = r.width;
        long h = r.height;

        splitPaneLeft.setLeftComponent(extractionPanel);
        splitPaneLeft.setRightComponent(assemblingPanel);
        splitPaneRight.setLeftComponent(splitPaneLeft);
        splitPaneRight.setRightComponent(outputPanel);
        
        splitPaneLeft.setDividerLocation((int) w * left / base);
        splitPaneRight.setDividerLocation((int) w * right / base);

        setOneTouchExpandable(true);
        setDividerLocation((int) h * 1 / 9);

        /* Add the components */
        addImpl(targetPanel, JSplitPane.TOP, 0);
        addImpl(splitPaneRight, JSplitPane.BOTTOM, 1);

    }//setup

    public void setSeparators(int panel, long width) {

        if (panel == currentPanel) {
            left = 1;
            right = 2;
            base = 3;
            currentPanel = -1;
            splitPaneLeft.setDividerLocation((int) width * left / base);
            splitPaneRight.setDividerLocation((int) width * right / base);
            getRootPane().revalidate();
            return;
        }

        if (panel == Defs.EXTRACTION) {
            left = 4;
            right = 5;
            base = 6;
            currentPanel = panel;
        }
        if (panel == Defs.ASSEMBLING){
            left = 1;
            right = 5;
            base = 6;
            currentPanel = panel;
        }
        if (panel == Defs.OUTPUT) {
            left = 1;
            right = 2;
            base = 6;
            currentPanel = panel;
        }
        
        splitPaneLeft.setDividerLocation((int) width * left / base);
        splitPaneRight.setDividerLocation((int) width * right / base);
        getRootPane().revalidate();

    }

    public void update(long width, long height) {
        
        splitPaneLeft.setDividerLocation((int) width * left / base);
        splitPaneRight.setDividerLocation((int) width * right / base);
        setDividerLocation((int) height * 1 / 9);
        getRootPane().revalidate();
    }

}//Bin
