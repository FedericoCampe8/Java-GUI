package jafatt;

import java.awt.BorderLayout;
import java.awt.event.AdjustmentEvent;
import javax.swing.JPanel;
import javax.swing.JScrollPane;


public class InfoPanel extends JPanel{
    
    private MessageArea infoArea;
    private JScrollPane scroll;
    private JPanel south;
    
    public InfoPanel(Boolean b){
        setup(b);
    }
    
    /* Set the panel */
    private void setup(Boolean b){
        infoArea = new MessageArea(6, 20);
        south = new JPanel();
        south.setBackground(Defs.INNERCOLOR);
        setLayout(new BorderLayout());
        add(infoArea, BorderLayout.CENTER);
        add(south, BorderLayout.SOUTH);
        
        /* Add the scroll bar and set auto-scroll */
        scroll = new JScrollPane(infoArea);
        /*
        scroll.getVerticalScrollBar().addAdjustmentListener(new java.awt.event.AdjustmentListener() {
            @Override
            public void adjustmentValueChanged(AdjustmentEvent e) {
               infoArea.select(infoArea.getHeight() + 1000, 0);
            }
        });*/

        add(scroll);
        /* Print some info */
        if(b){
            printStringLn( "Constraint Tool" );
            printStringLn( "Version 1.1" );
            printStringLn( "Operating system name: " + System.getProperty("os.name") );
            printStringLn( "Operating system version: " + System.getProperty("os.version") );
            printStringLn( "Operating system architecture: " + System.getProperty("os.arch") );
            printStringLn( "Current working directory " + System.getProperty("user.dir") );
            printStringLn();
        }
    }//setup
    
    /* Methods for printing */
    public void printString(String str){
        infoArea.write(str);
        infoArea.setCaretPosition(infoArea.getDocument().getLength());
    }//printString

    public void printStringLn(String str){
        infoArea.writeln(str);
        infoArea.setCaretPosition(infoArea.getDocument().getLength());
    }//printString
    
    public void printStringLn(String str, boolean b){
        infoArea.writeln(str, b);
        infoArea.setCaretPosition(infoArea.getDocument().getLength());
    }//printStringLn

    public void printStringLn(){
        infoArea.writeln();
        infoArea.setCaretPosition(infoArea.getDocument().getLength());
    }//printString
    
}//InfoPanel
