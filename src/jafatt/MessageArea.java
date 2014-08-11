package jafatt;

import java.awt.Color;
import javax.swing.JTextArea;

public class MessageArea extends JTextArea{ 
    
    String s;
    
    public MessageArea(){
        setEditable(false);
        //setLineWrap(true);
        //setWrapStyleWord(true);
        
        /* Old style :) */
        setBackground(Color.black);
        this.setForeground(Color.GREEN);
        
    }
    
    public MessageArea(int x, int y) {
        super(x,y);
        setEditable(false);
        
        /* Old style :) */
        setBackground(Color.black);
        this.setForeground(Color.GREEN); 
    }
    
    /* Some methods for printing and clear the area */
    public void write (String msg){
        append(msg);
    }//write
    
    public void write (String msg, boolean start){
        if(start)
            append("> " + msg);
    }//write

    public void writeln(){
        append("\n");
    }//writeln

    public void writeln(String msg){
        append("> " + msg + "\n");
                
    }//writeln

    public void writeln(String msg, boolean start){
        if(start)
            append("> " + msg + "\n");
        else
            append( msg + "\n");
    }//writeln

    public void clear(){
        setText("");
    }//clear
    
}//MessageArea
