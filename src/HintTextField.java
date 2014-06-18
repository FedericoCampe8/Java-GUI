package jafatt;

import java.awt.Color;
import java.awt.Font;
import java.awt.event.FocusEvent;
import java.awt.event.FocusListener;
import javax.swing.JTextField;
 
public class HintTextField extends JTextField implements FocusListener {
 
    private String hint;
    Font hintFont;
    Font normalFont;
 
    public HintTextField() {
        super(0);
        //super.addFocusListener(this);
    }

    public HintTextField(int columns) {
        super(columns);
    }

        
    public HintTextField(final String hint) {
        super(hint);
        this.hint = hint;
        setup();
    }
    
    public HintTextField(final String hint, int columns){
        super(hint,columns);
        this.hint = hint;
        setup();
    } 
     
    private void setup(){
        this.hintFont = new Font("Segoe UI", Font.ITALIC, 12);
        this.normalFont = new Font("Segoe UI", Font.PLAIN, 12);
        super.setForeground(Color.GRAY);
        super.setFont(hintFont);
        super.addFocusListener(this);
    }
    
    public void addListener(){
        super.addFocusListener(this);
    }
    
    public void removeListener(){
        super.removeFocusListener(this);
    }
 
    @Override
    public void focusGained(FocusEvent e) {
        if(this.getText().isEmpty()) {
            super.setText("");
            super.setForeground(Color.BLACK);
            super.setFont(normalFont);
        }
    }
 
    @Override
    public void focusLost(FocusEvent e) {
        if(this.getText().isEmpty()) {
            super.setText(hint);
            //Font hintFont = new Font("Segoe UI", Font.ITALIC, 12);
            super.setForeground(Color.GRAY);
            super.setFont(hintFont);
        }
    }    
 
    @Override
    public String getText() {
        String typed = super.getText();
        return typed.equals(hint) ? "" : typed;
    }
    
    public void setHintText(String text){
        if(text.equals("")){
            this.setText(hint);
            super.setForeground(Color.GRAY);
            super.setFont(hintFont);
        }else{
            setText(text);
            super.setForeground(Color.BLACK);
            super.setFont(normalFont);
        }
        
    }
}