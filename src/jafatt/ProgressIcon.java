package jafatt;

import javax.swing.ImageIcon;
import javax.swing.JLabel;

public class ProgressIcon extends JLabel{
    private String progressIcon;
    private String dir;
       
    public ProgressIcon( String str, String dir ){
        super(str);
        setup(dir);
    }
       
    private void setup(String dir){
        this.dir = dir;
        progressIcon = dir + "status-offline.png";
        ImageIcon icon = new ImageIcon(progressIcon);
        setIcon(icon);
    }//setup
       
    /* Invoked when a Panel is activated */
    public void enableProgress(){
        progressIcon = dir + "status.png";
        ImageIcon icon = new ImageIcon(progressIcon);
        setIcon(icon);
    }//enableProgress
}//ProgressIcon
