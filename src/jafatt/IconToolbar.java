package jafatt;

import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.Image;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.image.BufferedImage;
import java.awt.image.ImageObserver;
import java.awt.image.RescaleOp;
import javax.swing.AbstractButton;
import javax.swing.BorderFactory;
import javax.swing.ImageIcon;
import javax.swing.JButton;
import javax.swing.JToolBar;
import javax.swing.border.Border;

public class IconToolbar extends JToolBar{
    private final int LOADTARGET = 0;
    private final int DOWNLOAD = 1;
    private final int COCOS = 2;
    private final int NVIDIA = 3;
    private final int MEASURE = 4;
    private final int RESET = 5;
    private final int SHOWCON = 6;
   
    private UserFrameActions ufa;
    private JButton[] buttons;
    
    public IconToolbar( UserFrameActions ufa, int totalMoveButtons, 
            JButton[] buttons, String[] descriptions, String[] icons, 
            String[] info ){
        addSeparator();
        this.buttons = buttons;
        setFloatable(false);
        setBorder(BorderFactory.createMatteBorder(0,0,1,0, Defs.INNERCOLOR));
        setBackground(Defs.INNERCOLOR);
        addSeparator(new Dimension(7,15));
        
        /* Create the toolbar with proper actions */
        createToolbar(ufa, totalMoveButtons, buttons, descriptions, icons, info);
    }
    
    /* Create a JButton for the toolbar with associate an icon */
    public JButton createButton( String text, String iconName, String info, 
            int i ) {
        ImageIcon icon = new ImageIcon(iconName);
        JButton button = new JButton(icon);
        
        /* Set the button qualities */
        button.setToolTipText(info);
        button.setBorderPainted(false);
        button.setContentAreaFilled(false);
        button.setVerticalTextPosition( AbstractButton.BOTTOM );
        button.setHorizontalTextPosition( AbstractButton.CENTER );
        button.setBackground(Defs.BUTTONBACKGROUNDCOLOR);
        button.setIconTextGap(0);
        
        /* Set the borders for the buttons */
        Border border1 = BorderFactory.createMatteBorder(0,3,0,3,Defs.BACKGROUNDCOLOR);
        Border border2 =BorderFactory.createMatteBorder(1,0,1,0,Defs.BUTTONBACKGROUNDCOLOR);
        Border border3 =BorderFactory.createEmptyBorder(5,0,2,0);
        Border compoundBorder1 =BorderFactory.createCompoundBorder( border1, border2 );
        Border border =BorderFactory.createCompoundBorder( compoundBorder1, border3 );
        button.setBorder(border);
        
        /* Set the icon image */
        Image img = icon.getImage();
        ImageIcon pressedIcon = new ImageIcon(createSelectedIcon( img ) );
        button.setPressedIcon(pressedIcon);
        return button;
    }//createButton
    
    /* Effects when a user click on a button */
    private Image createSelectedIcon( Image image ) {
        ImageObserver observer = new ImageObserver() {
            @Override
            public boolean imageUpdate(Image image, int infoflags, int x, int y, int width, int height) {
                return (infoflags & ImageObserver.ALLBITS) == 0;}
        };
        int width = image.getWidth(observer);
        int height = image.getHeight(observer);
        if(width <= 0) width = 1;
        if(height <= 0) height = 1;
        int w2 = image.getWidth(null);
        int h2 = image.getHeight(null);
        if(w2 <= 0) w2 = 1;
        if(h2 <= 0) h2 = 1;
        
        /* Rescale operation */
        BufferedImage img1 = new BufferedImage(w2,h2,BufferedImage.TYPE_INT_RGB);
        Graphics g = img1.getGraphics();
        g.drawImage(image, 0, 0, null);
        BufferedImage img2 = new BufferedImage( width, height,BufferedImage.TYPE_INT_RGB);
        RescaleOp rop = new RescaleOp(0.8f, -1.0f, null);
        rop.filter(img1, img2);
        return img2;
    }//createSelectedIcon
    
    /* Create the toolbar with the object IconToolbar */
    private void createToolbar( UserFrameActions ufa, int totalMoveButtons, 
            JButton[] buttons, String[] descriptions, String[] icons, 
            String[] info ){
      this.ufa = ufa;
      int totalButtons = buttons.length;
      //for (int i = 0; i < totalButtons; i++) {
      for (int i = 0; i < 4; i++) {
         final int id1 = i;
         //final int id2 = i - totalMoveButtons;
         final int id2 = i - 0;
         
         /* create the button */
         buttons[ i ] = createButton( descriptions[i], icons[i], info[i], i );
         
         /* Set the action for each button */
         if(i <= totalMoveButtons)
             buttons[ i ].addActionListener( new ActionListener() {
                @Override
                public void actionPerformed(ActionEvent e) {
                        toolActions( id1 );
                }
             });
         else
             buttons[ i ].addActionListener( new ActionListener() {
                @Override
                public void actionPerformed(ActionEvent e) {
                        moveView( id2 );
                 }
             });
       }
      buttons[COCOS].setEnabled(false);
      buttons[NVIDIA].setEnabled(false);
    }//createToolbar
    
    public void enableCocos(){
        buttons[COCOS].setEnabled(true);   
        buttons[NVIDIA].setEnabled(true);   
    }
    
    /* Actions for the buttons over the show buttons */
    private void toolActions( int buttonIndex ){
        switch(buttonIndex){
            case LOADTARGET:
                ufa.loadTargetEvent();
                break;
            case DOWNLOAD:
                ufa.downloadProtein();
                break;
            case COCOS:
                ufa.runCocos();
                break;
            case NVIDIA:
                ufa.runCocosGPU();
                break;
            case MEASURE:
                ufa.measureEvent();
                break;
            case RESET:
                ufa.resetEvent();
                break;
            case SHOWCON:
                ufa.showEvent();
            default:
                break;
        }
   }//toolActions
    
    /* Set the actions for the view buttons */
   private void moveView(int view){
        //if(ctr != null){
           //ctr.setAxis(view);
           //ctr.moveView();
        //}
   }//moveView
    
}
