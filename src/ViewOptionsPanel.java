package jafatt;

import java.awt.BorderLayout;
import java.awt.Container;
import java.awt.Dimension;
import java.awt.GridLayout;
import java.awt.Toolkit;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import javax.swing.ButtonGroup;
import javax.swing.JFrame;
import javax.swing.JPanel;
import javax.swing.JRadioButton;
import javax.swing.BorderFactory;

public class ViewOptionsPanel extends JFrame implements Runnable{
    
    private UserFrame view;
    private Controller ctr;
    private ViewPanel viewPanel;
    private int panel;
    
    private JPanel inPanel;
    private String title;
    
    public ViewOptionsPanel(UserFrame view, int panel){
        super("View Options Panel");
        setup(view, panel);
    }
    
    /* Setup */
    private void setup(UserFrame view, int panel){
        Toolkit t = Toolkit.getDefaultToolkit();
        Dimension screensize = t.getScreenSize();
        double widthFrame = (screensize.getWidth() * 25.0) / 100.0;
        double heighFrame = (screensize.getHeight() * 45.0) / 100.0;
        
        this.view = view;
        this.panel = panel;
        ctr = view.getController();
        inPanel = new JPanel();
        title = "Select a view for the proten ";
        if(panel == Defs.EXTRACTION)
            title += "in the Extraction Panel";
        if(panel == Defs.ASSEMBLING)
            title += "in the Assembling Panel";
        if(panel == Defs.OUTPUT)
            title += "in the Output Panel";        
        
        /* Setup layout */
        setLocation((int)(view.getX() + (int)(view.getWidth()/2)),
                    (int)(view.getY() + (int)(view.getHeight()/2)));
        setPreferredSize(new Dimension((int)widthFrame, (int)heighFrame));
        setResizable(true);
        
        inPanel.setLayout(new BorderLayout());
        inPanel.setBorder(BorderFactory.createTitledBorder(title));
        
        /* Internal panel */
        viewPanel = new ViewPanel(ctr);
        
        /* Add panels */
        //inPanel.add(new JLabel(title), BorderLayout.NORTH);
        inPanel.add(viewPanel, BorderLayout.CENTER); 
    }//setup

    @Override
    public void run() {
        pack();
        setLocationRelativeTo(null);
        Container ct = getContentPane();
        ct.add(inPanel);
        setVisible(true);
    }//run
    
    private class ViewPanel extends JPanel implements ActionListener{
        
        Controller ctr;
        
        /* Group for selecting buttons in an exclusive mode */
        ButtonGroup group;
        
        /* Buttons */
        JRadioButton normalView;
        JRadioButton spacefillView;
        JRadioButton dotsView;
        JRadioButton dotswireframeView;
        JRadioButton backboneView;
        JRadioButton traceView;
        JRadioButton ribbonView;
        JRadioButton cartoonView;
        JRadioButton strandsView;
        JRadioButton ballandstickView;
        JRadioButton meshribbonsView;
        
        
        public ViewPanel(Controller ctr){
            setup(ctr);
        }
        
        /* Set the layout and components */
        private void setup(Controller ctr){
            setLayout(new GridLayout(11, 1));
            
            /* Set the controller */
            this.ctr = ctr;
            
            /* Buttons's group and buttons */
            group = new ButtonGroup();
            
            /* Buttons */
            normalView = new JRadioButton(Defs.NORMAL, true);
            normalView.setActionCommand(Defs.NORMAL);
            spacefillView = new JRadioButton(Defs.SPACEFILL);
            spacefillView.setActionCommand(Defs.SPACEFILL);
            dotsView = new JRadioButton(Defs.DOTS);
            dotsView.setActionCommand(Defs.DOTS);
            dotswireframeView = new JRadioButton(Defs.DOTSWIREFRAME);
            dotswireframeView.setActionCommand(Defs.DOTSWIREFRAME);
            backboneView = new JRadioButton(Defs.BACKBONE);
            backboneView.setActionCommand(Defs.BACKBONE);
            traceView = new JRadioButton(Defs.TRACE);
            traceView.setActionCommand(Defs.TRACE);
            ribbonView = new JRadioButton(Defs.RIBBON);
            ribbonView.setActionCommand(Defs.RIBBON);
            cartoonView = new JRadioButton(Defs.CARTOON);
            cartoonView.setActionCommand(Defs.CARTOON);
            strandsView = new JRadioButton(Defs.STRANDS);
            strandsView.setActionCommand(Defs.STRANDS);
            ballandstickView = new JRadioButton(Defs.BALLANDSTICK);
            ballandstickView.setActionCommand(Defs.BALLANDSTICK);
            meshribbonsView = new JRadioButton(Defs.MESHRIBBONS);
            meshribbonsView.setActionCommand(Defs.MESHRIBBONS);
            
            /* Add listeners */
            normalView.addActionListener(this);
            spacefillView.addActionListener(this);
            dotsView.addActionListener(this);
            dotswireframeView.addActionListener(this);
            backboneView.addActionListener(this);
            traceView.addActionListener(this);
            ribbonView.addActionListener(this);
            cartoonView.addActionListener(this);
            strandsView.addActionListener(this);
            ballandstickView.addActionListener(this);
            meshribbonsView.addActionListener(this);
        
            /* Add buttons to the group and to the panel */
            group.add(normalView);
            group.add(spacefillView);
            group.add(dotsView);
            group.add(dotswireframeView);
            group.add(backboneView);
            group.add(traceView);
            group.add(ribbonView);
            group.add(cartoonView);
            group.add(strandsView);
            group.add(ballandstickView);
            group.add(meshribbonsView);
            
            add(normalView);
            add(spacefillView);
            add(dotsView);
            add(dotswireframeView);
            add(backboneView);
            add(traceView);
            add(ribbonView);
            add(cartoonView);
            add(strandsView);
            add(ballandstickView);
            add(meshribbonsView);
        }

        @Override
        public void actionPerformed(ActionEvent ae){
            
            String switchViewString;
            switchViewString = Utilities.switchViewString(ae.getActionCommand());
            ((MolPanel) view.getPanel(panel)).switchView(switchViewString);
            
        }
    }//ViewPanel
    
}//ViewOptionPanel
