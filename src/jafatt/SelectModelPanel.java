package jafatt;

import java.awt.BorderLayout;
import java.awt.GridBagLayout;
import java.awt.GridBagConstraints;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import javax.swing.JFrame;
import javax.swing.ButtonGroup;
import javax.swing.JPanel;
import javax.swing.JRadioButton;
import javax.swing.JScrollPane;
import javax.swing.ScrollPaneConstants;
import javax.swing.BorderFactory;
import java.awt.event.WindowEvent;
import java.awt.event.WindowAdapter;
import java.awt.Container;
import javax.swing.ImageIcon;

public class SelectModelPanel extends JFrame implements Runnable {

    private UserFrame view;
    private ViewPanel viewPanel;
    private JPanel inPanel, buttonPanel;
    private OpButton ok, cancel;
    JScrollPane scrollFrame;

    public SelectModelPanel(UserFrame view, int modelNumber, int displayedModel) {
        super("Select Model Panel");

        /*
        Toolkit t = Toolkit.getDefaultToolkit();
        Dimension screensize = t.getScreenSize();
        double widthFrame = (screensize.getWidth() * 20.0) / 120.0;
        double heighFrame = (screensize.getHeight() * 45.0) / 150.0;
         * 
         */

        this.view = view;
        
        setIconImage(new ImageIcon(view.frameIcon).getImage());
        
        inPanel = new JPanel(new BorderLayout());
        buttonPanel = new JPanel();

        /* Setup layout */
        //setLocation((int) (view.getX() + (int) (view.getWidth() / 2)),
        //        (int) (view.getY() + (int) (view.getHeight() / 2)));
        //setPreferredSize(new Dimension((int) widthFrame, (int) heighFrame));
        setResizable(true);

        /* Internal panel */
        viewPanel = new ViewPanel(modelNumber,displayedModel);
        viewPanel.setBorder(BorderFactory.createTitledBorder("Select a model to transfer"));
        scrollFrame = new JScrollPane(viewPanel, 
                ScrollPaneConstants.VERTICAL_SCROLLBAR_AS_NEEDED,
                ScrollPaneConstants.HORIZONTAL_SCROLLBAR_AS_NEEDED);
        //scrollFrame.getVerticalScrollBar().setUnitIncrement(10);
        
        ok = new OpButton("Ok", "") {
            @Override
            public void buttonEvent(ActionEvent evt){
                buttonEvent(true);
            }
        };
        
        cancel = new OpButton("Cancel", "") {
            @Override
            public void buttonEvent(ActionEvent evt){
                buttonEvent(false);
            }
        };
        
        buttonPanel.add(ok);
        buttonPanel.add(cancel);

        /* Add panels */
        inPanel.add(scrollFrame, BorderLayout.CENTER);
        inPanel.add(buttonPanel, BorderLayout.SOUTH);
        
        addWindowListener(new WindowAdapter() {

            @Override
            public void windowClosing(WindowEvent e) {
                setVisible(false);
                dispose();
            }
        });
        
    }//setup
    
    private void buttonEvent(boolean ok){
       
        if(ok){
            setAlwaysOnTop(false);
            int model = ((OutputPanel)view.getPanel(Defs.OUTPUT)).molViewPanel.getDisplayedModel();
            String proteinPath = Defs.PROTEINS_PATH +
                    view.getModel().idProteinCode + ".model.pdb";
            ((OutputPanel)view.getPanel(Defs.OUTPUT)).executeCmd("select */" + (model-1) + ";");
            ((OutputPanel)view.getPanel(Defs.OUTPUT)).executeCmd(
                    "write pdb \""  + proteinPath + "\"; ");
            view.initProteinLoaded(Defs.EXTRACTION);
            boolean loaded = view.getController().loadStructure(proteinPath, Defs.EXTRACTION, true);
            if(loaded){
                view.proteinLoaded();
            }else{
                /* Stop the progress bar */
                view.proteinNotLoaded();
            }
            setVisible(false);
            dispose();
        }else{
            setAlwaysOnTop(false);
            setVisible(false);
            dispose();
            
        }
    }

    @Override
    public void run() {
        Container ct = getContentPane();
        ct.add(inPanel);
        update();
        setVisible(true);
        setAlwaysOnTop(true);
    }//run
    
    private void update(){
        pack();
        setLocationRelativeTo(view);
    }

    private class ViewPanel extends JPanel{// implements ActionListener {

        /* Group for selecting buttons in an exclusive mode */
        ButtonGroup group;

        /* Buttons */
        public ViewPanel(int mn, int dm) {

            //setLayout(new GridLayout((int) (mn / 3) + 1, 3));
            setLayout(new GridBagLayout());
            GridBagConstraints c = new GridBagConstraints();

            JRadioButton[] model = new JRadioButton[mn];

            /* Buttons's group and buttons */
            group = new ButtonGroup();
            c.fill = GridBagConstraints.HORIZONTAL;
            c.gridx = 0;
            c.gridy = 0;

            for (int i = 0; i < mn; i++) {
                if (i == dm - 1) {
                    model[i] = new JRadioButton("Model " + (i + 1), true);
                } else {
                    model[i] = new JRadioButton("Model " + (i + 1));
                }
                model[i].setActionCommand("model " + i + ";");
                model[i].addActionListener(new ActionListener(){
                    @Override
                    public void actionPerformed(ActionEvent ae) {
                        /* Set views */
                        ((OutputPanel) view.getPanel(Defs.OUTPUT)).executeCmd(ae.getActionCommand());
                        view.computeRmsd();
                    }
                });
                group.add(model[i]);
                add(model[i],c);
                if(c.gridx == 2){
                    c.gridy++;
                    c.gridx = 0;
                }else{
                    c.gridx++;
                }       
            }
        }
    }
    
}//ViewOptionPanel
