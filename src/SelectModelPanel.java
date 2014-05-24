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
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JRadioButton;
import javax.swing.JScrollPane;

public class SelectModelPanel extends JFrame implements Runnable {

    private UserFrame view;
    private ViewPanel viewPanel;
    private JPanel inPanel, buttonPanel;
    private OpButton ok, cancel;
    private String title;
    JScrollPane scrollFrame;

    public SelectModelPanel(UserFrame view, int modelNumber, int displayedModel) {
        super("Select Model Panel");

        Toolkit t = Toolkit.getDefaultToolkit();
        Dimension screensize = t.getScreenSize();
        double widthFrame = (screensize.getWidth() * 20.0) / 120.0;
        double heighFrame = (screensize.getHeight() * 45.0) / 150.0;

        this.view = view;
        inPanel = new JPanel(new BorderLayout());
        buttonPanel = new JPanel();
        title = "Select a model to transfer";

        /* Setup layout */
        setLocation((int) (view.getX() + (int) (view.getWidth() / 2)),
                (int) (view.getY() + (int) (view.getHeight() / 2)));
        setPreferredSize(new Dimension((int) widthFrame, (int) heighFrame));
        setResizable(true);

        /* Internal panel */
        viewPanel = new ViewPanel(modelNumber,displayedModel);
        scrollFrame = new JScrollPane(viewPanel);
        scrollFrame.getVerticalScrollBar().setUnitIncrement(10);
        
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
        inPanel.add(new JLabel(title), BorderLayout.NORTH);
        inPanel.add(scrollFrame, BorderLayout.CENTER);
        inPanel.add(buttonPanel, BorderLayout.SOUTH);
        
    }//setup
    
    private void buttonEvent(boolean ok){
        if(ok){
            int model = ((OutputPanel)view.getPanel(Defs.OUTPUT)).molViewPanel.getDisplayedModel();
            String proteinPath = Defs.path_prot +
                    view.getModel().idProteinCode + ".model.pdb";
            ((OutputPanel)view.getPanel(Defs.OUTPUT)).executeCmd("select */" + model + ";");
            ((OutputPanel)view.getPanel(Defs.OUTPUT)).executeCmd(
                    "write pdb \""  + proteinPath + "\"; ");
            view.getController().loadStructure(proteinPath, Defs.EXTRACTION, true);
            setVisible(false);
            dispose();
        }else{
            setVisible(false);
            dispose();
            
        }
    }

    @Override
    public void run() {
        pack();
        Container ct = getContentPane();
        ct.add(inPanel);
        setVisible(true);
    }//run

    private class ViewPanel extends JPanel implements ActionListener {

        /* Group for selecting buttons in an exclusive mode */
        ButtonGroup group;

        /* Buttons */
        public ViewPanel(int mn, int dm) {

            setLayout(new GridLayout((int) (mn / 3) + 1, 3));

            JRadioButton[] model = new JRadioButton[mn];

            /* Buttons's group and buttons */
            group = new ButtonGroup();

            for (int i = 0; i < mn; i++) {
                if (i == dm - 1) {
                    model[i] = new JRadioButton("Model " + (i + 1), true);
                } else {
                    model[i] = new JRadioButton("Model " + (i + 1));
                }
                model[i].setActionCommand("model " + i + ";");
                model[i].addActionListener(this);
                group.add(model[i]);
                add(model[i]);
            }

        }

        @Override
        public void actionPerformed(ActionEvent ae) {

            /* Set views */
            ((OutputPanel) view.getPanel(Defs.OUTPUT)).executeCmd(ae.getActionCommand());
        }
    }//ViewPanel
}//ViewOptionPanel
